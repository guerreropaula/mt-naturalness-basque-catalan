"""Configurable rewards for P5 GRPO machine translation."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import sacrebleu
import torch

from src.utils.errors import PipelineError

logger = logging.getLogger(__name__)


class QualityEstimationScorer(Protocol):
    def score(self, sources: list[str], candidates: list[str]) -> list[float]: ...


class NaturalnessScorer(Protocol):
    def score(self, target_texts: list[str], batch_size: int = 64) -> list[float]: ...


class ReferenceBasedQualityScorer(Protocol):
    def score(
        self, sources: list[str], candidates: list[str], references: list[str]
    ) -> list[float]: ...


@dataclass(frozen=True)
class RewardWeights:
    """Non-negative weights for reward components.

    The length component is already a non-positive penalty, so its positive
    weight subtracts length deviations from the final reward.
    """

    chrfpp: float = 0.0
    bleu: float = 0.0
    cometkiwi: float = 0.0
    comet: float = 0.0
    self_bleu: float = 0.0
    length: float = 0.0
    reference_likeness: float = 0.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "RewardWeights":
        return cls(
            chrfpp=float(values.get("chrfpp", 0.0)),
            bleu=float(values.get("bleu", 0.0)),
            cometkiwi=float(values.get("cometkiwi", 0.0)),
            comet=float(values.get("comet", 0.0)),
            self_bleu=float(values.get("self_bleu", 0.0)),
            length=float(values.get("length", 0.0)),
            reference_likeness=float(values.get("reference_likeness", 0.0)),
        )

    def validate(self) -> None:
        values = self.as_dict()
        if any(value < 0.0 for value in values.values()):
            raise PipelineError(f"Reward weights must be non-negative: {values}")
        total = sum(values.values())
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise PipelineError(f"Reward weights must sum to 1.0, found {total}: {values}")

    def as_dict(self) -> dict[str, float]:
        return {
            "chrfpp": self.chrfpp,
            "bleu": self.bleu,
            "cometkiwi": self.cometkiwi,
            "comet": self.comet,
            "self_bleu": self.self_bleu,
            "length": self.length,
            "reference_likeness": self.reference_likeness,
        }


@dataclass(frozen=True)
class RewardBatch:
    """Per-completion reward components retained for trainer logging."""

    total: list[float]
    chrfpp: list[float]
    bleu: list[float]
    cometkiwi: list[float]
    comet: list[float]
    self_bleu_diversity: list[float]
    length_penalty: list[float]
    reference_likeness: list[float]

    def component_values(self) -> dict[str, list[float]]:
        return {
            "chrfpp": self.chrfpp,
            "bleu": self.bleu,
            "cometkiwi": self.cometkiwi,
            "comet": self.comet,
            "self_bleu_diversity": self.self_bleu_diversity,
            "length_penalty": self.length_penalty,
            "reference_likeness": self.reference_likeness,
            "total": self.total,
        }


def _clip_unit_interval(value: float) -> float:
    if not math.isfinite(value):
        raise PipelineError(f"Reward component must be finite, found {value!r}")
    return min(max(float(value), 0.0), 1.0)


def completion_text(completion: Any) -> str:
    """Extract plain completion text from TRL standard or conversational formats."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Mapping):
        content = completion.get("content")
        if content is not None:
            return str(content)
    if isinstance(completion, Sequence) and not isinstance(completion, (bytes, bytearray)):
        for message in reversed(completion):
            if isinstance(message, Mapping) and message.get("content") is not None:
                return str(message["content"])
    raise PipelineError(f"Cannot extract text from completion of type {type(completion).__name__}")


def sentence_chrfpp(candidate: str, reference: str) -> float:
    """Return sentence-level chrF++ as a unit-interval reward."""
    score = sacrebleu.sentence_chrf(str(candidate), [str(reference)], word_order=2).score
    return _clip_unit_interval(score / 100.0)


def sentence_bleu_unit_interval(candidate: str, reference: str) -> float:
    """Return smoothed sentence BLEU in [0, 1] for Self-BLEU calculation."""
    score = sacrebleu.sentence_bleu(
        str(candidate), [str(reference)], use_effective_order=True
    ).score
    return _clip_unit_interval(score / 100.0)


def self_bleu_diversity(candidates: Sequence[str], group_size: int) -> list[float]:
    """Return 1 - mean pairwise sentence BLEU for every contiguous GRPO group."""
    if group_size < 2:
        raise PipelineError("Self-BLEU diversity requires group_size >= 2.")
    if len(candidates) % group_size:
        raise PipelineError(
            f"Expected a whole number of GRPO groups of {group_size}, got {len(candidates)} completions."
        )

    reward: list[float] = []
    for start in range(0, len(candidates), group_size):
        group = [str(value) for value in candidates[start : start + group_size]]
        for index, candidate in enumerate(group):
            pair_scores = [
                1.0
                if candidate.strip() == other.strip()
                else sentence_bleu_unit_interval(candidate, other)
                for other_index, other in enumerate(group)
                if other_index != index
            ]
            reward.append(_clip_unit_interval(1.0 - (sum(pair_scores) / len(pair_scores))))
    return reward


def length_log_ratio_penalty(candidate: str, reference: str) -> float:
    """Return the paper-style bounded log-ratio length penalty in [-2, 0]."""
    generated_length = len(str(candidate).split())
    reference_length = len(str(reference).split())
    if generated_length == 0 or reference_length == 0:
        return -2.0
    return max(-abs(math.log(generated_length / reference_length)), -2.0)


def is_lexically_valid_translation(candidate: str, reference: str) -> bool:
    """Reject non-lexical output when the reference contains letters."""
    reference_has_letters = any(character.isalpha() for character in str(reference))
    candidate_has_letters = any(character.isalpha() for character in str(candidate))
    return not reference_has_letters or candidate_has_letters


class COMETScorer:
    """Reusable sentence-level COMET or COMETKiwi scorer."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 4,
        gpus: int = 1,
        reference_based: bool = False,
    ) -> None:
        self.model_name = str(model_name)
        self.batch_size = int(batch_size)
        self.gpus = int(gpus)
        self.reference_based = bool(reference_based)
        self._model: Any | None = None
        self._device: Any | None = None

    @property
    def label(self) -> str:
        return "COMET" if self.reference_based else "COMETKiwi"

    @staticmethod
    def _move_to_device(value: Any, device: Any) -> Any:
        if hasattr(value, "to"):
            return value.to(device)
        if isinstance(value, Mapping):
            return {key: COMETScorer._move_to_device(item, device) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(COMETScorer._move_to_device(item, device) for item in value)
        if isinstance(value, list):
            return [COMETScorer._move_to_device(item, device) for item in value]
        return value

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                import torch
                from comet import download_model, load_from_checkpoint
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise PipelineError(
                    f"{self.label} reward requires unbabel-comet. Install the GRPO extra."
                ) from exc
            logger.info("Loading %s reward model: %s", self.label, self.model_name)
            self._model = load_from_checkpoint(download_model(self.model_name))
            if self.gpus > 0:
                if not torch.cuda.is_available():
                    raise PipelineError(
                        f"{self.label} was configured for GPU scoring, but CUDA is unavailable."
                    )
                self._device = torch.device("cuda")
            else:
                self._device = torch.device("cpu")
            self._model.to(self._device)
            self._model.eval()
        return self._model

    def _score_records(self, records: list[dict[str, str]]) -> list[float]:
        model = self._load_model()
        if self._device is None:  # pragma: no cover - set while loading
            raise PipelineError(f"{self.label} did not initialise an inference device.")
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - project dependency
            raise PipelineError(f"{self.label} reward requires PyTorch.") from exc

        scores: list[float] = []
        for start in range(0, len(records), self.batch_size):
            batch = model.prepare_sample(records[start : start + self.batch_size], stage="predict")
            batch = self._move_to_device(batch, self._device)
            with torch.inference_mode():
                outputs = model.predict_step(batch)
            batch_scores = getattr(outputs, "scores", None)
            if batch_scores is None and isinstance(outputs, Mapping):
                batch_scores = outputs.get("scores")
            if batch_scores is None:
                raise PipelineError(f"{self.label} did not return sentence scores.")
            if hasattr(batch_scores, "detach"):
                batch_scores = batch_scores.detach().float().cpu().tolist()
            scores.extend(float(score) for score in batch_scores)

        if len(scores) != len(records):
            raise PipelineError(f"{self.label} did not return one score per candidate.")
        return [_clip_unit_interval(score) for score in scores]

    def score(
        self,
        sources: list[str],
        candidates: list[str],
        references: list[str] | None = None,
    ) -> list[float]:
        if len(sources) != len(candidates):
            raise PipelineError(f"{self.label} inputs must have the same length.")
        records = [
            {"src": source, "mt": candidate}
            for source, candidate in zip(sources, candidates, strict=True)
        ]
        if self.reference_based:
            if references is None or len(references) != len(records):
                raise PipelineError("COMET requires one reference per candidate.")
            for record, reference in zip(records, references, strict=True):
                record["ref"] = reference
        return self._score_records(records)


class TranslationGRPOReward:
    """Compute the weighted P5 reward used by TRL GRPOTrainer."""

    def __init__(
        self,
        weights: RewardWeights,
        group_size: int,
        cometkiwi_scorer: QualityEstimationScorer | None = None,
        comet_scorer: ReferenceBasedQualityScorer | None = None,
        classifier_scorer: NaturalnessScorer | None = None,
        classifier_batch_size: int = 64,
        empty_candidate_reward: float = -1.0,
    ) -> None:
        self.__name__ = "translation_grpo_reward"
        weights.validate()
        if group_size < 2:
            raise PipelineError("GRPO reward requires at least two sampled completions per prompt.")
        if weights.cometkiwi > 0.0 and cometkiwi_scorer is None:
            raise PipelineError("A positive COMETKiwi weight requires a COMETKiwi scorer.")
        if weights.comet > 0.0 and comet_scorer is None:
            raise PipelineError("A positive COMET weight requires a reference-based COMET scorer.")
        if weights.reference_likeness > 0.0 and classifier_scorer is None:
            raise PipelineError(
                "A positive reference-likeness weight requires a classifier scorer."
            )
        self.weights = weights
        self.group_size = int(group_size)
        if not math.isfinite(empty_candidate_reward):
            raise PipelineError("empty_candidate_reward must be finite.")
        self.cometkiwi_scorer = cometkiwi_scorer
        self.comet_scorer = comet_scorer
        self.classifier_scorer = classifier_scorer
        self.classifier_batch_size = int(classifier_batch_size)
        # EOS-only output is invalid, not a separate reward component.
        self.empty_candidate_reward = float(empty_candidate_reward)

    def score_batch(
        self,
        sources: Sequence[str],
        candidates: Sequence[str],
        references: Sequence[str],
    ) -> RewardBatch:
        source_values = [str(value) for value in sources]
        candidate_values = [str(value) for value in candidates]
        reference_values = [str(value) for value in references]
        size = len(candidate_values)
        if not (len(source_values) == size == len(reference_values)):
            raise PipelineError("Sources, candidates, and references must have the same length.")
        if size % self.group_size:
            raise PipelineError(
                f"Reward batch size {size} is not divisible by group_size {self.group_size}."
            )
        for start in range(0, size, self.group_size):
            if len(set(source_values[start : start + self.group_size])) != 1:
                raise PipelineError("Each contiguous GRPO group must share one source sentence.")
            if len(set(reference_values[start : start + self.group_size])) != 1:
                raise PipelineError(
                    "Each contiguous GRPO group must share one reference translation."
                )

        chrfpp = [
            sentence_chrfpp(candidate, reference)
            for candidate, reference in zip(candidate_values, reference_values, strict=True)
        ]
        bleu = [
            sentence_bleu_unit_interval(candidate, reference)
            for candidate, reference in zip(candidate_values, reference_values, strict=True)
        ]
        cometkiwi = (
            self.cometkiwi_scorer.score(source_values, candidate_values)
            if self.weights.cometkiwi > 0.0 and self.cometkiwi_scorer is not None
            else [0.0] * size
        )
        if len(cometkiwi) != size:
            raise PipelineError("COMETKiwi scorer returned an unexpected number of scores.")
        cometkiwi = [_clip_unit_interval(score) for score in cometkiwi]
        comet = (
            self.comet_scorer.score(source_values, candidate_values, reference_values)
            if self.weights.comet > 0.0 and self.comet_scorer is not None
            else [0.0] * size
        )
        if len(comet) != size:
            raise PipelineError("COMET scorer returned an unexpected number of scores.")
        comet = [_clip_unit_interval(score) for score in comet]
        diversity = (
            self_bleu_diversity(candidate_values, self.group_size)
            if self.weights.self_bleu > 0.0
            else [0.0] * size
        )
        length = [
            length_log_ratio_penalty(candidate, reference)
            for candidate, reference in zip(candidate_values, reference_values, strict=True)
        ]
        reference_likeness = (
            self.classifier_scorer.score(candidate_values, batch_size=self.classifier_batch_size)
            if self.weights.reference_likeness > 0.0 and self.classifier_scorer is not None
            else [0.0] * size
        )
        if len(reference_likeness) != size:
            raise PipelineError("Classifier scorer returned an unexpected number of scores.")
        reference_likeness = [_clip_unit_interval(score) for score in reference_likeness]

        total = [
            self.weights.chrfpp * chrf
            + self.weights.bleu * bleu_score
            + self.weights.cometkiwi * kiwi_score
            + self.weights.comet * comet_score
            + self.weights.self_bleu * diverse
            + self.weights.length * length_penalty
            + self.weights.reference_likeness * natural
            for chrf, bleu_score, kiwi_score, comet_score, diverse, length_penalty, natural in zip(
                chrfpp,
                bleu,
                cometkiwi,
                comet,
                diversity,
                length,
                reference_likeness,
                strict=True,
            )
        ]
        lexical_validity = [
            is_lexically_valid_translation(candidate, reference)
            for candidate, reference in zip(candidate_values, reference_values, strict=True)
        ]
        # Reject punctuation-only output when the reference contains words.
        total = [
            reward if candidate.strip() and valid else self.empty_candidate_reward
            for candidate, valid, reward in zip(
                candidate_values, lexical_validity, total, strict=True
            )
        ]
        return RewardBatch(
            total, chrfpp, bleu, cometkiwi, comet, diversity, length, reference_likeness
        )

    def __call__(
        self,
        completions: Sequence[Any],
        source: Sequence[str] | None = None,
        reference: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        sources = source if source is not None else kwargs.get("sources")
        references = reference if reference is not None else kwargs.get("references")
        if sources is None or references is None:
            raise PipelineError("TRL reward data must include source and reference columns.")
        scored = self._score_distributed_batch(
            list(sources),
            [completion_text(value) for value in completions],
            list(references),
        )
        log_extra = kwargs.get("log_extra")
        if callable(log_extra):
            for name, values in scored.component_values().items():
                log_extra(f"p5_reward_{name}", values)
        log_metric = kwargs.get("log_metric")
        if callable(log_metric):
            for name, values in scored.component_values().items():
                log_metric(f"p5_reward/{name}", sum(values) / len(values))
        return scored.total

    def _score_distributed_batch(
        self,
        sources: Sequence[str],
        candidates: Sequence[str],
        references: Sequence[str],
    ) -> RewardBatch:
        """Score full GRPO groups when TRL splits candidates across ranks."""
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return self.score_batch(sources, candidates, references)

        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        payload = (list(sources), list(candidates), list(references))
        gathered: list[Any] = [None] * world_size
        torch.distributed.all_gather_object(gathered, payload)
        lengths = [len(item[1]) for item in gathered]
        all_sources = [value for item in gathered for value in item[0]]
        all_candidates = [value for item in gathered for value in item[1]]
        all_references = [value for item in gathered for value in item[2]]

        result: list[Any] = [None]
        if rank == 0:
            try:
                batch = self.score_batch(all_sources, all_candidates, all_references)
                result[0] = {"components": batch.component_values()}
            except Exception as exc:
                result[0] = {"error": f"{type(exc).__name__}: {exc}"}
        torch.distributed.broadcast_object_list(result, src=0)
        if "error" in result[0]:
            raise PipelineError(result[0]["error"])

        start = sum(lengths[:rank])
        end = start + lengths[rank]
        values = result[0]["components"]
        return RewardBatch(
            total=list(values["total"][start:end]),
            chrfpp=list(values["chrfpp"][start:end]),
            bleu=list(values["bleu"][start:end]),
            cometkiwi=list(values["cometkiwi"][start:end]),
            comet=list(values["comet"][start:end]),
            self_bleu_diversity=list(values["self_bleu_diversity"][start:end]),
            length_penalty=list(values["length_penalty"][start:end]),
            reference_likeness=list(values["reference_likeness"][start:end]),
        )


def build_translation_grpo_reward(
    config: Mapping[str, Any],
    dataset_key: str,
    *,
    ablation: str | None = None,
    cometkiwi_scorer: QualityEstimationScorer | None = None,
    comet_scorer: ReferenceBasedQualityScorer | None = None,
    classifier_scorer: NaturalnessScorer | None = None,
) -> TranslationGRPOReward:
    """Build the configured reward and load only the scorers it uses."""
    root = config.get("grpo", config)
    reward_config = root["reward"]
    selected_ablation = str(ablation or reward_config["active_ablation"])
    ablations = reward_config["ablations"]
    if selected_ablation not in ablations:
        raise PipelineError(
            f"Unknown GRPO ablation '{selected_ablation}'. Available: {sorted(ablations)}"
        )
    weights = RewardWeights.from_mapping(ablations[selected_ablation]["weights"])
    kiwi = cometkiwi_scorer
    if weights.cometkiwi > 0.0 and kiwi is None:
        cometkiwi_config = reward_config["cometkiwi"]
        kiwi = COMETScorer(
            cometkiwi_config["model"],
            batch_size=int(cometkiwi_config["batch_size"]),
            gpus=int(cometkiwi_config["gpus"]),
            reference_based=False,
        )
    comet = comet_scorer
    if weights.comet > 0.0 and comet is None:
        comet_config = reward_config["comet"]
        comet = COMETScorer(
            comet_config["model"],
            batch_size=int(comet_config["batch_size"]),
            gpus=int(comet_config["gpus"]),
            reference_based=True,
        )
    naturalness = classifier_scorer
    if weights.reference_likeness > 0.0 and naturalness is None:
        ablation_config = ablations[selected_ablation]
        contrastive_datasets = ablation_config.get("contrastive_lm_datasets")
        if contrastive_datasets is not None:
            if dataset_key not in set(contrastive_datasets):
                raise PipelineError(
                    f"Ablation '{selected_ablation}' does not support {dataset_key}; "
                    f"configured datasets: {list(contrastive_datasets)}."
                )
            from src.contrastive_lm.reward import ContrastiveHTMTNaturalnessScorer

            naturalness = ContrastiveHTMTNaturalnessScorer(
                dataset_key,
                ablation_config.get("contrastive_lm_config", "configs/contrastive_lm.yaml"),
            )
        else:
            # A3v2 uses five-sentence chunks; A3v5 uses sentence-level examples.
            model_dirs = ablation_config.get(
                "classifier_model_dirs", reward_config["classifier"]["model_dirs"]
            )
            if dataset_key not in model_dirs:
                raise PipelineError(
                    f"No classifier model configured for {dataset_key} in ablation '{selected_ablation}'."
                )
            from src.classifiers.reward import TargetSideReferenceLikenessScorer

            naturalness = TargetSideReferenceLikenessScorer(
                model_dirs[dataset_key],
                max_length=int(reward_config["classifier"]["max_length"]),
            )
    return TranslationGRPOReward(
        weights=weights,
        group_size=int(reward_config["group_size"]),
        cometkiwi_scorer=kiwi,
        comet_scorer=comet,
        classifier_scorer=naturalness,
        classifier_batch_size=int(reward_config["classifier"]["batch_size"]),
        empty_candidate_reward=float(reward_config.get("empty_candidate_reward", -1.0)),
    )
