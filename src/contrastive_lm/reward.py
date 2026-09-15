"""Calibrated contrastive HT-versus-MT scorer for GRPO naturalness rewards."""

from __future__ import annotations

import math
from pathlib import Path
import torch
from transformers import AutoTokenizer

from src.contrastive_lm.data import load_labeled_texts
from src.contrastive_lm.score import average_token_logprobs
from src.contrastive_lm.train import load_base_model
from src.utils.config import load_contrastive_lm_config
from src.utils.hf_auth import get_hf_token


class ContrastiveLMRewardError(RuntimeError):
    """Raised when a calibrated contrastive naturalness reward is unavailable."""


def calibration_parameters(ht_scores: list[float], mt_scores: list[float]) -> tuple[float, float]:
    """Return a dev-set midpoint and logistic scale for HT-minus-MT margins."""
    if not ht_scores or not mt_scores:
        raise ContrastiveLMRewardError("Contrastive calibration requires both HT and MT dev scores.")
    ht_mean = sum(ht_scores) / len(ht_scores)
    mt_mean = sum(mt_scores) / len(mt_scores)
    margin = ht_mean - mt_mean
    if not math.isfinite(margin) or margin <= 0.0:
        raise ContrastiveLMRewardError(
            "Contrastive HT scores must exceed MT scores on held-out dev chunks; "
            f"found HT={ht_mean:.6f}, MT={mt_mean:.6f}."
        )
    # Map the two held-out class means approximately to sigmoid(-2) and sigmoid(2).
    return (ht_mean + mt_mean) / 2.0, 4.0 / margin


def normalize_htmt_score(raw_score: float, *, center: float, scale: float) -> float:
    """Map a finite HT-minus-MT log-probability margin to [0, 1]."""
    if not math.isfinite(raw_score):
        raise ContrastiveLMRewardError(f"Contrastive score must be finite, found {raw_score!r}")
    value = scale * (raw_score - center)
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


class ContrastiveHTMTNaturalnessScorer:
    """Score text as HT-like using calibrated avg_logprob_HT - avg_logprob_MT."""

    def __init__(self, dataset_key: str, config_path: str | Path) -> None:
        config = load_contrastive_lm_config(config_path)["contrastive_lm"]
        if dataset_key not in config["languages"]:
            raise ContrastiveLMRewardError(f"No contrastive-LM configuration for {dataset_key}.")
        language = config["languages"][dataset_key]
        model_config = config["model"]
        self.dataset_key = dataset_key
        self.batch_size = int(config["scoring"]["batch_size"])
        self.max_length = int(model_config["max_length"])
        adapter_root = Path(config["adapter_root"]) / dataset_key
        ht_adapter = adapter_root / "ht"
        mt_adapter = adapter_root / "mt"
        if not ht_adapter.exists() or not mt_adapter.exists():
            raise ContrastiveLMRewardError(
                f"Both contrastive adapters are required under {adapter_root}."
            )
        try:
            from peft import PeftModel
        except ImportError as exc:  # pragma: no cover - project dependency
            raise ContrastiveLMRewardError("peft is required for contrastive GRPO rewards.") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(ht_adapter, token=get_hf_token(), use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            raise ContrastiveLMRewardError("Contrastive tokenizer must provide an EOS or PAD token.")
        self.tokenizer.padding_side = "right"
        base_model = load_base_model(
            str(language["base_model"]), bool(model_config["load_in_4bit"]), training=False
        )
        self.model = PeftModel.from_pretrained(base_model, ht_adapter, adapter_name="ht")
        self.model.load_adapter(mt_adapter, adapter_name="mt")
        self.model.eval()

        data_root = config["data_root"]
        ht_dev = load_labeled_texts(data_root, dataset_key, "dev", 1)
        mt_dev = load_labeled_texts(data_root, dataset_key, "dev", 0)
        ht_margins = self._raw_scores(ht_dev)
        mt_margins = self._raw_scores(mt_dev)
        self.center, self.scale = calibration_parameters(ht_margins, mt_margins)
        self.calibration = {
            "split": "dev",
            "ht_examples": len(ht_margins),
            "mt_examples": len(mt_margins),
            "center": self.center,
            "scale": self.scale,
            "ht_mean_margin": sum(ht_margins) / len(ht_margins),
            "mt_mean_margin": sum(mt_margins) / len(mt_margins),
        }

    def _adapter_logprobs(self, adapter_name: str, texts: list[str]) -> list[float | None]:
        self.model.set_adapter(adapter_name)
        return average_token_logprobs(
            self.model,
            self.tokenizer,
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )

    def _raw_scores(self, texts: list[str]) -> list[float]:
        ht_scores = self._adapter_logprobs("ht", texts)
        mt_scores = self._adapter_logprobs("mt", texts)
        margins: list[float] = []
        for ht_value, mt_value in zip(ht_scores, mt_scores, strict=True):
            if ht_value is None or mt_value is None:
                continue
            margins.append(float(ht_value - mt_value))
        if not margins:
            raise ContrastiveLMRewardError("No valid contrastive next-token scores were produced.")
        return margins

    @torch.inference_mode()
    def score(self, target_texts: list[str], batch_size: int = 64) -> list[float]:
        """Return calibrated HT-like probabilities for target-side candidates."""
        del batch_size  # The scorer uses the calibrated configuration's batch size.
        if not target_texts:
            return []
        ht_scores = self._adapter_logprobs("ht", [str(text) for text in target_texts])
        mt_scores = self._adapter_logprobs("mt", [str(text) for text in target_texts])
        result: list[float] = []
        for ht_value, mt_value in zip(ht_scores, mt_scores, strict=True):
            if ht_value is None or mt_value is None:
                result.append(0.5)
            else:
                result.append(
                    normalize_htmt_score(float(ht_value - mt_value), center=self.center, scale=self.scale)
                )
        return result
