"""Automatic MT and translationese-oriented evaluation metrics."""

from __future__ import annotations

import math
import re
import gc
import json
import logging
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import sacrebleu
from lexical_diversity import lex_div as ld

_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)
DEFAULT_XCOMET_XXL_MODEL = "Unbabel/XCOMET-XXL"
DEFAULT_COMETKIWI_MODEL = "Unbabel/wmt23-cometkiwi-da-xl"
DEFAULT_METRICX_MODEL = "google/metricx-24-hybrid-xl-v2p6"
DEFAULT_METRICX_TOKENIZER = "google/mt5-xl"
DEFAULT_METRICX_PYTHON = os.environ.get("METRICX_PYTHON", "python")
DEFAULT_METRICX_REPOSITORY = os.environ.get("METRICX_REPOSITORY", "third_party/metricx")
_COMET_MODEL_CACHE: dict[str, Any] = {}
logger = logging.getLogger(__name__)


def tokenize_text(text: Any) -> list[str]:
    """Tokenize text for lightweight corpus-level metric computation."""
    if text is None or pd.isna(text):
        return []
    return _TOKEN_RE.findall(str(text).lower())


def compute_bleu(predictions: list[str], references: list[str]) -> float | None:
    """Compute corpus BLEU with sacreBLEU."""
    if not predictions or not references:
        return None
    return float(sacrebleu.corpus_bleu(predictions, [references], use_effective_order=True).score)


def compute_chrf_pp(predictions: list[str], references: list[str]) -> float | None:
    """Compute corpus chrF++ using character and word n-grams."""
    if not predictions or not references:
        return None
    return float(sacrebleu.corpus_chrf(predictions, [references], word_order=2).score)


def compute_ter(predictions: list[str], references: list[str]) -> float | None:
    """Compute corpus TER; lower values indicate fewer required edits."""
    if not predictions or not references:
        return None
    return float(sacrebleu.corpus_ter(predictions, [references]).score)


def compute_ttr(tokens: list[str]) -> float | None:
    if not tokens:
        return None
    return len(set(tokens)) / len(tokens)


def compute_mattr(tokens: list[str], window_size: int = 50) -> float | None:
    """Compute corpus-level MATTR from an already-tokenized sequence."""
    if len(tokens) < window_size:
        return None
    return float(ld.mattr(tokens, window_length=window_size))


def compute_mtld(tokens: list[str], reliable_min_tokens: int = 100) -> float | None:
    """Compute corpus-level MTLD over a sufficiently long token sequence."""
    if len(tokens) < reliable_min_tokens:
        return None
    return float(ld.mtld(tokens))

def compute_yules_k(tokens: list[str]) -> float | None:
    """Compute Yule's K lexical concentration measure."""
    if not tokens:
        return None
    frequencies = Counter(tokens)
    total_tokens = len(tokens)
    frequency_of_frequencies = Counter(frequencies.values())
    m2 = sum((frequency**2) * count for frequency, count in frequency_of_frequencies.items())
    return 10000.0 * (m2 - total_tokens) / (total_tokens * total_tokens)


def compute_yules_i(tokens: list[str]) -> float | None:
    """Compute Yule's I (10,000 / K), where larger values indicate diversity."""
    yules_k = compute_yules_k(tokens)
    if yules_k in (None, 0.0):
        return None
    return 10000.0 / yules_k


def compute_shannon_diversity(tokens: list[str]) -> float | None:
    """Compute Shannon diversity over surface word forms."""
    if not tokens:
        return None
    counts = Counter(tokens)
    total = len(tokens)
    return -sum((count / total) * math.log(count / total) for count in counts.values())


def compute_simpson_diversity(tokens: list[str]) -> float | None:
    """Compute Simpson concentration over surface word forms as in evaluation.py."""
    if not tokens:
        return None
    counts = Counter(tokens)
    total = len(tokens)
    return sum((count / total) ** 2 for count in counts.values())


def compute_inverse_simpson_diversity(tokens: list[str]) -> float | None:
    """Compute the inverse Simpson index over surface word forms."""
    simpson = compute_simpson_diversity(tokens)
    if simpson in (None, 0.0):
        return None
    return 1.0 / simpson


def compute_unique_word_forms(tokens: list[str]) -> int:
    """Count unique surface word forms."""
    return len(set(tokens))


def compute_repeated_token_rate(tokens: list[str]) -> float | None:
    if not tokens:
        return None
    return 1.0 - (len(set(tokens)) / len(tokens))


def _ngram_counts(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))


def compute_repeated_ngram_rate(tokens: list[str], n: int) -> float | None:
    if len(tokens) < n:
        return None
    counts = _ngram_counts(tokens, n)
    total_ngrams = sum(counts.values())
    repeated_occurrences = sum(count - 1 for count in counts.values() if count > 1)
    return repeated_occurrences / total_ngrams if total_ngrams else None


def compute_consecutive_repetition_rate(tokens: list[str]) -> float | None:
    if len(tokens) < 2:
        return None
    repeated_positions = sum(
        1 for index in range(1, len(tokens)) if tokens[index] == tokens[index - 1]
    )
    return repeated_positions / (len(tokens) - 1)


def compute_content_word_repetition_rate(tokens: list[str]) -> float | None:
    """Heuristic proxy until POS-based content/function tagging is added."""
    if not tokens:
        return None
    content_like_tokens = [token for token in tokens if token.isalpha() and len(token) > 3]
    if not content_like_tokens:
        return None
    return compute_repeated_token_rate(content_like_tokens)


def compute_lexical_density_proxy(tokens: list[str]) -> float | None:
    if not tokens:
        return None
    content_like_tokens = [token for token in tokens if token.isalpha() and len(token) > 3]
    return len(content_like_tokens) / len(tokens)


def compute_function_to_content_ratio_proxy(tokens: list[str]) -> float | None:
    if not tokens:
        return None
    content_like_tokens = [token for token in tokens if token.isalpha() and len(token) > 3]
    function_like_tokens = len(tokens) - len(content_like_tokens)
    if not content_like_tokens:
        return None
    return function_like_tokens / len(content_like_tokens)


def compute_content_to_function_ratio_proxy(tokens: list[str]) -> float | None:
    ratio = compute_function_to_content_ratio_proxy(tokens)
    if ratio in (None, 0.0):
        return None
    return 1.0 / ratio


def compute_function_word_proportion_proxy(tokens: list[str]) -> float | None:
    if not tokens:
        return None
    content_like_tokens = [token for token in tokens if token.isalpha() and len(token) > 3]
    function_like_tokens = len(tokens) - len(content_like_tokens)
    return function_like_tokens / len(tokens)


def _build_frequency_reference(tokens: list[str]) -> Counter[str]:
    """Build a corpus-relative token frequency reference."""
    return Counter(tokens)


def _token_frequency_rank_map(frequency_counter: Counter[str]) -> dict[str, int]:
    sorted_tokens = sorted(
        frequency_counter.items(),
        key=lambda item: (-item[1], item[0]),
    )
    return {token: rank for rank, (token, _) in enumerate(sorted_tokens, start=1)}


def compute_lexical_frequency_profile(
    tokens: list[str],
    frequency_counter: Counter[str],
) -> dict[str, float | None]:
    """Compute the paper-style B1/B2/B3 lexical frequency profile.

    The supplied counter must come from a fixed natural target corpus. Prediction
    tokens are never used to construct the frequency list.
    """
    empty = {
        "b1_top_1000": None,
        "b2_1001_2000": None,
        "b3_beyond_2000": None,
        "beyond_2000": None,
    }
    if not tokens or not frequency_counter:
        return empty

    rank_map = _token_frequency_rank_map(frequency_counter)
    counts = Counter()
    for token in tokens:
        rank = rank_map.get(token)
        if rank is not None and rank <= 1000:
            counts["b1_top_1000"] += 1
        elif rank is not None and rank <= 2000:
            counts["b2_1001_2000"] += 1
        else:
            counts["b3_beyond_2000"] += 1
    total = len(tokens)
    b3 = counts["b3_beyond_2000"] / total
    return {
        "b1_top_1000": counts["b1_top_1000"] / total,
        "b2_1001_2000": counts["b2_1001_2000"] / total,
        "b3_beyond_2000": b3,
        "beyond_2000": b3,
    }


def compute_target_lexical_metrics(
    texts: list[str],
    frequency_reference_texts: list[str],
    *,
    mtld_reliable_min_tokens: int = 100,
) -> dict[str, Any]:
    """Compute corpus-level lexical metrics for one target-language corpus."""
    tokens = [token for text in texts for token in tokenize_text(text)]
    frequency_tokens = [
        token for text in frequency_reference_texts for token in tokenize_text(text)
    ]
    mtld = compute_mtld(tokens, reliable_min_tokens=mtld_reliable_min_tokens)
    mattr_50 = compute_mattr(tokens, window_size=50)
    return {
        "token_count": len(tokens),
        "type_count": len(set(tokens)),
        "ttr": compute_ttr(tokens),
        "yules_k": compute_yules_k(tokens),
        "yules_i": compute_yules_i(tokens),
        "mtld": mtld,
        "mattr_50": mattr_50,
        "mtld_status": ("insufficient_tokens" if mtld is None else "ok"),
        "lexical_frequency_profile": compute_lexical_frequency_profile(
            tokens, _build_frequency_reference(frequency_tokens)
        ),
    }


def _cosine_distance_from_uniform(counts: Counter[str], support: list[str]) -> float | None:
    if len(support) <= 1:
        return 0.0
    observed = [counts.get(item, 0) for item in support]
    total = sum(observed)
    if total == 0:
        return None
    observed_probs = [value / total for value in observed]
    uniform_value = 1.0 / len(support)
    uniform_probs = [uniform_value] * len(support)
    dot_product = sum(left * right for left, right in zip(observed_probs, uniform_probs))
    observed_norm = math.sqrt(sum(value * value for value in observed_probs))
    uniform_norm = math.sqrt(sum(value * value for value in uniform_probs))
    if observed_norm == 0.0 or uniform_norm == 0.0:
        return None
    cosine_similarity = dot_product / (observed_norm * uniform_norm)
    return 1.0 - cosine_similarity


def compute_repeated_source_translation_choice_metrics(
    df: pd.DataFrame,
) -> dict[str, float | None]:
    """Compute a repeated-source concentration proxy over complete predictions.

    This is not the dictionary-grounded Synonym Frequency Analysis (SFA)
    used in the literature. It is a coarse proxy based on how concentrated
    full-sentence predictions are within repeated-source groups.
    """
    if "source" not in df.columns or "prediction" not in df.columns:
        return {"ptf": None, "cdu": None, "groups_analyzed": 0}

    repeated_groups = [group for _, group in df.groupby("source", dropna=False) if len(group) > 1]
    if not repeated_groups:
        return {"ptf": None, "cdu": None, "groups_analyzed": 0}

    ptf_scores: list[float] = []
    cdu_scores: list[float] = []
    for group in repeated_groups:
        prediction_counts = Counter(group["prediction"].fillna("").astype(str).tolist())
        total = sum(prediction_counts.values())
        if total == 0:
            continue
        ptf_scores.append(max(prediction_counts.values()) / total)

        support = set(prediction_counts.keys())
        if "reference" in group.columns:
            support.update(group["reference"].fillna("").astype(str).tolist())
        cdu_score = _cosine_distance_from_uniform(prediction_counts, sorted(support))
        if cdu_score is not None:
            cdu_scores.append(cdu_score)

    return {
        "ptf": (sum(ptf_scores) / len(ptf_scores)) if ptf_scores else None,
        "cdu": (sum(cdu_scores) / len(cdu_scores)) if cdu_scores else None,
        "groups_analyzed": len(repeated_groups),
    }


def compute_translation_choice_metrics(df: pd.DataFrame) -> dict[str, float | None]:
    """Backward-compatible alias for the repeated-source concentration proxy."""
    return compute_repeated_source_translation_choice_metrics(df)


def compute_per_example_metrics(
    df: pd.DataFrame,
    background_texts: list[str] | None = None,
) -> pd.DataFrame:
    """Compute row-level lexical and repetition diagnostics."""
    result = df.copy()
    prediction_tokens = result["prediction"].map(tokenize_text)
    background_tokens = [
        token
        for text in (
            background_texts
            if background_texts is not None
            else result.get("reference", result["prediction"]).fillna("").astype(str).tolist()
        )
        for token in tokenize_text(text)
    ]
    frequency_counter = _build_frequency_reference(background_tokens)

    result["prediction_token_count"] = prediction_tokens.map(len)
    result["prediction_ttr"] = prediction_tokens.map(compute_ttr)
    result["prediction_yules_k"] = prediction_tokens.map(compute_yules_k)
    result["prediction_yules_i"] = prediction_tokens.map(compute_yules_i)
    result["prediction_unique_word_forms"] = prediction_tokens.map(compute_unique_word_forms)
    result["prediction_shannon_diversity"] = prediction_tokens.map(compute_shannon_diversity)
    result["prediction_simpson_diversity"] = prediction_tokens.map(compute_simpson_diversity)
    result["prediction_inverse_simpson_diversity"] = prediction_tokens.map(
        compute_inverse_simpson_diversity
    )
    result["prediction_repeated_token_rate"] = prediction_tokens.map(compute_repeated_token_rate)
    result["prediction_repeated_3gram_rate"] = prediction_tokens.map(
        lambda tokens: compute_repeated_ngram_rate(tokens, 3)
    )
    result["prediction_repeated_4gram_rate"] = prediction_tokens.map(
        lambda tokens: compute_repeated_ngram_rate(tokens, 4)
    )
    result["prediction_consecutive_repetition_rate"] = prediction_tokens.map(
        compute_consecutive_repetition_rate
    )
    result["prediction_content_word_repetition_rate"] = prediction_tokens.map(
        compute_content_word_repetition_rate
    )
    result["prediction_lexical_density_proxy"] = prediction_tokens.map(
        compute_lexical_density_proxy
    )
    result["prediction_function_to_content_ratio_proxy"] = prediction_tokens.map(
        compute_function_to_content_ratio_proxy
    )
    result["prediction_content_to_function_ratio_proxy"] = prediction_tokens.map(
        compute_content_to_function_ratio_proxy
    )
    result["prediction_function_word_proportion_proxy"] = prediction_tokens.map(
        compute_function_word_proportion_proxy
    )
    frequency_profiles = prediction_tokens.map(
        lambda tokens: compute_lexical_frequency_profile(tokens, frequency_counter)
    )
    result["prediction_lfp_b1_top_1000"] = frequency_profiles.map(
        lambda profile: profile["b1_top_1000"]
    )
    result["prediction_lfp_b2_1001_2000"] = frequency_profiles.map(
        lambda profile: profile["b2_1001_2000"]
    )
    result["prediction_lfp_b3_beyond_2000"] = frequency_profiles.map(
        lambda profile: profile["b3_beyond_2000"]
    )

    if "reference" in result.columns:
        references = result["reference"].fillna("").astype(str).tolist()
        predictions = result["prediction"].fillna("").astype(str).tolist()
        result["sentence_chrf_pp"] = [
            float(sacrebleu.sentence_chrf(prediction, [reference], word_order=2).score)
            for prediction, reference in zip(predictions, references)
        ]
        result["sentence_bleu"] = [
            float(sacrebleu.sentence_bleu(prediction, [reference], use_effective_order=True).score)
            for prediction, reference in zip(predictions, references)
        ]
    return result


def _default_comet_gpus() -> int:
    """Use a GPU for COMET-family metrics when one is available."""
    try:  # pragma: no cover - torch availability is environment-dependent
        import torch
    except Exception:
        return 0
    return 1 if torch.cuda.is_available() else 0


def _release_comet_model_cache() -> None:
    """Release cached COMET-family models before switching checkpoints.

    Full COMET, XCOMET, and COMETKiwi checkpoints cannot coexist on a single
    evaluation GPU. Keeping at most one in memory lets a full metric pass run
    sequentially without changing any score calculations.
    """
    if not _COMET_MODEL_CACHE:
        return
    _COMET_MODEL_CACHE.clear()
    gc.collect()
    try:  # pragma: no cover - torch availability is environment-dependent
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _load_comet_model(model_path: str) -> Any:
    """Download and cache one COMET-family model checkpoint at a time."""
    if model_path in _COMET_MODEL_CACHE:
        return _COMET_MODEL_CACHE[model_path]

    _release_comet_model_cache()

    from comet import download_model, load_from_checkpoint

    checkpoint_path = download_model(model_path)
    model = load_from_checkpoint(checkpoint_path)
    _COMET_MODEL_CACHE[model_path] = model
    return model


def _extract_comet_system_score(outputs: Any) -> float | None:
    """Support COMET outputs exposed either as attrs or mapping keys."""
    if outputs is None:
        return None
    if hasattr(outputs, "system_score"):
        system_score = getattr(outputs, "system_score")
        return float(system_score) if system_score is not None else None
    if isinstance(outputs, dict) and "system_score" in outputs:
        system_score = outputs["system_score"]
        return float(system_score) if system_score is not None else None
    return None


def compute_optional_comet(
    sources: list[str],
    predictions: list[str],
    references: list[str] | None = None,
    model_path: str | None = None,
    batch_size: int = 8,
    gpus: int | None = None,
) -> float | None:
    """Compute COMET-style scores when the optional dependency is available."""
    if model_path is None:
        return None
    model = None
    outputs = None
    try:
        try:  # pragma: no cover - optional dependency / model downloads
            model = _load_comet_model(model_path)
        except Exception:
            return None

        if references is None:
            records = [
                {"src": source, "mt": prediction}
                for source, prediction in zip(sources, predictions)
            ]
        else:
            records = [
                {"src": source, "mt": prediction, "ref": reference}
                for source, prediction, reference in zip(sources, predictions, references)
            ]
        outputs = model.predict(
            records,
            batch_size=batch_size,
            gpus=_default_comet_gpus() if gpus is None else gpus,
            progress_bar=False,
        )
        return _extract_comet_system_score(outputs)
    finally:
        # Lightning may retain references to the prediction trainer. Drop every
        # caller-owned reference before loading the next large COMET checkpoint.
        outputs = None
        model = None
        _release_comet_model_cache()


def compute_optional_metricx(
    sources: list[str],
    predictions: list[str],
    references: list[str],
    model_path: str | None = None,
    tokenizer: str = DEFAULT_METRICX_TOKENIZER,
    max_input_length: int = 1536,
    batch_size: int = 1,
    python_executable: str = DEFAULT_METRICX_PYTHON,
    repository_path: str | Path = DEFAULT_METRICX_REPOSITORY,
) -> float | None:
    """Run official MetricX-24 reference-based inference and return its mean error.

    MetricX-24 scores are clipped to [0, 25], with lower values indicating
    fewer predicted translation errors. The MetricX source pins an older
    Transformers stack, so inference runs in a dedicated environment rather
    than changing the training/evaluation environment.
    """
    if model_path is None:
        return None
    if not (len(sources) == len(predictions) == len(references)):
        raise ValueError("MetricX sources, predictions, and references must have equal length.")
    if not sources:
        return None

    executable = Path(python_executable)
    repository = Path(repository_path)
    if not executable.is_file() or not repository.is_dir():
        logger.warning("MetricX is configured but its environment or repository is unavailable.")
        return None

    try:
        with tempfile.TemporaryDirectory(prefix="metricx_") as temporary_dir:
            temporary_path = Path(temporary_dir)
            input_path = temporary_path / "input.jsonl"
            output_path = temporary_path / "output.jsonl"
            with input_path.open("w", encoding="utf-8") as handle:
                for source, prediction, reference in zip(sources, predictions, references):
                    handle.write(
                        json.dumps(
                            {
                                "source": source,
                                "hypothesis": prediction,
                                "reference": reference,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            environment = os.environ.copy()
            existing_pythonpath = environment.get("PYTHONPATH", "")
            environment["PYTHONPATH"] = (
                f"{repository.resolve()}{os.pathsep}{existing_pythonpath}"
                if existing_pythonpath
                else str(repository.resolve())
            )
            subprocess.run(
                [
                    str(executable),
                    "-m",
                    "metricx24.predict",
                    "--tokenizer",
                    tokenizer,
                    "--model_name_or_path",
                    model_path,
                    "--max_input_length",
                    str(max_input_length),
                    "--batch_size",
                    str(batch_size),
                    "--input_file",
                    str(input_path),
                    "--output_file",
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            scores = [
                float(json.loads(line)["prediction"])
                for line in output_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
    except subprocess.CalledProcessError as exc:
        logger.warning(
            "MetricX inference failed (exit %s): %s",
            exc.returncode,
            (exc.stderr or exc.stdout or str(exc)).strip(),
        )
        return None
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("MetricX inference failed: %s", exc)
        return None

    if len(scores) != len(predictions):
        logger.warning("MetricX returned %d scores for %d predictions.", len(scores), len(predictions))
        return None
    return float(sum(scores) / len(scores))


def compute_automatic_metrics(
    df: pd.DataFrame,
    comet_model: str | None = None,
    xcomet_model: str | None = None,
    comet_qe_model: str | None = None,
    cometkiwi_model: str | None = None,
    metricx_model: str | None = None,
    metricx_tokenizer: str = DEFAULT_METRICX_TOKENIZER,
    metricx_max_input_length: int = 1536,
    metricx_batch_size: int = 1,
    background_texts: list[str] | None = None,
    comet_batch_size: int = 8,
    comet_gpus: int | None = None,
) -> dict[str, Any]:
    """Compute MT adequacy only.

    Target-side naturalness and translationese metrics are computed separately
    for references and predictions by the analysis runner.
    """
    if "prediction" not in df.columns:
        raise ValueError("The predictions DataFrame must contain a 'prediction' column")

    predictions = df["prediction"].fillna("").astype(str).tolist()
    references = (
        df["reference"].fillna("").astype(str).tolist() if "reference" in df.columns else []
    )
    sources = df["source"].fillna("").astype(str).tolist() if "source" in df.columns else []
    effective_cometkiwi_model = cometkiwi_model if cometkiwi_model is not None else comet_qe_model
    comet_score = (
        compute_optional_comet(
            sources,
            predictions,
            references,
            comet_model,
            batch_size=comet_batch_size,
            gpus=comet_gpus,
        )
        if references and sources
        else None
    )
    xcomet_score = (
        compute_optional_comet(
            sources,
            predictions,
            references,
            xcomet_model,
            batch_size=comet_batch_size,
            gpus=comet_gpus,
        )
        if references and sources
        else None
    )
    cometkiwi_score = (
        compute_optional_comet(
            sources,
            predictions,
            None,
            effective_cometkiwi_model,
            batch_size=comet_batch_size,
            gpus=comet_gpus,
        )
        if sources
        else None
    )
    metricx_score = (
        compute_optional_metricx(
            sources,
            predictions,
            references,
            metricx_model,
            tokenizer=metricx_tokenizer,
            max_input_length=metricx_max_input_length,
            batch_size=metricx_batch_size,
        )
        if references and sources
        else None
    )
    # COMETKiwi is reference-free quality estimation, so the human reference
    # itself can be scored as a target-side translation of the source. This is
    # an observed baseline, unlike self-BLEU/chrF/TER, which would be trivial.
    reference_cometkiwi_score = (
        compute_optional_comet(
            sources,
            references,
            None,
            effective_cometkiwi_model,
            batch_size=comet_batch_size,
            gpus=comet_gpus,
        )
        if references and sources
        else None
    )
    return {
        "row_count": int(len(df)),
        "adequacy": {
            "bleu": compute_bleu(predictions, references) if references else None,
            "ter": compute_ter(predictions, references) if references else None,
            "chrf_pp": compute_chrf_pp(predictions, references) if references else None,
            "comet": comet_score,
            "xcomet": xcomet_score,
            "cometkiwi": cometkiwi_score,
            "comet_qe": cometkiwi_score,
            "metricx": metricx_score,
        },
        "reference_adequacy": {
            "cometkiwi": reference_cometkiwi_score,
            "comet_qe": reference_cometkiwi_score,
        },
    }
