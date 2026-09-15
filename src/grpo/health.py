"""Deterministic health checks for P5 GRPO policy collapse."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

from src.grpo.reward import sentence_chrfpp


@dataclass(frozen=True)
class HealthThresholds:
    """Conservative failure thresholds for a fixed GRPO development sample."""

    min_mean_chrfpp: float
    max_repeated_token_ratio: float
    max_repeated_rows_fraction: float
    max_unterminated_fraction: float
    max_empty_fraction: float
    max_duplicate_fraction: float

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "HealthThresholds":
        return cls(
            min_mean_chrfpp=float(values["min_mean_chrfpp"]),
            max_repeated_token_ratio=float(values["max_repeated_token_ratio"]),
            max_repeated_rows_fraction=float(values["max_repeated_rows_fraction"]),
            max_unterminated_fraction=float(values["max_unterminated_fraction"]),
            max_empty_fraction=float(values["max_empty_fraction"]),
            max_duplicate_fraction=float(values["max_duplicate_fraction"]),
        )


def _largest_token_fraction(token_ids: Sequence[int]) -> float:
    if len(token_ids) < 8:
        return 0.0
    return max(Counter(int(token_id) for token_id in token_ids).values()) / len(token_ids)


def assess_translation_health(
    predictions: Sequence[str],
    references: Sequence[str],
    completion_token_ids: Sequence[Sequence[int]],
    terminated: Sequence[bool],
    thresholds: HealthThresholds,
) -> dict[str, Any]:
    """Summarize a deterministic sample and flag unmistakable policy collapse."""
    size = len(predictions)
    if size == 0 or not (
        len(references) == size == len(completion_token_ids) == len(terminated)
    ):
        raise ValueError("GRPO health inputs must be non-empty and have the same length.")

    empty_rows = sum(not str(prediction).strip() for prediction in predictions)
    unterminated_rows = sum(not bool(value) for value in terminated)
    repeated_token_fractions = [
        _largest_token_fraction(token_ids) for token_ids in completion_token_ids
    ]
    repeated_rows = sum(
        fraction >= thresholds.max_repeated_token_ratio
        for fraction in repeated_token_fractions
    )
    normalized_predictions = [str(prediction).strip() for prediction in predictions]
    duplicate_rows = max(Counter(normalized_predictions).values())
    mean_chrfpp = sum(
        sentence_chrfpp(prediction, reference)
        for prediction, reference in zip(predictions, references, strict=True)
    ) / size

    fractions = {
        "empty": empty_rows / size,
        "unterminated": unterminated_rows / size,
        "repeated_rows": repeated_rows / size,
        "largest_duplicate_cluster": duplicate_rows / size,
    }
    failures: list[str] = []
    if mean_chrfpp < thresholds.min_mean_chrfpp:
        failures.append(
            f"mean_chrFpp={mean_chrfpp:.4f} < {thresholds.min_mean_chrfpp:.4f}"
        )
    if fractions["empty"] > thresholds.max_empty_fraction:
        failures.append(
            f"empty_fraction={fractions['empty']:.4f} > "
            f"{thresholds.max_empty_fraction:.4f}"
        )
    if fractions["unterminated"] > thresholds.max_unterminated_fraction:
        failures.append(
            f"unterminated_fraction={fractions['unterminated']:.4f} > "
            f"{thresholds.max_unterminated_fraction:.4f}"
        )
    if fractions["repeated_rows"] > thresholds.max_repeated_rows_fraction:
        failures.append(
            f"repeated_rows_fraction={fractions['repeated_rows']:.4f} > "
            f"{thresholds.max_repeated_rows_fraction:.4f}"
        )
    if fractions["largest_duplicate_cluster"] > thresholds.max_duplicate_fraction:
        failures.append(
            "largest_duplicate_fraction="
            f"{fractions['largest_duplicate_cluster']:.4f} > "
            f"{thresholds.max_duplicate_fraction:.4f}"
        )

    return {
        "passed": not failures,
        "examples": size,
        "mean_chrfpp": mean_chrfpp,
        "empty_rows": empty_rows,
        "unterminated_rows": unterminated_rows,
        "repeated_rows": repeated_rows,
        "largest_duplicate_cluster": duplicate_rows,
        "fractions": fractions,
        "max_observed_token_repetition": max(repeated_token_fractions),
        "failures": failures,
        "predictions": list(predictions),
    }
