"""Paired corpus-bootstrap confidence intervals for lexical diversity metrics."""

from __future__ import annotations

import argparse
import re
from itertools import chain
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.evaluation.metrics.automatic import compute_mattr, compute_mtld, tokenize_text
from src.evaluation.statistics.paired import (
    _ALIASES,
    _CONDITIONS,
    _metrics_path,
    load_per_example_metrics,
)
from src.utils.io import save_dataframe_tsv, save_json

CorpusMetric = Callable[[list[str]], float | None]
_METRICS: dict[str, CorpusMetric] = {
    "mtld": lambda tokens: compute_mtld(tokens, reliable_min_tokens=100),
    "mattr_50": lambda tokens: compute_mattr(tokens, window_size=50),
}


def align_prediction_tokens(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
) -> tuple[list[list[str]], list[list[str]]]:
    """Validate paired inputs and tokenize each aligned output once."""
    if "prediction" not in baseline or "prediction" not in candidate:
        raise ValueError("Both per-example files must contain a prediction column.")
    if baseline.index.tolist() != candidate.index.tolist():
        raise ValueError("Sentence IDs differ or are in a different order between paired runs.")
    for column in ("source", "reference"):
        if not baseline[column].equals(candidate[column]):
            raise ValueError(f"Paired runs have different {column} texts.")
    baseline_rows = [tokenize_text(text) for text in baseline["prediction"].fillna("")]
    candidate_rows = [tokenize_text(text) for text in candidate["prediction"].fillna("")]
    return baseline_rows, candidate_rows


def _flatten(rows: list[list[str]], indices: np.ndarray | None = None) -> list[str]:
    selected = rows if indices is None else (rows[int(index)] for index in indices)
    return list(chain.from_iterable(selected))


def paired_corpus_bootstrap(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    metric: str,
    iterations: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    """Resample aligned rows and recompute a non-additive corpus metric."""
    if metric not in _METRICS:
        raise ValueError(f"Unsupported corpus metric: {metric}")
    if iterations < 1:
        raise ValueError("Bootstrap iterations must be positive.")
    baseline_rows, candidate_rows = align_prediction_tokens(baseline, candidate)
    row_count = len(baseline_rows)
    if row_count == 0:
        raise ValueError("Cannot bootstrap an empty aligned corpus.")

    scorer = _METRICS[metric]
    baseline_score = scorer(_flatten(baseline_rows))
    candidate_score = scorer(_flatten(candidate_rows))
    if baseline_score is None or candidate_score is None:
        raise ValueError(f"Metric '{metric}' is undefined for the complete corpus.")

    baseline_samples: list[float] = []
    candidate_samples: list[float] = []
    for _ in range(iterations):
        indices = rng.integers(0, row_count, size=row_count)
        baseline_value = scorer(_flatten(baseline_rows, indices))
        candidate_value = scorer(_flatten(candidate_rows, indices))
        if baseline_value is not None and candidate_value is not None:
            baseline_samples.append(float(baseline_value))
            candidate_samples.append(float(candidate_value))
    if not baseline_samples:
        raise ValueError(f"Metric '{metric}' was undefined in every bootstrap sample.")

    baseline_array = np.asarray(baseline_samples, dtype=float)
    candidate_array = np.asarray(candidate_samples, dtype=float)
    deltas = candidate_array - baseline_array
    baseline_lower, baseline_upper = np.quantile(baseline_array, (0.025, 0.975))
    candidate_lower, candidate_upper = np.quantile(candidate_array, (0.025, 0.975))
    delta_lower, delta_upper = np.quantile(deltas, (0.025, 0.975))
    return {
        "paired_rows": row_count,
        "baseline_score": float(baseline_score),
        "candidate_score": float(candidate_score),
        "delta_candidate_minus_baseline": float(candidate_score - baseline_score),
        "baseline_ci_95_lower": float(baseline_lower),
        "baseline_ci_95_upper": float(baseline_upper),
        "candidate_ci_95_lower": float(candidate_lower),
        "candidate_ci_95_upper": float(candidate_upper),
        "delta_ci_95_lower": float(delta_lower),
        "delta_ci_95_upper": float(delta_upper),
        "valid_bootstrap_samples": len(deltas),
    }


def run_corpus_bootstrap_comparisons(
    frames: dict[str, pd.DataFrame],
    *,
    metrics: tuple[str, ...],
    candidates: tuple[str, ...],
    baseline_label: str,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    """Compare each candidate with one baseline using paired corpus resampling."""
    if baseline_label not in frames:
        raise ValueError(f"Baseline '{baseline_label}' is not loaded.")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        for metric in metrics:
            row: dict[str, Any] = {
                "baseline": baseline_label,
                "candidate": candidate,
                "metric": metric,
                "direction": "higher_is_better",
                "analysis_level": "corpus",
                "status": "ok",
                "skip_reason": "",
            }
            try:
                row.update(
                    paired_corpus_bootstrap(
                        frames[baseline_label],
                        frames[candidate],
                        metric=metric,
                        iterations=iterations,
                        rng=rng,
                    )
                )
            except ValueError as exc:
                row.update({"status": "skipped", "skip_reason": str(exc), "paired_rows": 0})
            rows.append(row)
    return pd.DataFrame(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="latxa_8b_instruct")
    parser.add_argument("--evaluation-split", default="test")
    parser.add_argument("--baseline", default="P0")
    parser.add_argument(
        "--candidates",
        default="P1,P2,P3,P4_SFT,P5_GRPO_A0,P5_GRPO_A1,P5_GRPO_A2,P5_GRPO_A3,P5_GRPO_A4,P5_GRPO_A3V2,P5_GRPO_A3V3,P5_GRPO_A3V4,P5_GRPO_A3V5,P5_GRPO_A5",
    )
    parser.add_argument("--metrics", default="mtld,mattr_50")
    parser.add_argument("--iterations", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    baseline = _ALIASES.get(args.baseline.upper(), args.baseline.upper())
    candidates = tuple(
        _ALIASES.get(label.upper(), label.upper())
        for label in re.split(r"[\s,]+", args.candidates.strip())
        if label
    )
    metrics = tuple(metric.strip() for metric in args.metrics.split(",") if metric.strip())
    unknown_conditions = ({baseline} | set(candidates)) - _CONDITIONS
    if unknown_conditions:
        raise ValueError(f"Unknown experiment labels: {', '.join(sorted(unknown_conditions))}")
    unknown_metrics = set(metrics) - set(_METRICS)
    if unknown_metrics:
        raise ValueError(f"Unsupported corpus metrics: {', '.join(sorted(unknown_metrics))}")
    if baseline in candidates:
        raise ValueError("The baseline condition cannot also be a candidate.")
    if args.iterations < 1:
        raise ValueError("Bootstrap iterations must be positive.")

    baseline_path = _metrics_path(
        args.results_root, baseline, args.dataset, args.model, args.evaluation_split
    )
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline per-example metrics at {baseline_path}")
    frames = {baseline: load_per_example_metrics(baseline_path)}
    available: list[str] = []
    for label in candidates:
        path = _metrics_path(
            args.results_root, label, args.dataset, args.model, args.evaluation_split
        )
        if not path.exists():
            print(f"Skipping {label}: missing per-example metrics at {path}")
            continue
        frames[label] = load_per_example_metrics(path)
        available.append(label)
    if not available:
        raise ValueError("No candidate per-example metric files were found.")

    result = run_corpus_bootstrap_comparisons(
        frames,
        metrics=metrics,
        candidates=tuple(available),
        baseline_label=baseline,
        iterations=args.iterations,
        seed=args.seed,
    )
    project_root = (
        args.results_root.parent if args.results_root.name == "p0_p3" else args.results_root
    )
    output = args.output or (
        project_root
        / "significance"
        / args.dataset
        / args.model
        / f"paired_corpus_bootstrap_{baseline.lower()}_vs_candidates.tsv"
    )
    save_dataframe_tsv(result, output)
    save_json(
        {
            "dataset": args.dataset,
            "model": args.model,
            "evaluation_split": args.evaluation_split,
            "baseline": baseline,
            "metrics": list(metrics),
            "requested_candidates": list(candidates),
            "available_candidates": available,
            "iterations": args.iterations,
            "seed": args.seed,
            "method": "paired row bootstrap; concatenate sampled outputs and recompute corpus metric",
            "results": result.to_dict(orient="records"),
        },
        output.with_suffix(".json"),
    )
    print(f"Wrote paired corpus-bootstrap results to {output}")


if __name__ == "__main__":
    main()
