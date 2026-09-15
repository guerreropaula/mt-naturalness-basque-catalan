"""Paired significance testing for aligned per-example evaluation metrics."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.io import save_dataframe_tsv, save_json

_EXPERIMENT_DIRS = {label: Path(label.lower()) for label in ("P0", "P1", "P2", "P3")}
_LEGACY_EXPERIMENT_DIRS = {
    "P0": (Path("baseline/p0_baseline"),),
    "P1": (Path("prompting/system_prompts/p1_naturalness"),),
    "P2": (Path("prompting/self_polishing/p2_polishing"),),
    "P3": (Path("advanced/step_by_step/p3_step_by_step"), Path("advanced/cot/p3_cot")),
}
_DEFAULT_METRICS = ("sentence_bleu", "sentence_chrf_pp", "astred_ted")
_LOWER_IS_BETTER = {
    "astred_ted",
    "astred_word_cross",
    "astred_seq_cross",
    "astred_sacr_cross",
    "astred_source_prediction_ted",
    "astred_source_prediction_word_cross",
    "astred_source_prediction_seq_cross",
    "astred_source_prediction_sacr_cross",
}
_REQUIRED_ID_COLUMNS = ("sentence_id", "source", "reference")
_CONDITIONS = {
    "P0",
    "P1",
    "P2",
    "P3",
    "P4_SFT",
    "P5_GRPO_A0",
    "P5_GRPO_A1",
    "P5_GRPO_A2",
    "P5_GRPO_A3",
    "P5_GRPO_A4",
    "P5_GRPO_A3V2",
    "P5_GRPO_A3V3",
    "P5_GRPO_A3V4",
    "P5_GRPO_A3V5",
    "P5_GRPO_A5",
}
_ALIASES = {"P4": "P4_SFT", "P5": "P5_GRPO_A2", "P5_GRPO": "P5_GRPO_A2"}


def _metrics_path(
    results_root: Path,
    label: str,
    dataset: str,
    model: str,
    evaluation_split: str = "test",
) -> Path:
    """Return the canonical per-example metric path for a P0-P5 condition."""
    key = str(label).strip().lower()
    key = {"p4": "p4_sft", "p5": "p5_grpo_a2", "p5_grpo": "p5_grpo_a2"}.get(key, key)
    project_root = results_root.parent if results_root.name == "p0_p3" else results_root
    if key in {"p0", "p1", "p2", "p3"}:
        prompt_root = results_root if results_root.name == "p0_p3" else project_root / "p0_p3"
        base = prompt_root / key / dataset / model
        split_path = base / evaluation_split / "per_example_metrics.jsonl"
        if split_path.exists():
            return split_path
        canonical = base / "per_example_metrics.jsonl"
        if canonical.exists():
            return canonical
        for legacy_dir in _LEGACY_EXPERIMENT_DIRS[key.upper()]:
            legacy = prompt_root / legacy_dir / dataset / model / "per_example_metrics.jsonl"
            if legacy.exists():
                return legacy
        return split_path
    if key == "p4_sft":
        return (
            project_root
            / "p4_sft"
            / dataset
            / model
            / evaluation_split
            / "per_example_metrics.jsonl"
        )
    if key.startswith("p5_grpo_") and key.rsplit("_", 1)[1] in {"a0", "a1", "a2", "a3", "a4", "a3v2", "a3v3", "a3v4", "a3v5", "a5"}:
        ablation = key.rsplit("_", 1)[1]
        return (
            project_root
            / "p5_grpo"
            / ablation
            / dataset
            / model
            / evaluation_split
            / "per_example_metrics.jsonl"
        )
    raise ValueError(f"Unknown experiment condition: {label}")


def load_per_example_metrics(path: Path) -> pd.DataFrame:
    """Load metrics and validate that sentence identifiers are unique."""
    frame = pd.read_json(path, lines=True)
    missing = [column for column in _REQUIRED_ID_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    if frame["sentence_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate sentence_id values.")
    return frame.set_index("sentence_id", drop=False)


def align_metric_values(
    baseline: pd.DataFrame, candidate: pd.DataFrame, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return paired finite values after checking the parallel inputs match exactly."""
    if metric not in baseline or metric not in candidate:
        raise ValueError(f"Metric '{metric}' is not available in both per-example files.")
    if baseline.index.tolist() != candidate.index.tolist():
        raise ValueError("Sentence IDs differ or are in a different order between paired runs.")
    for column in ("source", "reference"):
        if not baseline[column].equals(candidate[column]):
            raise ValueError(f"Paired runs have different {column} texts.")

    baseline_values = pd.to_numeric(baseline[metric], errors="coerce").to_numpy(dtype=float)
    candidate_values = pd.to_numeric(candidate[metric], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(baseline_values) & np.isfinite(candidate_values)
    if not valid.any():
        raise ValueError(f"Metric '{metric}' has no finite paired observations.")
    return baseline_values[valid], candidate_values[valid]


def paired_bootstrap(
    differences: np.ndarray,
    *,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Return the 95% percentile interval for the mean paired difference."""
    sample_indices = rng.integers(0, len(differences), size=(iterations, len(differences)))
    bootstrap_means = differences[sample_indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, (0.025, 0.975))
    return float(lower), float(upper)


def paired_randomization_p_value(
    differences: np.ndarray,
    *,
    iterations: int,
    rng: np.random.Generator,
) -> float:
    """Approximate a two-sided paired randomization p-value by sign flipping."""
    observed = abs(float(differences.mean()))
    signs = rng.choice((-1.0, 1.0), size=(iterations, len(differences)))
    null_means = abs((signs * differences).mean(axis=1))
    return float((np.count_nonzero(null_means >= observed) + 1) / (iterations + 1))


def holm_adjust(p_values: list[float]) -> list[float]:
    """Apply Holm's family-wise correction while preserving input order."""
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running_max = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running_max = max(running_max, min(1.0, p_values[index] * (total - rank)))
        adjusted[index] = running_max
    return adjusted


def compare_metric(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    metric: str,
    baseline_label: str,
    candidate_label: str,
    bootstrap_iterations: int,
    randomization_iterations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Compare a candidate to baseline for one per-example metric."""
    baseline_values, candidate_values = align_metric_values(baseline, candidate, metric)
    differences = candidate_values - baseline_values
    ci_lower, ci_upper = paired_bootstrap(differences, iterations=bootstrap_iterations, rng=rng)
    return {
        "baseline": baseline_label,
        "candidate": candidate_label,
        "metric": metric,
        "direction": "lower_is_better" if metric in _LOWER_IS_BETTER else "higher_is_better",
        "status": "ok",
        "skip_reason": "",
        "paired_rows": int(len(differences)),
        "baseline_mean": float(baseline_values.mean()),
        "candidate_mean": float(candidate_values.mean()),
        "mean_delta_candidate_minus_baseline": float(differences.mean()),
        "bootstrap_ci_95_lower": ci_lower,
        "bootstrap_ci_95_upper": ci_upper,
        "randomization_p_value": paired_randomization_p_value(
            differences, iterations=randomization_iterations, rng=rng
        ),
    }


def run_comparisons(
    frames: dict[str, pd.DataFrame],
    *,
    metrics: tuple[str, ...],
    candidates: tuple[str, ...],
    baseline_label: str = "P0",
    bootstrap_iterations: int,
    randomization_iterations: int,
    seed: int,
) -> pd.DataFrame:
    """Run all requested candidate comparisons against one aligned baseline."""
    if baseline_label not in frames:
        raise ValueError(f"Baseline '{baseline_label}' is not loaded.")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        for metric in metrics:
            try:
                rows.append(
                    compare_metric(
                        frames[baseline_label],
                        frames[candidate],
                        metric=metric,
                        baseline_label=baseline_label,
                        candidate_label=candidate,
                        bootstrap_iterations=bootstrap_iterations,
                        randomization_iterations=randomization_iterations,
                        rng=rng,
                    )
                )
            except ValueError as exc:
                rows.append(
                    {
                        "baseline": baseline_label,
                        "candidate": candidate,
                        "metric": metric,
                        "direction": "lower_is_better" if metric in _LOWER_IS_BETTER else "higher_is_better",
                        "status": "skipped",
                        "skip_reason": str(exc),
                        "paired_rows": 0,
                        "baseline_mean": np.nan,
                        "candidate_mean": np.nan,
                        "mean_delta_candidate_minus_baseline": np.nan,
                        "bootstrap_ci_95_lower": np.nan,
                        "bootstrap_ci_95_upper": np.nan,
                        "randomization_p_value": np.nan,
                    }
                )
    result = pd.DataFrame(rows)
    result["holm_adjusted_p_value"] = np.nan
    valid = result["randomization_p_value"].notna()
    if valid.any():
        result.loc[valid, "holm_adjusted_p_value"] = holm_adjust(
            result.loc[valid, "randomization_p_value"].tolist()
        )
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--dataset", required=True, help="Dataset key, for example en_ca or en_eu.")
    parser.add_argument("--model", default="latxa_8b_instruct", help="Model output directory name.")
    parser.add_argument("--evaluation-split", default="test", help="P4/P5 evaluation split.")
    parser.add_argument(
        "--baseline",
        default="P0",
        help="Condition used as the paired baseline (default: P0).",
    )
    parser.add_argument(
        "--metrics",
        default=",".join(_DEFAULT_METRICS),
        help="Comma-separated per-example metric columns.",
    )
    parser.add_argument(
        "--candidates",
        default="P1,P2,P3,P4_SFT,P5_GRPO_A0,P5_GRPO_A1,P5_GRPO_A2,P5_GRPO_A3,P5_GRPO_A4,P5_GRPO_A3V2,P5_GRPO_A3V3,P5_GRPO_A3V4,P5_GRPO_A3V5,P5_GRPO_A5",
        help="Comma-separated conditions compared with --baseline.",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--randomization-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None, help="Output TSV path.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    metrics = tuple(metric.strip() for metric in args.metrics.split(",") if metric.strip())
    # Slurm's --export treats commas as variable separators. Accept either
    # commas or whitespace so submitted CANDIDATES values remain valid.
    candidates = tuple(label for label in re.split(r"[\s,]+", args.candidates.strip()) if label)
    baseline = _ALIASES.get(args.baseline.upper(), args.baseline.upper())
    candidates = tuple(_ALIASES.get(label.upper(), label.upper()) for label in candidates)
    unknown = ({baseline} | set(candidates)) - _CONDITIONS
    if unknown:
        raise ValueError(f"Unknown experiment labels: {', '.join(sorted(unknown))}")
    if baseline in candidates:
        raise ValueError("The baseline condition cannot also be a candidate.")
    if not metrics:
        raise ValueError("At least one metric is required.")
    if args.bootstrap_iterations < 1 or args.randomization_iterations < 1:
        raise ValueError("Iteration counts must be positive.")

    baseline_path = _metrics_path(args.results_root, baseline, args.dataset, args.model, args.evaluation_split)
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline per-example metrics at {baseline_path}")
    frames = {baseline: load_per_example_metrics(baseline_path)}
    available_candidates: list[str] = []
    for label in candidates:
        path = _metrics_path(args.results_root, label, args.dataset, args.model, args.evaluation_split)
        if not path.exists():
            print(f"Skipping {label}: missing per-example metrics at {path}")
            continue
        frames[label] = load_per_example_metrics(path)
        available_candidates.append(label)
    if not available_candidates:
        raise ValueError("No candidate per-example metric files were found.")

    result = run_comparisons(
        frames,
        metrics=metrics,
        candidates=tuple(available_candidates),
        baseline_label=baseline,
        bootstrap_iterations=args.bootstrap_iterations,
        randomization_iterations=args.randomization_iterations,
        seed=args.seed,
    )
    output_name = (
        "paired_significance_p0_p5.tsv"
        if baseline == "P0"
        else f"paired_significance_{baseline.lower()}_vs_candidates.tsv"
    )
    output = args.output or (
        (args.results_root.parent if args.results_root.name == "p0_p3" else args.results_root)
        / "significance"
        / args.dataset
        / args.model
        / output_name
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
            "candidates": list(result["candidate"].drop_duplicates()),
            "bootstrap_iterations": args.bootstrap_iterations,
            "randomization_iterations": args.randomization_iterations,
            "seed": args.seed,
            "results": result.to_dict(orient="records"),
        },
        output.with_suffix(".json"),
    )
    print(f"Wrote paired significance results to {output}")


if __name__ == "__main__":
    main()
