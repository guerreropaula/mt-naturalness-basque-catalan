"""Describe P3 proofreading acceptance and fallback behavior."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.io import save_dataframe_tsv, save_json

_DEFAULT_METRICS = ("sentence_bleu", "sentence_chrf_pp", "astred_ted")


def _load_jsonl(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def build_p3_diagnostic(predictions: pd.DataFrame, metrics: pd.DataFrame, metric_names: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize P3's accepted proofreading and refined-output fallback groups."""
    required = {"sentence_id", "prediction", "refined_translation", "proofreading_accepted"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"P3 predictions are missing required columns: {', '.join(missing)}")
    if predictions["sentence_id"].duplicated().any() or metrics["sentence_id"].duplicated().any():
        raise ValueError("P3 diagnostic requires unique sentence_id values.")

    metric_columns = ["sentence_id", *metric_names]
    missing_metrics = sorted(set(metric_columns) - set(metrics.columns))
    if missing_metrics:
        raise ValueError(f"P3 per-example metrics are missing: {', '.join(missing_metrics)}")
    merged = predictions.merge(metrics[metric_columns], on="sentence_id", how="inner", validate="one_to_one")
    if len(merged) != len(predictions):
        raise ValueError("P3 predictions and per-example metrics do not contain the same sentence IDs.")

    merged["proofreading_group"] = merged["proofreading_accepted"].map(
        {True: "accepted", False: "fallback_to_refined"}
    )
    if merged["proofreading_group"].isna().any():
        raise ValueError("proofreading_accepted must be boolean for every P3 row.")
    merged["proofreading_changed_output"] = (
        merged["prediction"].fillna("").astype(str)
        != merged["refined_translation"].fillna("").astype(str)
    )

    rows: list[dict[str, object]] = []
    total = len(merged)
    for group, group_frame in merged.groupby("proofreading_group", sort=False):
        row: dict[str, object] = {
            "proofreading_group": group,
            "rows": int(len(group_frame)),
            "row_proportion": float(len(group_frame) / total),
            "outputs_changed_from_refinement": int(group_frame["proofreading_changed_output"].sum()),
            "outputs_changed_proportion": float(group_frame["proofreading_changed_output"].mean()),
        }
        for metric in metric_names:
            values = pd.to_numeric(group_frame[metric], errors="coerce").dropna()
            row[f"{metric}_rows"] = int(len(values))
            row[f"{metric}_mean"] = float(values.mean()) if not values.empty else None
            row[f"{metric}_median"] = float(values.median()) if not values.empty else None
        rows.append(row)

    rejection_column = "proofreading_rejection_reason"
    if rejection_column not in merged:
        rejections = pd.DataFrame(columns=[rejection_column, "rows"])
    else:
        rejections = (
            merged.loc[merged["proofreading_group"] == "fallback_to_refined", rejection_column]
            .fillna("unspecified")
            .value_counts()
            .rename_axis(rejection_column)
            .reset_index(name="rows")
        )
    return pd.DataFrame(rows), rejections


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results/p0_p3"))
    parser.add_argument("--dataset", required=True, help="Dataset key, for example en_ca or en_eu.")
    parser.add_argument("--model", default="latxa_8b_instruct", help="Model output directory name.")
    parser.add_argument(
        "--metrics", default=",".join(_DEFAULT_METRICS), help="Comma-separated per-example metric columns."
    )
    parser.add_argument("--output", type=Path, default=None, help="Output TSV path.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    metric_names = tuple(metric.strip() for metric in args.metrics.split(",") if metric.strip())
    if not metric_names:
        raise ValueError("At least one metric is required.")
    flat_dir = args.results_root / "p3" / args.dataset / args.model
    canonical_dir = args.results_root / "advanced/step_by_step/p3_step_by_step" / args.dataset / args.model
    legacy_dir = args.results_root / "advanced/cot/p3_cot" / args.dataset / args.model
    output_dir = next(
        (path for path in (flat_dir, canonical_dir, legacy_dir) if (path / "predictions.jsonl").exists()),
        flat_dir,
    )
    predictions = _load_jsonl(output_dir / "predictions.jsonl")
    metrics = _load_jsonl(output_dir / "per_example_metrics.jsonl")
    summary, rejections = build_p3_diagnostic(predictions, metrics, metric_names)

    output = args.output or output_dir / "p3_proofreading_diagnostic.tsv"
    save_dataframe_tsv(summary, output)
    rejection_output = output.with_name(f"{output.stem}_rejections.tsv")
    save_dataframe_tsv(rejections, rejection_output)
    save_json(
        {
            "dataset": args.dataset,
            "model": args.model,
            "metrics": list(metric_names),
            "summary": summary.to_dict(orient="records"),
            "rejection_reasons": rejections.to_dict(orient="records"),
        },
        output.with_suffix(".json"),
    )
    print(f"Wrote P3 proofreading diagnostic to {output}")


if __name__ == "__main__":
    main()
