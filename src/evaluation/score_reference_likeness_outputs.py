"""Compare target-side reference-likeness classifiers on aligned MT outputs."""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.classifiers.reward import TargetSideReferenceLikenessScorer
from src.utils.io import save_dataframe_csv, save_json, save_jsonl

_DEFAULT_CORRELATION_METRICS = (
    "sentence_chrf_pp",
    "sentence_bleu",
    "astred_ted",
    "astred_seq_cross",
    "astred_source_prediction_ted",
    "astred_source_prediction_seq_cross",
    "prediction_token_count",
)


def _parse_mapping(values: list[str], kind: str) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{kind} must use NAME=PATH syntax: {value}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError(f"Invalid {kind} name: {name!r}")
        if name in parsed:
            raise ValueError(f"Duplicate {kind} name: {name}")
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Missing {kind} path: {path}")
        parsed[name] = path
    return parsed


def _read_predictions(path: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sentence_id = str(row.get("sentence_id") or row.get("id") or "").strip()
            required = {
                "sentence_id": sentence_id,
                "source": str(row.get("source", "")).strip(),
                "reference": str(row.get("reference", row.get("target", ""))).strip(),
                "prediction": str(row.get("prediction", "")).strip(),
            }
            missing = [key for key, value in required.items() if not value]
            if missing:
                raise ValueError(f"{path}:{line_number} has empty fields: {missing}")
            records.append({**row, **required})
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError(f"Prediction file is empty: {path}")
    if frame["sentence_id"].duplicated().any():
        raise ValueError(f"Prediction file contains duplicate sentence IDs: {path}")
    return frame.set_index("sentence_id", drop=False)


def _validate_alignment(frames: dict[str, pd.DataFrame]) -> None:
    first_name, first = next(iter(frames.items()))
    for name, frame in list(frames.items())[1:]:
        if frame.index.tolist() != first.index.tolist():
            raise ValueError(f"Sentence IDs differ between {first_name} and {name}.")
        for column in ("source", "reference"):
            if not frame[column].equals(first[column]):
                raise ValueError(f"{column} texts differ between {first_name} and {name}.")


def _finite_correlation(left: pd.Series, right: pd.Series, method: str) -> tuple[int, float | None]:
    pair = pd.DataFrame({"left": left, "right": right}).apply(pd.to_numeric, errors="coerce")
    pair = pair.replace([math.inf, -math.inf], pd.NA).dropna()
    if len(pair) < 3 or pair["left"].nunique() < 2 or pair["right"].nunique() < 2:
        return len(pair), None
    return len(pair), float(pair["left"].corr(pair["right"], method=method))


def score_outputs(
    systems: dict[str, Path],
    classifiers: dict[str, Path],
    output_dir: Path,
    *,
    batch_size: int,
    max_length: int,
    correlation_metrics: tuple[str, ...],
) -> dict[str, Any]:
    """Score aligned predictions and references, then summarize classifier agreement."""
    frames = {name: _read_predictions(path) for name, path in systems.items()}
    _validate_alignment(frames)
    reference = next(iter(frames.values()))["reference"].tolist()

    prediction_scores: dict[tuple[str, str], list[float]] = {}
    reference_scores: dict[str, list[float]] = {}
    for classifier_name, model_dir in classifiers.items():
        scorer = TargetSideReferenceLikenessScorer(model_dir, max_length=max_length)
        reference_scores[classifier_name] = scorer.score(reference, batch_size=batch_size)
        for system_name, frame in frames.items():
            prediction_scores[(classifier_name, system_name)] = scorer.score(
                frame["prediction"].tolist(), batch_size=batch_size
            )
        del scorer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    correlations: list[dict[str, Any]] = []
    scored_paths: dict[str, str] = {}
    classifier_names = list(classifiers)

    for system_name, frame in frames.items():
        scored = frame[["sentence_id", "source", "reference", "prediction"]].copy()
        for classifier_name in classifier_names:
            prediction_column = f"{classifier_name}_prediction_score"
            reference_column = f"{classifier_name}_reference_score"
            delta_column = f"{classifier_name}_prediction_minus_reference"
            scored[prediction_column] = prediction_scores[(classifier_name, system_name)]
            scored[reference_column] = reference_scores[classifier_name]
            scored[delta_column] = scored[prediction_column] - scored[reference_column]
            summaries.append(
                {
                    "system": system_name,
                    "classifier": classifier_name,
                    "rows": len(scored),
                    "prediction_mean": float(scored[prediction_column].mean()),
                    "prediction_std": float(scored[prediction_column].std(ddof=1)),
                    "prediction_median": float(scored[prediction_column].median()),
                    "reference_mean": float(scored[reference_column].mean()),
                    "mean_prediction_minus_reference": float(scored[delta_column].mean()),
                    "classifier_max_length": max_length,
                }
            )

        analysis_path = systems[system_name].with_name("analysis_summary.json")
        if analysis_path.exists():
            analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
            analysis["prediction_classifier_reference_likeness"] = {
                name: float(scored[f"{name}_prediction_score"].mean())
                for name in classifier_names
            }
            analysis["reference_classifier_reference_likeness"] = {
                name: float(scored[f"{name}_reference_score"].mean())
                for name in classifier_names
            }
            save_json(analysis, analysis_path)

        if len(classifier_names) > 1:
            for left_index, left_name in enumerate(classifier_names[:-1]):
                for right_name in classifier_names[left_index + 1 :]:
                    for method in ("pearson", "spearman"):
                        rows, value = _finite_correlation(
                            scored[f"{left_name}_prediction_score"],
                            scored[f"{right_name}_prediction_score"],
                            method,
                        )
                        correlations.append(
                            {
                                "system": system_name,
                                "classifier": left_name,
                                "comparison": right_name,
                                "method": method,
                                "rows": rows,
                                "correlation": value,
                            }
                        )

        metrics_path = systems[system_name].with_name("per_example_metrics.jsonl")
        if metrics_path.exists():
            metrics = pd.read_json(metrics_path, lines=True).set_index("sentence_id", drop=False)
            if metrics.index.tolist() != frame.index.tolist():
                raise ValueError(f"Metric rows are not aligned with predictions: {metrics_path}")
            for classifier_name in classifier_names:
                classifier_values = scored[f"{classifier_name}_prediction_score"]
                for metric in correlation_metrics:
                    if metric not in metrics:
                        continue
                    for method in ("pearson", "spearman"):
                        rows, value = _finite_correlation(classifier_values, metrics[metric], method)
                        correlations.append(
                            {
                                "system": system_name,
                                "classifier": classifier_name,
                                "comparison": metric,
                                "method": method,
                                "rows": rows,
                                "correlation": value,
                            }
                        )

        records = scored.reset_index(drop=True).to_dict(orient="records")
        scored_path = output_dir / f"{system_name}_classifier_scores.jsonl"
        save_jsonl(records, scored_path)
        scored_paths[system_name] = str(scored_path)

    summary_path = save_dataframe_csv(pd.DataFrame(summaries), output_dir / "classifier_summary.csv")
    correlation_path = save_dataframe_csv(
        pd.DataFrame(correlations), output_dir / "classifier_correlations.csv"
    )
    metadata = {
        "systems": {name: str(path) for name, path in systems.items()},
        "classifiers": {name: str(path) for name, path in classifiers.items()},
        "rows_per_system": len(next(iter(frames.values()))),
        "batch_size": batch_size,
        "max_length": max_length,
        "correlation_metrics": list(correlation_metrics),
        "scored_outputs": scored_paths,
        "summary_csv": str(summary_path),
        "correlations_csv": str(correlation_path),
        "interpretation": "Higher classifier scores indicate more reference-like target text; they are not direct adequacy scores.",
    }
    save_json(metadata, output_dir / "classifier_analysis_metadata.json")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", action="append", required=True, help="NAME=PREDICTIONS_JSONL")
    parser.add_argument("--classifier", action="append", required=True, help="NAME=MODEL_DIRECTORY")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--correlation-metrics",
        default=",".join(_DEFAULT_CORRELATION_METRICS),
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_length < 1:
        raise ValueError("batch-size and max-length must be positive.")
    metrics = tuple(value.strip() for value in args.correlation_metrics.split(",") if value.strip())
    result = score_outputs(
        _parse_mapping(args.system, "system"),
        _parse_mapping(args.classifier, "classifier"),
        args.output_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        correlation_metrics=metrics,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
