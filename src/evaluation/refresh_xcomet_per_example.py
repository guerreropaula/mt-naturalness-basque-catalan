"""Append sentence-level XCOMET scores to an existing evaluation run."""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.analyze_translationese import _build_report, _flatten_summary
from src.evaluation.metrics import (
    DEFAULT_XCOMET_XXL_MODEL,
    _default_comet_gpus,
    _load_comet_model,
    _release_comet_model_cache,
)
from src.utils.io import save_dataframe_csv, save_dataframe_jsonl, save_json

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def extract_segment_scores(outputs: Any) -> list[float]:
    """Extract COMET segment scores from object- or mapping-style outputs."""
    values = None
    if hasattr(outputs, "scores"):
        values = getattr(outputs, "scores")
    elif isinstance(outputs, dict):
        values = outputs.get("scores")
    if values is None:
        raise RuntimeError("XCOMET output does not contain segment scores.")
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    scores = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(scores).all():
        raise RuntimeError("XCOMET returned non-finite segment scores.")
    return scores.tolist()


def compute_xcomet_segment_scores(
    sources: list[str],
    predictions: list[str],
    references: list[str],
    *,
    model_path: str = DEFAULT_XCOMET_XXL_MODEL,
    batch_size: int = 8,
    gpus: int | None = None,
) -> list[float]:
    """Score aligned source, candidate, and reference triples with XCOMET."""
    if not (len(sources) == len(predictions) == len(references)):
        raise ValueError("XCOMET inputs must have equal lengths.")
    if not sources:
        raise ValueError("XCOMET requires at least one input row.")
    model = None
    outputs = None
    try:
        model = _load_comet_model(model_path)
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
        scores = extract_segment_scores(outputs)
        if len(scores) != len(records):
            raise RuntimeError(
                f"XCOMET returned {len(scores)} segment scores for {len(records)} rows."
            )
        return scores
    finally:
        outputs = None
        model = None
        _release_comet_model_cache()


def _load_existing_outputs(output_dir: Path) -> tuple[Path, dict[str, Any], pd.DataFrame, dict[str, Any]]:
    summary_path = output_dir / "analysis_summary.json"
    if not summary_path.exists():
        summary_path = output_dir / "automatic_metrics.json"
    per_example_path = output_dir / "per_example_metrics.jsonl"
    metadata_path = output_dir / "analysis_metadata.json"
    missing = [path for path in (summary_path, per_example_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "XCOMET refresh requires existing evaluation outputs: "
            + ", ".join(str(path) for path in missing)
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    per_example = pd.read_json(per_example_path, lines=True)
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    return summary_path, summary, per_example, metadata


def _validate_alignment(predictions: pd.DataFrame, per_example: pd.DataFrame) -> None:
    required = {"source", "prediction", "reference"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError("Predictions are missing columns: " + ", ".join(missing))
    if len(predictions) != len(per_example):
        raise ValueError(
            f"Row count mismatch: {len(predictions)} predictions and "
            f"{len(per_example)} per-example rows."
        )
    for column in ("sentence_id", "source", "reference", "prediction"):
        if column in predictions and column in per_example:
            left = predictions[column].fillna("").astype(str).tolist()
            right = per_example[column].fillna("").astype(str).tolist()
            if left != right:
                raise ValueError(f"Existing per-example rows are misaligned on {column}.")


def refresh_xcomet_per_example(
    predictions_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    model_path: str = DEFAULT_XCOMET_XXL_MODEL,
    batch_size: int = 8,
    gpus: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Append paired sentence XCOMET scores while preserving all other metrics."""
    predictions_path = Path(predictions_path)
    destination = Path(output_dir) if output_dir else predictions_path.parent
    predictions = pd.read_json(predictions_path, lines=True)
    summary_path, summary, per_example, metadata = _load_existing_outputs(destination)
    _validate_alignment(predictions, per_example)
    if "sentence_xcomet" in per_example and not force:
        raise FileExistsError(
            f"sentence_xcomet already exists in {destination}; pass --force to recompute it."
        )

    started = time.perf_counter()
    started_at = _utc_now_iso()
    scores = compute_xcomet_segment_scores(
        predictions["source"].fillna("").astype(str).tolist(),
        predictions["prediction"].fillna("").astype(str).tolist(),
        predictions["reference"].fillna("").astype(str).tolist(),
        model_path=model_path,
        batch_size=batch_size,
        gpus=gpus,
    )
    per_example["sentence_xcomet"] = scores
    mean_score = float(np.mean(scores))
    summary.setdefault("adequacy", {})["xcomet"] = mean_score
    metadata["xcomet_segment_refresh"] = {
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now_iso(),
        "runtime_seconds": time.perf_counter() - started,
        "predictions_path": str(predictions_path),
        "model": model_path,
        "batch_size": batch_size,
        "gpus": _default_comet_gpus() if gpus is None else gpus,
        "row_count": len(scores),
        "mean_score": mean_score,
        "per_example_column": "sentence_xcomet",
    }
    if summary_path.name == "analysis_summary.json":
        summary.setdefault("report_context", {}).setdefault("optional_requests", {})[
            "xcomet"
        ] = True
        summary["analysis_metadata"] = metadata

    save_json(summary, summary_path)
    save_dataframe_jsonl(per_example, destination / "per_example_metrics.jsonl")
    save_dataframe_csv(per_example, destination / "per_example_metrics.csv")
    save_json(metadata, destination / "analysis_metadata.json")
    if summary_path.name == "analysis_summary.json":
        save_dataframe_csv(_flatten_summary(summary), destination / "analysis_summary.csv")
        if "row_count" in summary and "implemented_metrics_note" in summary:
            (destination / "analysis_report.md").write_text(
                _build_report(summary), encoding="utf-8"
            )
    return {
        "output_dir": str(destination),
        "row_count": len(scores),
        "mean_score": mean_score,
        "column": "sentence_xcomet",
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model", default=DEFAULT_XCOMET_XXL_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    result = refresh_xcomet_per_example(
        args.predictions,
        output_dir=args.output_dir,
        model_path=args.model,
        batch_size=args.batch_size,
        gpus=args.gpus,
        force=args.force,
    )
    logger.info("XCOMET segment refresh complete: %s", result)


if __name__ == "__main__":
    main()
