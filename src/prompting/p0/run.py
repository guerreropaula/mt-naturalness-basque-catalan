"""Deterministic baseline translation inference."""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.prompting._shared import load_experiment_split, resolve_generation_profile
from src.utils.config import get_dataset_entry, get_experiment_entry, load_generation_config
from src.utils.io import save_json, save_jsonl
from src.utils.model_loader import (
    LoadedModel,
    generate_batch,
    load_inference_model,
    render_translation_prompts,
)

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_processed_split(
    dataset_key: str,
    split: str,
    processed_dir: str | Path = "data/processed",
) -> pd.DataFrame:
    return load_experiment_split(dataset_key, split, processed_dir)


def _load_explicit_evaluation(path: str | Path, dataset_key: str) -> pd.DataFrame:
    evaluation_path = Path(path)
    if not evaluation_path.exists():
        raise FileNotFoundError(f"Explicit evaluation file not found: {evaluation_path}")
    frame = pd.read_json(evaluation_path, lines=True)
    missing = {"source", "target"} - set(frame.columns)
    if missing:
        raise ValueError(
            f"Explicit evaluation file {evaluation_path} is missing columns: {sorted(missing)}"
        )
    if "sentence_id" not in frame.columns:
        if "id" in frame.columns:
            frame["sentence_id"] = frame["id"].astype(str)
        else:
            frame["sentence_id"] = [f"{dataset_key}_{index:06d}" for index in range(1, len(frame) + 1)]
    return frame


def _results_dir(
    dataset_key: str,
    model_key: str,
    results_dir: str | Path = "results/p0_p3/p0",
    output_split: str | None = None,
) -> Path:
    base = Path(results_dir) / dataset_key / model_key
    if output_split is None:
        return base
    if Path(output_split).name != output_split or output_split in {".", ".."}:
        raise ValueError("output_split must be a single directory name")
    return base / output_split


def _predictions_path(
    dataset_key: str,
    model_key: str,
    results_dir: str | Path = "results/p0_p3/p0",
    output_split: str | None = None,
) -> Path:
    return _results_dir(dataset_key, model_key, results_dir, output_split) / "predictions.jsonl"


def _load_existing_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pd.read_json(path, lines=True).to_dict(orient="records")


def _build_generation_runtime_metadata(
    loaded_model: LoadedModel,
    batch_size: int,
    generation_profile_path: str,
    generation_config: dict[str, Any],
    enable_thinking: bool,
) -> dict[str, Any]:
    return {
        "backend": loaded_model.backend,
        "model_family": loaded_model.model_entry.family,
        "batch_size": batch_size,
        "seed": int(generation_config["seed"]),
        "resume": bool(generation_config.get("resume", False)),
        "generation_profile": generation_profile_path,
        "enable_thinking": bool(enable_thinking),
    }


def run_baseline_inference(
    dataset_key: str,
    model_key: str,
    experiments_config_path: str | Path = "configs/experiments.yaml",
    dataset_config_path: str | Path = "configs/datasets.yaml",
    generation_config_path: str | Path = "configs/generation.yaml",
    processed_dir: str | Path = "data/processed",
    results_dir: str | Path = "results/p0_p3/p0",
    split: str = "test",
    output_split: str | None = None,
    evaluation_file: str | Path | None = None,
    output_dataset_key: str | None = None,
    limit: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run deterministic baseline translation and persist predictions."""
    experiment_entry = get_experiment_entry("p0_baseline", experiments_config_path)
    dataset_config = get_dataset_entry(dataset_key, dataset_config_path)
    generation_root = load_generation_config(generation_config_path)
    generation_profile_path = str(experiment_entry["generation_profile"])
    generation_config = resolve_generation_profile(generation_root, generation_profile_path)
    prompt_spec = dict(experiment_entry["prompt"])
    enable_thinking = bool(prompt_spec.get("enable_thinking", False))

    loaded_model = load_inference_model(
        model_key,
        generation_config=generation_config,
    )
    loaded_model.runtime_metadata["enable_thinking"] = enable_thinking

    split_df = (
        _load_explicit_evaluation(evaluation_file, output_dataset_key or dataset_key)
        if evaluation_file is not None
        else _load_processed_split(dataset_key, split, processed_dir)
    )
    if limit is not None:
        split_df = split_df.head(limit).copy()

    result_dataset_key = output_dataset_key or dataset_key
    output_dir = _results_dir(result_dataset_key, model_key, results_dir, output_split)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = _predictions_path(result_dataset_key, model_key, results_dir, output_split)
    metadata_path = output_dir / "run_metadata.json"

    existing_records = (
        []
        if force
        else (_load_existing_predictions(predictions_path) if generation_config.get("resume") else [])
    )
    existing_sentence_ids = {record["sentence_id"] for record in existing_records}
    if (
        predictions_path.exists()
        and existing_records
        and not generation_config.get("resume")
        and not force
    ):
        raise FileExistsError(
            f"Predictions already exist at {predictions_path}; use --force or enable resume."
        )

    candidate_rows = (
        split_df
        if force or not existing_sentence_ids
        else split_df[~split_df["sentence_id"].isin(existing_sentence_ids)]
    )
    batch_size = int(generation_config["batch_size"])
    runtime_records: list[dict[str, Any]] = list(existing_records)

    started_at_utc = _utc_now_iso()
    start_time = time.perf_counter()
    for batch_start in range(0, len(candidate_rows), batch_size):
        batch = candidate_rows.iloc[batch_start : batch_start + batch_size]
        source_texts = batch["source"].astype(str).tolist()
        prompts = render_translation_prompts(
            loaded_model,
            source_texts,
            target_lang=dataset_config["target_lang"],
            enable_thinking=enable_thinking,
            prompt_spec=prompt_spec,
        )
        predictions = generate_batch(loaded_model, prompts, generation_config)
        for (_, row), prompt_text, prediction in zip(batch.iterrows(), prompts, predictions):
            runtime_records.append(
                {
                    "sentence_id": row["sentence_id"],
                    "source": row["source"],
                    "reference": row.get("target", row.get("target_raw")),
                    "prediction": prediction,
                    "model_key": model_key,
                    "model_id": loaded_model.model_entry.hf_id,
                    "experiment_key": "p0_baseline",
                    "experiment_family": experiment_entry["family"],
                    "experiment_track": experiment_entry.get("track"),
                    "prompt_variant": prompt_spec.get("label", "p0_baseline"),
                    "prompt": prompt_text,
                    "generation_config": generation_config,
                    "runtime": {
                        "batch_runtime_seconds": loaded_model.runtime_metadata.get(
                            "last_batch_runtime_seconds"
                        ),
                    },
                }
            )

    runtime_records = sorted(runtime_records, key=lambda record: record["sentence_id"])
    save_jsonl(runtime_records, predictions_path)

    elapsed = time.perf_counter() - start_time
    finished_at_utc = _utc_now_iso()
    metadata = {
        "experiment_key": "p0_baseline",
        "experiment_family": experiment_entry["family"],
        "experiment_track": experiment_entry.get("track"),
        "experiment_mode": experiment_entry["mode"],
        "experiment_description": experiment_entry["description"],
        "prompt_spec": prompt_spec,
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "dataset_key": dataset_key,
        "result_dataset_key": result_dataset_key,
        "split": split,
        "output_split": output_split,
        "explicit_evaluation_file": str(evaluation_file) if evaluation_file is not None else None,
        "model_key": model_key,
        "model_id": loaded_model.model_entry.hf_id,
        "total_rows": int(len(split_df)),
        "generated_rows": int(len(runtime_records)),
        "runtime_seconds": elapsed,
        "generation": _build_generation_runtime_metadata(
            loaded_model,
            batch_size,
            generation_profile_path,
            generation_config,
            enable_thinking,
        ),
    }
    save_json(metadata, metadata_path)
    return {
        "predictions_path": str(predictions_path),
        "metadata_path": str(metadata_path),
        "generated_rows": int(len(runtime_records)),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic baseline translation inference."
    )
    parser.add_argument("--dataset", required=True, help="Dataset key from configs/datasets.yaml")
    parser.add_argument("--model", required=True, help="Model key from configs/models.yaml")
    parser.add_argument("--experiments-config", default="configs/experiments.yaml")
    parser.add_argument("--dataset-config", default="configs/datasets.yaml")
    parser.add_argument("--generation-config", default="configs/generation.yaml")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--results-dir", default="results/p0_p3/p0")
    parser.add_argument("--split", default="test", choices=["test"])
    parser.add_argument("--output-split", default=None, help="Optional isolated output directory label, for example flores_test.")
    parser.add_argument("--evaluation-file", default=None, help="Immutable JSONL with id/source/target fields.")
    parser.add_argument("--output-dataset-key", default=None, help="Dataset label used only for result paths.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_baseline_inference(
        dataset_key=args.dataset,
        model_key=args.model,
        experiments_config_path=args.experiments_config,
        dataset_config_path=args.dataset_config,
        generation_config_path=args.generation_config,
        processed_dir=args.processed_dir,
        results_dir=args.results_dir,
        split=args.split,
        output_split=args.output_split,
        evaluation_file=args.evaluation_file,
        output_dataset_key=args.output_dataset_key,
        limit=args.limit,
        force=args.force,
    )


if __name__ == "__main__":
    main()
