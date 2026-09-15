"""Run traceable single-pass translation experiments with editable system prompts."""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


from src.prompting._shared import (
    ExperimentError,
    experiment_output_dir,
    load_existing_records,
    load_experiment_split as load_processed_split,
    resolve_generation_profile,
)
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


def _build_runtime_metadata(
    loaded_model: LoadedModel,
    generation_profile_path: str,
    generation_config: dict[str, Any],
    enable_thinking: bool,
) -> dict[str, Any]:
    return {
        "backend": loaded_model.backend,
        "model_family": loaded_model.model_entry.family,
        "batch_size": int(generation_config["batch_size"]),
        "seed": int(generation_config["seed"]),
        "resume": bool(generation_config.get("resume", False)),
        "generation_profile": generation_profile_path,
        "enable_thinking": bool(enable_thinking),
    }


def run_system_prompt_experiment(
    experiment_key: str,
    dataset_key: str,
    model_key: str,
    experiments_config_path: str | Path = "configs/experiments.yaml",
    dataset_config_path: str | Path = "configs/datasets.yaml",
    generation_config_path: str | Path = "configs/generation.yaml",
    processed_dir: str | Path = "data/processed",
    results_dir: str | Path = "results/p0_p3",
    split: str = "test",
    limit: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run a single-pass prompt experiment and persist traceable outputs."""
    experiment_entry = get_experiment_entry(experiment_key, experiments_config_path)
    if experiment_entry["mode"] != "single_pass":
        raise ExperimentError(
            f"Experiment '{experiment_key}' uses mode '{experiment_entry['mode']}', not 'single_pass'"
        )

    dataset_entry = get_dataset_entry(dataset_key, dataset_config_path)
    generation_root = load_generation_config(generation_config_path)
    generation_profile_path = str(experiment_entry["generation_profile"])
    generation_config = resolve_generation_profile(generation_root, generation_profile_path)
    prompt_spec = dict(experiment_entry["prompt"])
    enable_thinking = bool(prompt_spec.get("enable_thinking", False))

    loaded_model = load_inference_model(model_key, generation_config=generation_config)
    loaded_model.runtime_metadata["enable_thinking"] = enable_thinking

    split_df = load_processed_split(dataset_key, split, processed_dir)
    if limit is not None:
        split_df = split_df.head(limit).copy()

    output_dir = experiment_output_dir(
        results_dir, experiment_entry, experiment_key, dataset_key, model_key
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    metadata_path = output_dir / "run_metadata.json"

    resume_enabled = bool(generation_config.get("resume", False))
    existing_records = [] if force else (load_existing_records(predictions_path) if resume_enabled else [])
    existing_sentence_ids = {record["sentence_id"] for record in existing_records}
    if predictions_path.exists() and existing_records and not resume_enabled and not force:
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
            target_lang=dataset_entry["target_lang"],
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
                    "experiment_key": experiment_key,
                    "experiment_family": experiment_entry["family"],
                    "experiment_track": experiment_entry.get("track"),
                    "prompt_variant": prompt_spec.get("label", experiment_key),
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
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "experiment_key": experiment_key,
        "experiment_family": experiment_entry["family"],
        "experiment_track": experiment_entry.get("track"),
        "experiment_mode": experiment_entry["mode"],
        "experiment_description": experiment_entry["description"],
        "dataset_key": dataset_key,
        "split": split,
        "model_key": model_key,
        "model_id": loaded_model.model_entry.hf_id,
        "total_rows": int(len(split_df)),
        "generated_rows": int(len(runtime_records)),
        "runtime_seconds": elapsed,
        "prompt_spec": prompt_spec,
        "generation": _build_runtime_metadata(
            loaded_model,
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
        description="Run a single-pass system-prompt translation experiment."
    )
    parser.add_argument(
        "--experiment", required=True, help="Experiment key from configs/experiments.yaml"
    )
    parser.add_argument("--dataset", required=True, help="Dataset key from configs/datasets.yaml")
    parser.add_argument("--model", required=True, help="Model key from configs/models.yaml")
    parser.add_argument("--experiments-config", default="configs/experiments.yaml")
    parser.add_argument("--dataset-config", default="configs/datasets.yaml")
    parser.add_argument("--generation-config", default="configs/generation.yaml")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--results-dir", default="results/p0_p3")
    parser.add_argument("--split", default="test", choices=["test"])
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
    run_system_prompt_experiment(
        experiment_key=args.experiment,
        dataset_key=args.dataset,
        model_key=args.model,
        experiments_config_path=args.experiments_config,
        dataset_config_path=args.dataset_config,
        generation_config_path=args.generation_config,
        processed_dir=args.processed_dir,
        results_dir=args.results_dir,
        split=args.split,
        limit=args.limit,
        force=args.force,
    )


if __name__ == "__main__":
    main()
