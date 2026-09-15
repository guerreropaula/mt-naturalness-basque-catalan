"""Run a four-stage step-by-step translation pipeline."""

from __future__ import annotations

import argparse
import logging
import time
from collections import Counter
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
from src.prompting.p2.run import (
    _load_p2_language_model as _load_output_language_model,
    _validate_refinement_prediction as _validate_final_translation,
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

_STAGE_KEYS = ("research", "draft", "refinement", "proofreading")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_runtime_metadata(
    loaded_model: LoadedModel,
    generation_profile_path: str,
    generation_config: dict[str, Any],
    stages: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "backend": loaded_model.backend,
        "model_family": loaded_model.model_entry.family,
        "batch_size": int(generation_config["batch_size"]),
        "seed": int(generation_config["seed"]),
        "resume": bool(generation_config.get("resume", False)),
        "generation_profile": generation_profile_path,
        "stages": [str(stage["key"]) for stage in stages],
    }


def _stage_prompt_specs(experiment_entry: dict[str, Any]) -> list[dict[str, Any]]:
    stages = experiment_entry.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ExperimentError("Step-by-step pipeline experiments must define a non-empty stages list")
    stage_keys = [str(stage.get("key")) for stage in stages]
    if tuple(stage_keys) != _STAGE_KEYS:
        raise ExperimentError(f"Step-by-step pipeline stages must be ordered as {list(_STAGE_KEYS)}")
    return [dict(stage["prompt"]) | {"key": stage["key"]} for stage in stages]


def run_step_by_step_experiment(
    experiment_key: str = "p3_step_by_step",
    dataset_key: str | None = None,
    model_key: str | None = None,
    experiments_config_path: str | Path = "configs/experiments.yaml",
    dataset_config_path: str | Path = "configs/datasets.yaml",
    generation_config_path: str | Path = "configs/generation.yaml",
    preprocessing_config_path: str | Path = "configs/preprocessing.yaml",
    processed_dir: str | Path = "data/processed",
    results_dir: str | Path = "results/experiments",
    split: str = "dev",
    limit: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run research, drafting, refinement, and proofreading calls for each source."""
    if dataset_key is None or model_key is None:
        raise ExperimentError("dataset_key and model_key are required")

    experiment_entry = get_experiment_entry(experiment_key, experiments_config_path)
    if experiment_entry["mode"] != "step_by_step_pipeline":
        raise ExperimentError(
            f"Experiment '{experiment_key}' uses mode '{experiment_entry['mode']}', not 'step_by_step_pipeline'"
        )

    dataset_entry = get_dataset_entry(dataset_key, dataset_config_path)
    generation_root = load_generation_config(generation_config_path)
    generation_profile_path = str(experiment_entry["generation_profile"])
    generation_config = resolve_generation_profile(generation_root, generation_profile_path)
    stage_specs = _stage_prompt_specs(experiment_entry)
    language_model, language_confidence_threshold = _load_output_language_model(
        preprocessing_config_path
    )

    loaded_model = load_inference_model(model_key, generation_config=generation_config)

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
        stage_prompts: dict[str, list[str]] = {}
        stage_outputs: dict[str, list[str]] = {}

        for stage_spec in stage_specs:
            stage_key = str(stage_spec["key"])
            contexts: list[dict[str, Any]] = []
            for index in range(len(source_texts)):
                context = {
                    "research": stage_outputs.get("research", [None] * len(source_texts))[index],
                    "draft_translation": stage_outputs.get("draft", [None] * len(source_texts))[
                        index
                    ],
                    "refined_translation": stage_outputs.get(
                        "refinement", [None] * len(source_texts)
                    )[index],
                }
                contexts.append({key: value for key, value in context.items() if value is not None})

            prompts = render_translation_prompts(
                loaded_model,
                source_texts,
                target_lang=dataset_entry["target_lang"],
                enable_thinking=bool(stage_spec.get("enable_thinking", False)),
                prompt_spec=stage_spec,
                extra_template_contexts=contexts,
            )
            stage_prompts[stage_key] = prompts
            stage_outputs[stage_key] = generate_batch(loaded_model, prompts, generation_config)

        validated_predictions = [
            _validate_final_translation(
                raw_prediction,
                refined_translation,
                target_lang=str(dataset_entry["target_lang"]),
                language_model=language_model,
                language_confidence_threshold=language_confidence_threshold,
            )
            for raw_prediction, refined_translation in zip(
                stage_outputs["proofreading"], stage_outputs["refinement"]
            )
        ]

        for index, (_, row) in enumerate(batch.iterrows()):
            final_prediction, proofreading_accepted, rejection_reason = validated_predictions[index]
            runtime_records.append(
                {
                    "sentence_id": row["sentence_id"],
                    "source": row["source"],
                    "reference": row.get("target", row.get("target_raw")),
                    "research": stage_outputs["research"][index],
                    "draft_translation": stage_outputs["draft"][index],
                    "refined_translation": stage_outputs["refinement"][index],
                    "raw_proofreading_prediction": stage_outputs["proofreading"][index],
                    "prediction": final_prediction,
                    "proofreading_accepted": proofreading_accepted,
                    "proofreading_rejection_reason": rejection_reason,
                    "model_key": model_key,
                    "model_id": loaded_model.model_entry.hf_id,
                    "experiment_key": experiment_key,
                    "experiment_family": experiment_entry["family"],
                    "experiment_track": experiment_entry.get("track"),
                    "prompt_variant": experiment_entry.get("label", experiment_key),
                    "prompts": {
                        stage_key: prompts[index] for stage_key, prompts in stage_prompts.items()
                    },
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

    validated_records = [
        record for record in runtime_records if "proofreading_accepted" in record
    ]
    accepted_proofreading = sum(
        bool(record["proofreading_accepted"]) for record in validated_records
    )
    validation_rejections = Counter(
        str(record["proofreading_rejection_reason"])
        for record in validated_records
        if not record["proofreading_accepted"]
    )
    validation_total = len(validated_records)

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
        "stages": stage_specs,
        "generation": _build_runtime_metadata(
            loaded_model,
            generation_profile_path,
            generation_config,
            stage_specs,
        ),
        "proofreading_validation": {
            "accepted": accepted_proofreading,
            "accepted_display": f"P3 proofreading accepted: {accepted_proofreading} / {validation_total}",
            "fallback_to_refined_translation": int(validation_total - accepted_proofreading),
            "fallback_display": (
                "P3 fallback to refined translation: "
                f"{validation_total - accepted_proofreading} / {validation_total}"
            ),
            "rejection_reasons": dict(validation_rejections.most_common()),
        },
    }
    save_json(metadata, metadata_path)
    return {
        "predictions_path": str(predictions_path),
        "metadata_path": str(metadata_path),
        "generated_rows": int(len(runtime_records)),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the four-stage step-by-step translation pipeline.")
    parser.add_argument(
        "--experiment", default="p3_step_by_step", help="Experiment key from configs/experiments.yaml"
    )
    parser.add_argument("--dataset", required=True, help="Dataset key from configs/datasets.yaml")
    parser.add_argument("--model", required=True, help="Model key from configs/models.yaml")
    parser.add_argument("--experiments-config", default="configs/experiments.yaml")
    parser.add_argument("--dataset-config", default="configs/datasets.yaml")
    parser.add_argument("--generation-config", default="configs/generation.yaml")
    parser.add_argument("--preprocessing-config", default="configs/preprocessing.yaml")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--results-dir", default="results/experiments")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
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
    run_step_by_step_experiment(
        experiment_key=args.experiment,
        dataset_key=args.dataset,
        model_key=args.model,
        experiments_config_path=args.experiments_config,
        dataset_config_path=args.dataset_config,
        generation_config_path=args.generation_config,
        preprocessing_config_path=args.preprocessing_config,
        processed_dir=args.processed_dir,
        results_dir=args.results_dir,
        split=args.split,
        limit=args.limit,
        force=args.force,
    )


if __name__ == "__main__":
    main()
