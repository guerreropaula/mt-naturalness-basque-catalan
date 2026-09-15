"""Run a two-pass self-polishing translation experiment."""

from __future__ import annotations

import argparse
import logging
import re
import time
import unicodedata
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
from src.data.preprocessing import load_fasttext_model, predict_language
from src.utils.config import (
    get_dataset_entry,
    get_experiment_entry,
    load_generation_config,
    load_preprocessing_config,
)
from src.utils.io import save_json, save_jsonl
from src.utils.model_loader import (
    LoadedModel,
    generate_batch,
    load_inference_model,
    render_translation_prompts,
)

logger = logging.getLogger(__name__)

_TRANSLATION_BLOCK_RE = re.compile(
    r"^\s*<translation>\s*(.*?)\s*</translation>", re.IGNORECASE | re.DOTALL
)
_TRANSLATION_TAG_RE = re.compile(r"</?translation>", re.IGNORECASE)
_COMMENTARY_MARKERS = (
    "this translation",
    "polished translation",
    "the polished",
    "la traduccio correcta",
    "la traduccio es correcta",
    "traduccio millorada",
    "a more idiomatic",
    "a more natural",
    "here is a",
)
_ALTERNATIVE_MARKERS = (
    "\nalternatively",
    "\nor, ",
    "\notherwise",
    "\nanother option",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalized_for_marker_check(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", " ".join(text.lower().split()))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_text.translate(str.maketrans("", "", "'.,:;!?"))


def validate_refinement_prediction(
    raw_prediction: str,
    draft_prediction: str,
    target_lang: str,
    language_model: Any | None,
    language_confidence_threshold: float,
) -> tuple[str, bool, str | None]:
    """Return the first leading tagged translation, ignoring any trailing model commentary."""
    raw_prediction = str(raw_prediction).strip()
    draft_prediction = str(draft_prediction).strip()
    match = _TRANSLATION_BLOCK_RE.match(raw_prediction)
    if match is None:
        return draft_prediction, False, "missing_translation_tags"
    candidate = match.group(1).strip()
    if not candidate:
        return draft_prediction, False, "empty_translation"
    if _TRANSLATION_TAG_RE.search(candidate):
        return draft_prediction, False, "nested_translation_tags"

    normalized = _normalized_for_marker_check(candidate)
    if any(marker in normalized for marker in _COMMENTARY_MARKERS):
        return draft_prediction, False, "commentary_marker"
    if any(marker in candidate.lower() for marker in _ALTERNATIVE_MARKERS):
        return draft_prediction, False, "alternatives"

    candidate_tokens = len(candidate.split())
    draft_tokens = max(len(draft_prediction.split()), 1)
    max_candidate_tokens = max(24, draft_tokens * 2 + 12)
    if candidate_tokens > max_candidate_tokens:
        return draft_prediction, False, "too_long"

    if language_model is not None and candidate_tokens >= 5:
        predicted_lang, confidence = predict_language(language_model, candidate)
        if predicted_lang != target_lang and confidence >= language_confidence_threshold:
            return draft_prediction, False, "wrong_language"

    return candidate, True, None


def load_p2_language_model(preprocessing_config_path: str | Path) -> tuple[Any | None, float]:
    """Load fastText once for refinement validation without blocking P2 if unavailable."""
    try:
        config = load_preprocessing_config(preprocessing_config_path)
        language_config = config["language_id"]
        return (
            load_fasttext_model(str(language_config["model_path"])),
            float(language_config["confidence_threshold"]),
        )
    except Exception as exc:  # pragma: no cover - environment-dependent optional safeguard
        logger.warning("P2 language validation disabled: %s", exc)
        return None, 1.0


def _build_runtime_metadata(
    loaded_model: LoadedModel,
    initial_generation_profile: str,
    refinement_generation_profile: str,
    initial_generation_config: dict[str, Any],
    refinement_generation_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "backend": loaded_model.backend,
        "model_family": loaded_model.model_entry.family,
        "initial_generation_profile": initial_generation_profile,
        "refinement_generation_profile": refinement_generation_profile,
        "initial_batch_size": int(initial_generation_config["batch_size"]),
        "refinement_batch_size": int(refinement_generation_config["batch_size"]),
        "initial_seed": int(initial_generation_config["seed"]),
        "refinement_seed": int(refinement_generation_config["seed"]),
    }


def run_self_polishing_experiment(
    experiment_key: str,
    dataset_key: str,
    model_key: str,
    experiments_config_path: str | Path = "configs/experiments.yaml",
    dataset_config_path: str | Path = "configs/datasets.yaml",
    generation_config_path: str | Path = "configs/generation.yaml",
    preprocessing_config_path: str | Path = "configs/preprocessing.yaml",
    processed_dir: str | Path = "data/processed",
    results_dir: str | Path = "results/p0_p3",
    split: str = "test",
    limit: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run a draft-and-revise self-polishing experiment."""
    experiment_entry = get_experiment_entry(experiment_key, experiments_config_path)
    if experiment_entry["mode"] != "self_polish":
        raise ExperimentError(
            f"Experiment '{experiment_key}' uses mode '{experiment_entry['mode']}', not 'self_polish'"
        )

    dataset_entry = get_dataset_entry(dataset_key, dataset_config_path)
    generation_root = load_generation_config(generation_config_path)
    initial_generation_profile = str(experiment_entry["initial_generation_profile"])
    refinement_generation_profile = str(experiment_entry["refinement_generation_profile"])
    initial_generation_config = resolve_generation_profile(
        generation_root,
        initial_generation_profile,
    )
    refinement_generation_config = resolve_generation_profile(
        generation_root,
        refinement_generation_profile,
    )
    initial_prompt = dict(experiment_entry["initial_prompt"])
    refinement_prompt = dict(experiment_entry["refinement_prompt"])
    language_model, language_confidence_threshold = load_p2_language_model(
        preprocessing_config_path
    )

    loaded_model = load_inference_model(model_key, generation_config=initial_generation_config)

    split_df = load_processed_split(dataset_key, split, processed_dir)
    if limit is not None:
        split_df = split_df.head(limit).copy()

    output_dir = experiment_output_dir(
        results_dir, experiment_entry, experiment_key, dataset_key, model_key
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    metadata_path = output_dir / "run_metadata.json"

    resume_enabled = bool(
        initial_generation_config.get("resume", False)
        or refinement_generation_config.get("resume", False)
    )
    # --force replaces the complete prediction set; it never appends a second
    # full pass to resumed rows for the same sentence IDs.
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
    batch_size = int(initial_generation_config["batch_size"])
    runtime_records: list[dict[str, Any]] = list(existing_records)

    started_at_utc = _utc_now_iso()
    start_time = time.perf_counter()
    for batch_start in range(0, len(candidate_rows), batch_size):
        batch = candidate_rows.iloc[batch_start : batch_start + batch_size]
        source_texts = batch["source"].astype(str).tolist()
        initial_prompts = render_translation_prompts(
            loaded_model,
            source_texts,
            target_lang=dataset_entry["target_lang"],
            enable_thinking=bool(initial_prompt.get("enable_thinking", False)),
            prompt_spec=initial_prompt,
        )
        draft_predictions = generate_batch(
            loaded_model,
            initial_prompts,
            initial_generation_config,
        )
        refinement_contexts = [
            {
                "baseline_prediction": draft_prediction,
                "draft_translation": draft_prediction,
            }
            for draft_prediction in draft_predictions
        ]
        refinement_prompts = render_translation_prompts(
            loaded_model,
            source_texts,
            target_lang=dataset_entry["target_lang"],
            enable_thinking=bool(refinement_prompt.get("enable_thinking", False)),
            prompt_spec=refinement_prompt,
            extra_template_contexts=refinement_contexts,
        )
        final_predictions = generate_batch(
            loaded_model,
            refinement_prompts,
            refinement_generation_config,
        )
        validated_predictions = [
            validate_refinement_prediction(
                raw_prediction,
                draft_prediction,
                target_lang=str(dataset_entry["target_lang"]),
                language_model=language_model,
                language_confidence_threshold=language_confidence_threshold,
            )
            for raw_prediction, draft_prediction in zip(final_predictions, draft_predictions)
        ]

        for row_item, initial_prompt_text, refinement_prompt_text, draft_prediction, raw_prediction, validation in zip(
            batch.iterrows(),
            initial_prompts,
            refinement_prompts,
            draft_predictions,
            final_predictions,
            validated_predictions,
        ):
            _, row = row_item
            final_prediction, refinement_accepted, rejection_reason = validation
            runtime_records.append(
                {
                    "sentence_id": row["sentence_id"],
                    "source": row["source"],
                    "reference": row.get("target", row.get("target_raw")),
                    "draft_prediction": draft_prediction,
                    "raw_refinement_prediction": raw_prediction,
                    "prediction": final_prediction,
                    "refinement_accepted": refinement_accepted,
                    "refinement_rejection_reason": rejection_reason,
                    "model_key": model_key,
                    "model_id": loaded_model.model_entry.hf_id,
                    "experiment_key": experiment_key,
                    "experiment_family": experiment_entry["family"],
                    "experiment_track": experiment_entry.get("track"),
                    "initial_prompt_variant": initial_prompt.get(
                        "label", f"{experiment_key}_initial"
                    ),
                    "refinement_prompt_variant": refinement_prompt.get(
                        "label", f"{experiment_key}_refinement"
                    ),
                    "initial_prompt": initial_prompt_text,
                    "refinement_prompt": refinement_prompt_text,
                    "generation_config": {
                        "initial_pass": initial_generation_config,
                        "refinement_pass": refinement_generation_config,
                    },
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
        record for record in runtime_records if "refinement_accepted" in record
    ]
    accepted_refinements = sum(
        bool(record["refinement_accepted"]) for record in validated_records
    )
    validation_rejections = Counter(
        str(record["refinement_rejection_reason"])
        for record in validated_records
        if not record["refinement_accepted"]
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
        "initial_prompt_spec": initial_prompt,
        "refinement_prompt_spec": refinement_prompt,
        "generation": _build_runtime_metadata(
            loaded_model,
            initial_generation_profile,
            refinement_generation_profile,
            initial_generation_config,
            refinement_generation_config,
        ),
        "refinement_validation": {
            "accepted": accepted_refinements,
            "accepted_display": f"P2 refinement accepted: {accepted_refinements} / {validation_total}",
            "fallback_to_draft": int(validation_total - accepted_refinements),
            "fallback_display": (
                f"P2 fallback to draft: {validation_total - accepted_refinements} / "
                f"{validation_total}"
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
    parser = argparse.ArgumentParser(
        description="Run a two-pass self-polishing translation experiment."
    )
    parser.add_argument(
        "--experiment", required=True, help="Experiment key from configs/experiments.yaml"
    )
    parser.add_argument("--dataset", required=True, help="Dataset key from configs/datasets.yaml")
    parser.add_argument("--model", required=True, help="Model key from configs/models.yaml")
    parser.add_argument("--experiments-config", default="configs/experiments.yaml")
    parser.add_argument("--dataset-config", default="configs/datasets.yaml")
    parser.add_argument("--generation-config", default="configs/generation.yaml")
    parser.add_argument("--preprocessing-config", default="configs/preprocessing.yaml")
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
    run_self_polishing_experiment(
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
