"""Run the P0-P3 prompting experiments from one configuration-driven entry point."""

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

import pandas as pd

from src.utils.errors import PipelineError
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

EXPERIMENT_KEYS = {
    "p0": "p0_baseline",
    "p1": "p1_naturalness",
    "p2": "p2_polishing",
    "p3": "p3_step_by_step",
}
STAGE_KEYS = ("research", "draft", "refinement", "proofreading")
TRANSLATION_BLOCK_RE = re.compile(
    r"^\s*<translation>\s*(.*?)\s*</translation>", re.IGNORECASE | re.DOTALL
)
TRANSLATION_TAG_RE = re.compile(r"</?translation>", re.IGNORECASE)
COMMENTARY_MARKERS = (
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
ALTERNATIVE_MARKERS = ("\nalternatively", "\nor, ", "\notherwise", "\nanother option")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_split(
    dataset_key: str,
    split: str,
    processed_dir: str | Path,
    training_dir: str | Path,
) -> pd.DataFrame:
    candidates = [
        Path(training_dir) / dataset_key / "eval" / f"{split}.jsonl",
        Path(training_dir) / dataset_key / "eval" / f"{split}.parquet",
        Path(processed_dir) / dataset_key / f"{split}.jsonl",
        Path(processed_dir) / dataset_key / f"{split}.parquet",
    ]
    for path in candidates:
        if path.exists():
            frame = (
                pd.read_json(path, lines=True) if path.suffix == ".jsonl" else pd.read_parquet(path)
            )
            if "sentence_id" not in frame and "id" in frame:
                frame["sentence_id"] = frame["id"].astype(str)
            return frame
    raise PipelineError(f"No {split!r} split found for dataset {dataset_key!r}.")


def _load_evaluation_file(path: str | Path, dataset_key: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise PipelineError(f"Evaluation file not found: {path}")
    frame = pd.read_json(path, lines=True)
    missing = {"source", "target"} - set(frame.columns)
    if missing:
        raise PipelineError(f"Evaluation file is missing columns: {sorted(missing)}")
    if "sentence_id" not in frame:
        if "id" in frame:
            frame["sentence_id"] = frame["id"].astype(str)
        else:
            frame["sentence_id"] = [
                f"{dataset_key}_{index:06d}" for index in range(1, len(frame) + 1)
            ]
    return frame


def _generation_profile(config: dict[str, Any], path: str) -> dict[str, Any]:
    value: Any = config
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise PipelineError(f"Unknown generation profile {path!r}.")
        value = value[part]
    if not isinstance(value, dict):
        raise PipelineError(f"Generation profile {path!r} is not a mapping.")
    baseline = config.get("baseline", {})
    return {**baseline, **value} if path != "baseline" else dict(value)


def _output_dir(
    results_dir: str | Path,
    experiment_key: str,
    dataset_key: str,
    model_key: str,
    output_split: str | None,
) -> Path:
    phase = experiment_key.split("_", 1)[0]
    output = Path(results_dir) / phase / dataset_key / model_key
    if output_split:
        if Path(output_split).name != output_split or output_split in {".", ".."}:
            raise PipelineError("output_split must be a single directory name.")
        output /= output_split
    return output


def _load_language_model(config_path: str | Path) -> tuple[Any | None, float]:
    """Load fastText for P2 and P3 output validation when it is available."""
    try:
        config = load_preprocessing_config(config_path)["language_id"]
        return load_fasttext_model(str(config["model_path"])), float(config["confidence_threshold"])
    except Exception as exc:  # optional runtime dependency
        logger.warning("Output language validation disabled: %s", exc)
        return None, 1.0


def _normalise_for_markers(text: str) -> str:
    text = unicodedata.normalize("NFKD", " ".join(text.lower().split()))
    return (
        text.encode("ascii", "ignore").decode("ascii").translate(str.maketrans("", "", "'.,:;!?"))
    )


def _validated_translation(
    raw: str,
    fallback: str,
    target_lang: str,
    language_model: Any | None,
    confidence_threshold: float,
) -> tuple[str, bool, str | None]:
    """Extract the first tagged translation or return the previous-stage output."""
    raw, fallback = str(raw).strip(), str(fallback).strip()
    match = TRANSLATION_BLOCK_RE.match(raw)
    if match is None:
        return fallback, False, "missing_translation_tags"
    candidate = match.group(1).strip()
    if not candidate:
        return fallback, False, "empty_translation"
    if TRANSLATION_TAG_RE.search(candidate):
        return fallback, False, "nested_translation_tags"
    if any(marker in _normalise_for_markers(candidate) for marker in COMMENTARY_MARKERS):
        return fallback, False, "commentary_marker"
    if any(marker in candidate.lower() for marker in ALTERNATIVE_MARKERS):
        return fallback, False, "alternatives"
    if len(candidate.split()) > max(24, len(fallback.split()) * 2 + 12):
        return fallback, False, "too_long"
    if language_model is not None and len(candidate.split()) >= 5:
        predicted_lang, confidence = predict_language(language_model, candidate)
        if predicted_lang != target_lang and confidence >= confidence_threshold:
            return fallback, False, "wrong_language"
    return candidate, True, None


def _stage_specs(experiment: dict[str, Any]) -> list[dict[str, Any]]:
    stages = experiment.get("stages", [])
    if tuple(str(stage.get("key")) for stage in stages) != STAGE_KEYS:
        raise PipelineError(f"P3 stages must be ordered as {list(STAGE_KEYS)}.")
    return [dict(stage["prompt"], key=stage["key"]) for stage in stages]


def _record_base(
    row: pd.Series,
    model: LoadedModel,
    model_key: str,
    experiment_key: str,
    experiment: dict[str, Any],
) -> dict[str, Any]:
    return {
        "sentence_id": row["sentence_id"],
        "source": row["source"],
        "reference": row.get("target", row.get("target_raw")),
        "model_key": model_key,
        "model_id": model.model_entry.hf_id,
        "experiment_key": experiment_key,
        "experiment_family": experiment["family"],
        "experiment_track": experiment.get("track"),
    }


def _single_pass_batch(
    batch: pd.DataFrame,
    model: LoadedModel,
    model_key: str,
    experiment_key: str,
    experiment: dict[str, Any],
    target_lang: str,
    generation: dict[str, Any],
) -> list[dict[str, Any]]:
    prompt_spec = dict(experiment["prompt"])
    prompts = render_translation_prompts(
        model,
        batch["source"].astype(str).tolist(),
        target_lang=target_lang,
        enable_thinking=bool(prompt_spec.get("enable_thinking", False)),
        prompt_spec=prompt_spec,
    )
    predictions = generate_batch(model, prompts, generation)
    records = []
    for (_, row), prompt, prediction in zip(batch.iterrows(), prompts, predictions):
        record = _record_base(row, model, model_key, experiment_key, experiment)
        record.update(
            prediction=prediction,
            prompt_variant=prompt_spec.get("label", experiment_key),
            prompt=prompt,
            generation_config=generation,
            runtime={
                "batch_runtime_seconds": model.runtime_metadata.get("last_batch_runtime_seconds")
            },
        )
        records.append(record)
    return records


def _self_polish_batch(
    batch: pd.DataFrame,
    model: LoadedModel,
    model_key: str,
    experiment_key: str,
    experiment: dict[str, Any],
    target_lang: str,
    initial_generation: dict[str, Any],
    refinement_generation: dict[str, Any],
    language_model: Any | None,
    confidence_threshold: float,
) -> list[dict[str, Any]]:
    sources = batch["source"].astype(str).tolist()
    initial_spec = dict(experiment["initial_prompt"])
    refinement_spec = dict(experiment["refinement_prompt"])
    initial_prompts = render_translation_prompts(
        model,
        sources,
        target_lang=target_lang,
        enable_thinking=bool(initial_spec.get("enable_thinking", False)),
        prompt_spec=initial_spec,
    )
    drafts = generate_batch(model, initial_prompts, initial_generation)
    refinement_prompts = render_translation_prompts(
        model,
        sources,
        target_lang=target_lang,
        enable_thinking=bool(refinement_spec.get("enable_thinking", False)),
        prompt_spec=refinement_spec,
        extra_template_contexts=[
            {"baseline_prediction": draft, "draft_translation": draft} for draft in drafts
        ],
    )
    refinements = generate_batch(model, refinement_prompts, refinement_generation)
    records = []
    for row_item, initial_prompt, refinement_prompt, draft, raw in zip(
        batch.iterrows(), initial_prompts, refinement_prompts, drafts, refinements
    ):
        _, row = row_item
        prediction, accepted, reason = _validated_translation(
            raw, draft, target_lang, language_model, confidence_threshold
        )
        record = _record_base(row, model, model_key, experiment_key, experiment)
        record.update(
            draft_prediction=draft,
            raw_refinement_prediction=raw,
            prediction=prediction,
            refinement_accepted=accepted,
            refinement_rejection_reason=reason,
            initial_prompt_variant=initial_spec.get("label", f"{experiment_key}_initial"),
            refinement_prompt_variant=refinement_spec.get("label", f"{experiment_key}_refinement"),
            initial_prompt=initial_prompt,
            refinement_prompt=refinement_prompt,
            generation_config={
                "initial_pass": initial_generation,
                "refinement_pass": refinement_generation,
            },
            runtime={
                "batch_runtime_seconds": model.runtime_metadata.get("last_batch_runtime_seconds")
            },
        )
        records.append(record)
    return records


def _step_by_step_batch(
    batch: pd.DataFrame,
    model: LoadedModel,
    model_key: str,
    experiment_key: str,
    experiment: dict[str, Any],
    target_lang: str,
    generation: dict[str, Any],
    language_model: Any | None,
    confidence_threshold: float,
) -> list[dict[str, Any]]:
    sources = batch["source"].astype(str).tolist()
    specs = _stage_specs(experiment)
    prompts_by_stage: dict[str, list[str]] = {}
    outputs: dict[str, list[str]] = {}
    for spec in specs:
        key = str(spec["key"])
        contexts = []
        for index in range(len(sources)):
            context = {
                "research": outputs.get("research", [None] * len(sources))[index],
                "draft_translation": outputs.get("draft", [None] * len(sources))[index],
                "refined_translation": outputs.get("refinement", [None] * len(sources))[index],
            }
            contexts.append({name: value for name, value in context.items() if value is not None})
        prompts_by_stage[key] = render_translation_prompts(
            model,
            sources,
            target_lang=target_lang,
            enable_thinking=bool(spec.get("enable_thinking", False)),
            prompt_spec=spec,
            extra_template_contexts=contexts,
        )
        outputs[key] = generate_batch(model, prompts_by_stage[key], generation)

    records = []
    for index, (_, row) in enumerate(batch.iterrows()):
        prediction, accepted, reason = _validated_translation(
            outputs["proofreading"][index],
            outputs["refinement"][index],
            target_lang,
            language_model,
            confidence_threshold,
        )
        record = _record_base(row, model, model_key, experiment_key, experiment)
        record.update(
            research=outputs["research"][index],
            draft_translation=outputs["draft"][index],
            refined_translation=outputs["refinement"][index],
            raw_proofreading_prediction=outputs["proofreading"][index],
            prediction=prediction,
            proofreading_accepted=accepted,
            proofreading_rejection_reason=reason,
            prompt_variant=experiment.get("label", experiment_key),
            prompts={key: values[index] for key, values in prompts_by_stage.items()},
            generation_config=generation,
            runtime={
                "batch_runtime_seconds": model.runtime_metadata.get("last_batch_runtime_seconds")
            },
        )
        records.append(record)
    return records


def _validation_summary(
    records: list[dict[str, Any]], accepted_key: str, reason_key: str, fallback_key: str
) -> dict[str, Any]:
    validated = [record for record in records if accepted_key in record]
    accepted = sum(bool(record[accepted_key]) for record in validated)
    return {
        "accepted": accepted,
        fallback_key: len(validated) - accepted,
        "rejection_reasons": dict(
            Counter(
                str(record[reason_key]) for record in validated if not record[accepted_key]
            ).most_common()
        ),
    }


def run_prompting_experiment(
    experiment: str,
    dataset_key: str,
    model_key: str,
    experiments_config_path: str | Path = "configs/experiments.yaml",
    dataset_config_path: str | Path = "configs/datasets.yaml",
    generation_config_path: str | Path = "configs/generation.yaml",
    preprocessing_config_path: str | Path = "configs/preprocessing.yaml",
    processed_dir: str | Path = "data/processed",
    training_dir: str | Path = "data/training",
    results_dir: str | Path = "results/p0_p3",
    split: str = "test",
    evaluation_file: str | Path | None = None,
    output_dataset_key: str | None = None,
    output_split: str | None = None,
    limit: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run one P0-P3 experiment."""
    experiment_key = EXPERIMENT_KEYS.get(experiment, experiment)
    experiment_config = get_experiment_entry(experiment_key, experiments_config_path)
    mode = str(experiment_config["mode"])
    if mode not in {"single_pass", "self_polish", "step_by_step_pipeline"}:
        raise PipelineError(f"Experiment {experiment_key!r} is not a P0-P3 prompting mode.")

    dataset = get_dataset_entry(dataset_key, dataset_config_path)
    generation_root = load_generation_config(generation_config_path)
    if mode == "self_polish":
        initial_profile = str(experiment_config["initial_generation_profile"])
        refinement_profile = str(experiment_config["refinement_generation_profile"])
        initial_generation = _generation_profile(generation_root, initial_profile)
        refinement_generation = _generation_profile(generation_root, refinement_profile)
        generation = initial_generation
        resume = bool(generation.get("resume") or refinement_generation.get("resume"))
    else:
        profile = str(experiment_config["generation_profile"])
        generation = _generation_profile(generation_root, profile)
        initial_generation = refinement_generation = generation
        resume = bool(generation.get("resume"))

    model = load_inference_model(model_key, generation_config=generation)
    frame = (
        _load_evaluation_file(evaluation_file, output_dataset_key or dataset_key)
        if evaluation_file
        else _load_split(dataset_key, split, processed_dir, training_dir)
    )
    if limit is not None:
        frame = frame.head(limit).copy()

    result_dataset = output_dataset_key or dataset_key
    output_dir = _output_dir(results_dir, experiment_key, result_dataset, model_key, output_split)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    metadata_path = output_dir / "run_metadata.json"
    existing = (
        []
        if force or not resume or not predictions_path.exists()
        else pd.read_json(predictions_path, lines=True).to_dict(orient="records")
    )
    if predictions_path.exists() and not resume and not force:
        raise PipelineError(f"{predictions_path} already exists; use --force to replace it.")
    completed_ids = {record["sentence_id"] for record in existing}
    pending = frame if force else frame[~frame["sentence_id"].isin(completed_ids)]

    language_model: Any | None = None
    confidence_threshold = 1.0
    if mode in {"self_polish", "step_by_step_pipeline"}:
        language_model, confidence_threshold = _load_language_model(preprocessing_config_path)

    records = list(existing)
    started_at = _utc_now()
    start_time = time.perf_counter()
    batch_size = int(generation["batch_size"])
    for start in range(0, len(pending), batch_size):
        batch = pending.iloc[start : start + batch_size]
        if mode == "single_pass":
            new_records = _single_pass_batch(
                batch,
                model,
                model_key,
                experiment_key,
                experiment_config,
                str(dataset["target_lang"]),
                generation,
            )
        elif mode == "self_polish":
            new_records = _self_polish_batch(
                batch,
                model,
                model_key,
                experiment_key,
                experiment_config,
                str(dataset["target_lang"]),
                initial_generation,
                refinement_generation,
                language_model,
                confidence_threshold,
            )
        else:
            new_records = _step_by_step_batch(
                batch,
                model,
                model_key,
                experiment_key,
                experiment_config,
                str(dataset["target_lang"]),
                generation,
                language_model,
                confidence_threshold,
            )
        records.extend(new_records)

    records.sort(key=lambda record: record["sentence_id"])
    save_jsonl(records, predictions_path)
    metadata: dict[str, Any] = {
        "experiment_key": experiment_key,
        "experiment_family": experiment_config["family"],
        "experiment_track": experiment_config.get("track"),
        "experiment_mode": mode,
        "experiment_description": experiment_config["description"],
        "dataset_key": dataset_key,
        "result_dataset_key": result_dataset,
        "split": split,
        "output_split": output_split,
        "explicit_evaluation_file": str(evaluation_file) if evaluation_file else None,
        "model_key": model_key,
        "model_id": model.model_entry.hf_id,
        "total_rows": len(frame),
        "generated_rows": len(records),
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "runtime_seconds": time.perf_counter() - start_time,
        "generation": {
            "backend": model.backend,
            "model_family": model.model_entry.family,
            "batch_size": batch_size,
            "seed": int(generation["seed"]),
            "resume": resume,
        },
    }
    if mode == "single_pass":
        metadata["prompt_spec"] = dict(experiment_config["prompt"])
        metadata["generation"]["generation_profile"] = profile
    elif mode == "self_polish":
        metadata["initial_prompt_spec"] = dict(experiment_config["initial_prompt"])
        metadata["refinement_prompt_spec"] = dict(experiment_config["refinement_prompt"])
        metadata["generation"].update(
            initial_generation_profile=initial_profile,
            refinement_generation_profile=refinement_profile,
        )
        metadata["refinement_validation"] = _validation_summary(
            records,
            "refinement_accepted",
            "refinement_rejection_reason",
            "fallback_to_draft",
        )
    else:
        metadata["stages"] = _stage_specs(experiment_config)
        metadata["generation"].update(
            generation_profile=profile,
            stages=list(STAGE_KEYS),
        )
        metadata["proofreading_validation"] = _validation_summary(
            records,
            "proofreading_accepted",
            "proofreading_rejection_reason",
            "fallback_to_refined_translation",
        )
    save_json(metadata, metadata_path)
    return {
        "predictions_path": str(predictions_path),
        "metadata_path": str(metadata_path),
        "generated_rows": len(records),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a P0-P3 prompting experiment.")
    parser.add_argument("--experiment", required=True, choices=tuple(EXPERIMENT_KEYS))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--experiments-config", default="configs/experiments.yaml")
    parser.add_argument("--dataset-config", default="configs/datasets.yaml")
    parser.add_argument("--generation-config", default="configs/generation.yaml")
    parser.add_argument("--preprocessing-config", default="configs/preprocessing.yaml")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--training-dir", default="data/training")
    parser.add_argument("--results-dir", default="results/p0_p3")
    parser.add_argument("--split", default="test", choices=["test"])
    parser.add_argument("--evaluation-file")
    parser.add_argument("--output-dataset-key")
    parser.add_argument("--output-split")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_prompting_experiment(
        experiment=args.experiment,
        dataset_key=args.dataset,
        model_key=args.model,
        experiments_config_path=args.experiments_config,
        dataset_config_path=args.dataset_config,
        generation_config_path=args.generation_config,
        preprocessing_config_path=args.preprocessing_config,
        processed_dir=args.processed_dir,
        training_dir=args.training_dir,
        results_dir=args.results_dir,
        split=args.split,
        evaluation_file=args.evaluation_file,
        output_dataset_key=args.output_dataset_key,
        output_split=args.output_split,
        limit=args.limit,
        force=args.force,
    )


if __name__ == "__main__":
    main()
