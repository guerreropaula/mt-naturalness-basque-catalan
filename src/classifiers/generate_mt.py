"""Generate clean P0 machine-translation negatives for classifier source splits."""

from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from src.data.preprocessing import load_fasttext_model, predict_language
from src.utils.config import (
    get_dataset_entry,
    get_experiment_entry,
    load_classifier_config,
    load_generation_config,
    load_preprocessing_config,
)
from src.utils.io import save_json, save_jsonl
from src.utils.model_loader import (
    generate_batch,
    load_inference_model,
    release_inference_model,
    render_translation_prompts,
)

logger = logging.getLogger(__name__)


class NegativeGenerationError(RuntimeError):
    """Raised when clean balanced MT negatives cannot be generated."""


_COMMENTARY_RE = re.compile(
    r"(?:^|\n)\s*(?:polished translation|translation:|note:|explanation:|"
    r"this translation|here(?:'s| is)|la traduccio (?:es|correcta)|traduccio millorada)",
    re.IGNORECASE,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _append_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        handle.flush()


def _normalised_ascii(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return decomposed.encode("ascii", "ignore").decode("ascii")


def clean_mt_reason(
    prediction: str,
    reference: str,
    target_lang: str,
    language_model: Any | None,
    confidence_threshold: float,
) -> str | None:
    """Reject P0 outputs that are not a standalone target-language translation."""
    text = str(prediction).strip()
    if not text:
        return "empty"
    if "<translation>" in text.lower() or "</translation>" in text.lower():
        return "format_markup"
    if _COMMENTARY_RE.search(_normalised_ascii(text)):
        return "commentary_marker"
    if len(text.split()) >= 5 and language_model is not None:
        language, confidence = predict_language(language_model, text)
        if language != target_lang and confidence >= confidence_threshold:
            return "wrong_language"
    if text == str(reference).strip():
        return "matches_reference"
    return None


def _group_by_model(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["negative_model_key"])].append(record)
    return grouped


def _p0_generation_config(path: str | Path) -> dict[str, Any]:
    return dict(load_generation_config(path)["baseline"])


def generate_mt_negatives(
    dataset_key: str,
    splits: tuple[str, ...] = ("train", "dev", "test"),
    model_keys: list[str] | None = None,
    datasets_config_path: str | Path = "configs/datasets.yaml",
    experiments_config_path: str | Path = "configs/experiments.yaml",
    generation_config_path: str | Path = "configs/generation.yaml",
    preprocessing_config_path: str | Path = "configs/preprocessing.yaml",
    classifier_config_path: str | Path = "configs/classifier.yaml",
    force: bool = False,
) -> dict[str, str]:
    """Generate one P0 negative per selected source pair and write balanced splits."""
    classifier_config = load_classifier_config(classifier_config_path)["classifier"]
    dataset_entry = get_dataset_entry(dataset_key, datasets_config_path)
    p0_prompt = dict(get_experiment_entry("p0_baseline", experiments_config_path)["prompt"])
    generation_config = _p0_generation_config(generation_config_path)
    configured_models = classifier_config["negative_generation"]["model_keys"]
    model_keys = model_keys or [str(key) for key in configured_models]
    if len(set(model_keys)) < 2:
        raise NegativeGenerationError("At least two negative-generation models are required.")
    max_attempts = int(classifier_config["negative_generation"]["max_attempts"])
    if max_attempts < 1:
        raise NegativeGenerationError("max_attempts must be positive.")

    preprocessing_config = load_preprocessing_config(preprocessing_config_path)
    language_config = preprocessing_config["language_id"]
    language_model = load_fasttext_model(str(language_config["model_path"]))
    confidence_threshold = float(language_config["confidence_threshold"])
    pairs_root = Path(classifier_config["source_pairs_dir"]) / dataset_key
    output_root = Path(classifier_config["output_dir"]) / dataset_key
    progress_path = output_root / "negative_generation_progress.jsonl"

    all_tasks: list[dict[str, Any]] = []
    split_pairs: dict[str, list[dict[str, Any]]] = {}
    for split in splits:
        records = _read_jsonl(pairs_root / f"{split}.jsonl")
        if not records:
            raise NegativeGenerationError(f"No classifier source pairs found for {dataset_key}/{split}.")
        split_pairs[split] = records
        for position, record in enumerate(records, start=1):
            all_tasks.append(
                {
                    **record,
                    "split": split,
                    "position": position,
                    "negative_model_key": model_keys[(position - 1) % len(model_keys)],
                    "attempt": 0,
                }
            )

    task_keys = {(str(record["split"]), int(record["position"])) for record in all_tasks}
    completed: dict[tuple[str, int], dict[str, Any]] = {}
    if progress_path.exists():
        for record in _read_jsonl(progress_path):
            key = (str(record["split"]), int(record["position"]))
            if key in task_keys:
                completed[key] = record
        logger.info("Resuming %s with %d accepted MT negatives.", dataset_key, len(completed))
    resumed_count = len(completed)

    rejection_counts: Counter[str] = Counter()
    for model_key, initial_tasks in _group_by_model(all_tasks).items():
        initial_tasks = [
            record
            for record in initial_tasks
            if (str(record["split"]), int(record["position"])) not in completed
        ]
        if not initial_tasks:
            continue
        loaded_model = load_inference_model(model_key, generation_config=generation_config)
        pending = initial_tasks
        for attempt in range(1, max_attempts + 1):
            next_pending: list[dict[str, Any]] = []
            batch_size = int(generation_config["batch_size"])
            for start in range(0, len(pending), batch_size):
                batch = pending[start : start + batch_size]
                prompts = render_translation_prompts(
                    loaded_model,
                    [str(record["source"]) for record in batch],
                    target_lang=str(dataset_entry["target_lang"]),
                    enable_thinking=False,
                    prompt_spec=p0_prompt,
                )
                outputs = generate_batch(loaded_model, prompts, generation_config)
                if len(outputs) != len(batch):
                    raise NegativeGenerationError(
                        f"{model_key} returned {len(outputs)} outputs for {len(batch)} prompts."
                    )
                accepted_batch: list[dict[str, Any]] = []
                for record, output in zip(batch, outputs):
                    reason = clean_mt_reason(
                        output,
                        str(record["target"]),
                        str(dataset_entry["target_lang"]),
                        language_model,
                        confidence_threshold,
                    )
                    if reason is None:
                        accepted = {
                            **record,
                            "raw_mt_prediction": output,
                            "mt_prediction": str(output).strip(),
                            "generation_attempt": attempt,
                            "model_id": loaded_model.model_entry.hf_id,
                        }
                        completed[(str(record["split"]), int(record["position"]))] = accepted
                        accepted_batch.append(accepted)
                    else:
                        rejection_counts[reason] += 1
                        next_pending.append(record)
                _append_jsonl(accepted_batch, progress_path)
            pending = next_pending
            if not pending:
                break
        if pending:
            logger.warning(
                "Dropping %d %s P0 outputs that remained invalid after %d attempts.",
                len(pending),
                model_key,
                max_attempts,
            )
        release_inference_model(loaded_model)

    paths: dict[str, str] = {}
    model_counts: Counter[str] = Counter()
    accepted_by_split: dict[str, int] = {}
    dropped_by_split: dict[str, int] = {}
    for split, pairs in split_pairs.items():
        references = _read_jsonl(output_root / f"{split}.reference.jsonl")
        if len(references) != len(pairs):
            raise NegativeGenerationError(f"Reference/pair count mismatch for {dataset_key}/{split}.")
        combined: list[dict[str, Any]] = []
        negatives: list[dict[str, Any]] = []
        for position, (pair, reference) in enumerate(zip(pairs, references), start=1):
            generated = completed.get((split, position))
            if generated is None:
                continue
            negative = {
                "id": f"{dataset_key}_cls_{split}_{position:06d}_mt",
                "source_id": pair["source_id"],
                "text": generated["mt_prediction"],
                "label": 0,
                "origin": "mt",
                "language": dataset_entry["target_lang"],
                "split": split,
                "mt_model_key": generated["negative_model_key"],
                "mt_model_id": generated["model_id"],
                "generation_attempt": generated["generation_attempt"],
                "raw_mt_prediction": generated["raw_mt_prediction"],
            }
            model_counts[negative["mt_model_key"]] += 1
            negatives.append(negative)
            # Pair-adjacent ordering preserves corpus order and is not a shuffle.
            combined.extend((reference, negative))
        accepted_by_split[split] = len(negatives)
        dropped_by_split[split] = len(pairs) - len(negatives)
        negative_path = output_root / f"{split}.mt.jsonl"
        combined_path = output_root / f"{split}.jsonl"
        if not force and (negative_path.exists() or combined_path.exists()):
            raise FileExistsError(f"Classifier MT data already exists: {negative_path} or {combined_path}")
        save_jsonl(negatives, negative_path)
        save_jsonl(combined, combined_path)
        paths[f"{split}_mt"] = str(negative_path)
        paths[f"{split}"] = str(combined_path)

    report_path = output_root / "negative_generation_report.json"
    save_json(
        {
            "dataset_key": dataset_key,
            "prompt": "p0_baseline",
            "input": "English source is used only to generate MT negatives; classifier examples contain target text only.",
            "model_keys": model_keys,
            "model_example_counts": dict(model_counts),
            "rejected_outputs": dict(rejection_counts),
            "resumed_accepted_outputs": resumed_count,
            "requested_pairs_by_split": {split: len(records) for split, records in split_pairs.items()},
            "accepted_pairs_by_split": accepted_by_split,
            "dropped_pairs_by_split": dropped_by_split,
            "balanced_examples_by_split": {
                split: 2 * count for split, count in accepted_by_split.items()
            },
            "progress_path": str(progress_path),
        },
        report_path,
    )
    paths["report"] = str(report_path)
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate mixed-model P0 MT negatives for classifier data.")
    parser.add_argument("--dataset", required=True, choices=("en_eu", "en_ca"))
    parser.add_argument("--splits", default="train,dev,test")
    parser.add_argument("--model-keys", default=None, help="Comma-separated override for the P0 mixture.")
    parser.add_argument("--datasets-config", default="configs/datasets.yaml")
    parser.add_argument("--experiments-config", default="configs/experiments.yaml")
    parser.add_argument("--generation-config", default="configs/generation.yaml")
    parser.add_argument("--preprocessing-config", default="configs/preprocessing.yaml")
    parser.add_argument("--classifier-config", default="configs/classifier.yaml")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    model_keys = args.model_keys.split(",") if args.model_keys else None
    paths = generate_mt_negatives(
        dataset_key=args.dataset,
        splits=tuple(part for part in args.splits.split(",") if part),
        model_keys=model_keys,
        datasets_config_path=args.datasets_config,
        experiments_config_path=args.experiments_config,
        generation_config_path=args.generation_config,
        preprocessing_config_path=args.preprocessing_config,
        classifier_config_path=args.classifier_config,
        force=args.force,
    )
    logger.info("Classifier negatives written: %s", paths)


if __name__ == "__main__":
    main()
