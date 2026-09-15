"""Build ordered, disjoint target-side classifier data from unused corpus rows."""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.data.loaders import load_dataset
from src.data.normalization import normalize_text
from src.data.preprocessing import (
    basic_text_rejection_reason,
    length_rejection_reason,
    load_fasttext_model,
    predict_language,
)
from src.prompting._shared import load_processed_split
from src.utils.config import get_dataset_entry, load_classifier_config, load_preprocessing_config
from src.utils.io import save_json, save_jsonl

logger = logging.getLogger(__name__)


class ClassifierQualityError(RuntimeError):
    """Raised when COMETKiwi quality scoring for classifier positives fails."""


_COMET_MODEL_CACHE: dict[str, Any] = {}


def _default_quality_gpus() -> int:
    try:
        import torch
    except Exception:
        return 0
    return 1 if torch.cuda.is_available() else 0


def _load_quality_model(model_name: str) -> Any:
    if model_name in _COMET_MODEL_CACHE:
        return _COMET_MODEL_CACHE[model_name]
    try:
        from comet import download_model, load_from_checkpoint
    except Exception as exc:
        raise ClassifierQualityError(
            "COMET is required for Catalan classifier-positive filtering."
        ) from exc
    model = load_from_checkpoint(download_model(model_name))
    _COMET_MODEL_CACHE[model_name] = model
    return model


def _score_quality_batch(
    sources: list[str], targets: list[str], quality_config: dict[str, Any]
) -> list[float]:
    records = [
        {"src": source, "mt": target}
        for source, target in zip(sources, targets, strict=True)
    ]
    outputs = _load_quality_model(str(quality_config["model"])).predict(
        records,
        batch_size=int(quality_config.get("batch_size", 8)),
        gpus=int(quality_config.get("gpus", _default_quality_gpus())),
        progress_bar=bool(quality_config.get("progress_bar", False)),
    )
    if hasattr(outputs, "scores"):
        scores = [float(score) for score in outputs.scores]
    elif isinstance(outputs, dict) and "scores" in outputs:
        scores = [float(score) for score in outputs["scores"]]
    else:
        raise ClassifierQualityError("COMETKiwi output has no per-example scores.")
    if len(scores) != len(records):
        raise ClassifierQualityError(
            f"COMETKiwi returned {len(scores)} scores for {len(records)} records."
        )
    return scores


class ClassifierDataError(RuntimeError):
    """Raised when classifier data cannot be built safely."""


@dataclass(frozen=True)
class PairSplitSizes:
    train: int
    dev: int
    test: int

    @property
    def total(self) -> int:
        return self.train + self.dev + self.test


def _source_id(prefix: str, original_index: int) -> str:
    return f"{prefix}_{original_index + 1:06d}"


def _pair_key(source: str, target: str) -> tuple[str, str]:
    return source, target


def classifier_sizes(config: dict[str, Any], size_mode: str) -> PairSplitSizes:
    try:
        values = config["classifier"]["size_modes"][size_mode]
        return PairSplitSizes(
            train=int(values["train_pairs"]),
            dev=int(values["dev_pairs"]),
            test=int(values["test_pairs"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ClassifierDataError(f"Unknown or invalid classifier size mode: {size_mode}") from exc


def _existing_records(dataset_key: str, processed_dir: str | Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    processed_root = Path(processed_dir) / dataset_key
    splits = ["train", "test"]
    if (processed_root / "lfp_background.jsonl").exists() or (
        processed_root / "lfp_background.parquet"
    ).exists():
        splits.append("lfp_background")

    for split in splits:
        frame = load_processed_split(dataset_key, split, processed_dir)
        required = {"sentence_id", "source", "target"}
        missing = required - set(frame.columns)
        if missing:
            raise ClassifierDataError(
                f"Existing {dataset_key}/{split} split is missing columns: {sorted(missing)}"
            )
        records.extend(
            {
                "source_id": str(row.sentence_id),
                "source": str(row.source),
                "target": str(row.target),
                "split": split,
            }
            for row in frame.itertuples(index=False)
        )
    return records


def _exclusion_sets(records: Iterable[dict[str, str]]) -> dict[str, set[Any]]:
    records = list(records)
    return {
        "source_id": {record["source_id"] for record in records},
        "source": {record["source"] for record in records},
        "target": {record["target"] for record in records},
        "pair": {_pair_key(record["source"], record["target"]) for record in records},
    }


def _candidate_reason(
    source: str,
    target: str,
    dataset_entry: dict[str, Any],
    preprocessing_config: dict[str, Any],
    language_model: Any,
) -> str | None:
    text_filtering = preprocessing_config["text_filtering"]
    reason = basic_text_rejection_reason(source, text_filtering, "source")
    reason = reason or basic_text_rejection_reason(target, text_filtering, "target")
    if reason is not None:
        return reason
    language_config = preprocessing_config["language_id"]
    threshold = float(language_config["confidence_threshold"])
    source_lang, source_score = predict_language(language_model, source)
    target_lang, target_score = predict_language(language_model, target)
    if source_lang != dataset_entry["source_lang"] or source_score < threshold:
        return "source_language_mismatch"
    if target_lang != dataset_entry["target_lang"] or target_score < threshold:
        return "target_language_mismatch"
    return length_rejection_reason(source, target, preprocessing_config["length_filtering"])


def _overlap_reason(
    source_id: str,
    source: str,
    target: str,
    exclusions: dict[str, set[Any]],
) -> str | None:
    """Return one deterministic first overlap reason, if any."""
    if source_id in exclusions["source_id"]:
        return "source_id_already_allocated"
    if source in exclusions["source"]:
        return "normalized_source_already_allocated"
    if target in exclusions["target"]:
        return "normalized_target_already_allocated"
    if _pair_key(source, target) in exclusions["pair"]:
        return "normalized_pair_already_allocated"
    return None


def _initialise_selection_audit(audit: dict[str, Any] | None) -> Counter[str] | None:
    """Initialise optional first-failure counters for a selection pass."""
    if audit is None:
        return None
    audit.clear()
    audit["rows_scanned"] = 0
    return Counter()


def _finalise_selection_audit(
    audit: dict[str, Any] | None,
    reasons: Counter[str] | None,
    selected: list[dict[str, Any]],
) -> None:
    if audit is None or reasons is None:
        return
    audit["selected"] = len(selected)
    audit["first_failure_reasons"] = dict(sorted(reasons.items()))


def select_disjoint_classifier_pairs(
    raw_frame: pd.DataFrame,
    dataset_entry: dict[str, Any],
    preprocessing_config: dict[str, Any],
    existing_records: Iterable[dict[str, str]],
    sizes: PairSplitSizes,
    language_model: Any,
    audit: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Select the first eligible unused pairs in original corpus order."""
    required = {"original_index", "source", "target"}
    missing = required - set(raw_frame.columns)
    if missing:
        raise ClassifierDataError(f"Raw corpus is missing columns: {sorted(missing)}")

    exclusions = _exclusion_sets(existing_records)
    selected: list[dict[str, Any]] = []
    reasons = _initialise_selection_audit(audit)
    prefix = str(dataset_entry.get("id_prefix", dataset_entry["corpus_id"]))
    normalization = preprocessing_config["normalization"]
    for row in raw_frame.itertuples(index=False):
        if audit is not None:
            audit["rows_scanned"] += 1
        original_index = int(getattr(row, "original_index"))
        source_value = normalize_text(getattr(row, "source"), normalization)
        target_value = normalize_text(getattr(row, "target"), normalization)
        if pd.isna(source_value) or pd.isna(target_value):
            if reasons is not None:
                reasons["missing_source_or_target"] += 1
            continue
        source = str(source_value)
        target = str(target_value)
        source_id = _source_id(prefix, original_index)
        overlap_reason = _overlap_reason(source_id, source, target, exclusions)
        if overlap_reason is not None:
            if reasons is not None:
                reasons[overlap_reason] += 1
            continue
        candidate_reason = _candidate_reason(
            source, target, dataset_entry, preprocessing_config, language_model
        )
        if candidate_reason is not None:
            if reasons is not None:
                reasons[candidate_reason] += 1
            continue
        record = {
            "source_id": source_id,
            "source": source,
            "target": target,
            "original_index": original_index,
        }
        selected.append(record)
        exclusions["source_id"].add(source_id)
        exclusions["source"].add(source)
        exclusions["target"].add(target)
        exclusions["pair"].add(_pair_key(source, target))
        if len(selected) == sizes.total:
            break

    if len(selected) != sizes.total:
        raise ClassifierDataError(
            f"Only selected {len(selected)} unused valid rows; need {sizes.total} classifier pairs."
        )
    _finalise_selection_audit(audit, reasons, selected)
    boundaries = (sizes.train, sizes.train + sizes.dev)
    return {
        "train": selected[: boundaries[0]],
        "dev": selected[boundaries[0] : boundaries[1]],
        "test": selected[boundaries[1] :],
    }


def select_quality_filtered_classifier_pairs(
    raw_frame: pd.DataFrame,
    dataset_entry: dict[str, Any],
    preprocessing_config: dict[str, Any],
    existing_records: Iterable[dict[str, str]],
    sizes: PairSplitSizes,
    language_model: Any,
    quality_config: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Fill classifier splits from unused domain-matched rows above a QE threshold."""
    required = {"original_index", "source", "target", "domain"}
    missing = required - set(raw_frame.columns)
    if missing:
        raise ClassifierDataError(f"Raw corpus is missing columns: {sorted(missing)}")

    domain_config = dict(preprocessing_config.get("domain_filtering", {}))
    allowed_domains = {str(value) for value in domain_config.get("allowed_values", [])}
    if not allowed_domains:
        raise ClassifierDataError("Catalan classifier selection requires allowed domains.")
    threshold = float(quality_config["threshold"])
    candidate_batch_size = int(quality_config.get("candidate_batch_size", 4096))
    if candidate_batch_size < 1:
        raise ClassifierDataError("positive_quality_filter.candidate_batch_size must be positive.")

    exclusions = _exclusion_sets(existing_records)
    considered = {name: set(values) for name, values in exclusions.items()}
    selected: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    prefix = str(dataset_entry.get("id_prefix", dataset_entry["corpus_id"]))
    normalization = preprocessing_config["normalization"]
    reasons = _initialise_selection_audit(audit)
    stats = {
        "eligible_rows_scored": 0,
        "below_threshold": 0,
        "accepted": 0,
    }

    def flush_pending() -> None:
        if not pending or len(selected) >= sizes.total:
            return
        scores = _score_quality_batch(
            [str(record["source"]) for record in pending],
            [str(record["target"]) for record in pending],
            quality_config,
        )
        stats["eligible_rows_scored"] += len(scores)
        for record, score in zip(pending, scores, strict=True):
            if float(score) < threshold:
                stats["below_threshold"] += 1
                if reasons is not None:
                    reasons["cometkiwi_below_threshold"] += 1
                continue
            if len(selected) >= sizes.total:
                if reasons is not None:
                    reasons["qualified_after_selection_quota"] += 1
                continue
            record["positive_cometkiwi_score"] = float(score)
            selected.append(record)
        pending.clear()

    for row in raw_frame.itertuples(index=False):
        audit["rows_scanned"] += 1
        if str(getattr(row, "domain")) not in allowed_domains:
            if reasons is not None:
                reasons["domain_outside_hrm_cul"] += 1
            continue
        original_index = int(getattr(row, "original_index"))
        source_value = normalize_text(getattr(row, "source"), normalization)
        target_value = normalize_text(getattr(row, "target"), normalization)
        if pd.isna(source_value) or pd.isna(target_value):
            if reasons is not None:
                reasons["missing_source_or_target"] += 1
            continue
        source = str(source_value)
        target = str(target_value)
        source_id = _source_id(prefix, original_index)
        pair = _pair_key(source, target)
        overlap_reason = _overlap_reason(source_id, source, target, considered)
        if overlap_reason is not None:
            if reasons is not None:
                reasons[overlap_reason] += 1
            continue
        candidate_reason = _candidate_reason(
            source, target, dataset_entry, preprocessing_config, language_model
        )
        if candidate_reason is not None:
            if reasons is not None:
                reasons[candidate_reason] += 1
            continue
        considered["source_id"].add(source_id)
        considered["source"].add(source)
        considered["target"].add(target)
        considered["pair"].add(pair)
        pending.append(
            {
                "source_id": source_id,
                "source": source,
                "target": target,
                "original_index": original_index,
                "domain": str(getattr(row, "domain")),
                "type": str(getattr(row, "type", "")),
                "corpus_alignment": float(getattr(row, "corpus_alignment", 0.0)),
            }
        )
        if len(pending) >= candidate_batch_size:
            flush_pending()
            if len(selected) >= sizes.total:
                break
    flush_pending()

    if len(selected) != sizes.total:
        raise ClassifierDataError(
            f"Only selected {len(selected)} unused HRM/CUL rows with COMETKiwi >= "
            f"{threshold}; need {sizes.total} classifier pairs."
        )
    accepted_scores = [float(record["positive_cometkiwi_score"]) for record in selected]
    stats.update(
        {
            "accepted": len(selected),
            "model": str(quality_config["model"]),
            "threshold": threshold,
            "allowed_domains": sorted(allowed_domains),
            "accepted_score_min": min(accepted_scores),
            "accepted_score_mean": sum(accepted_scores) / len(accepted_scores),
            "accepted_score_max": max(accepted_scores),
        }
    )
    audit.update(stats)
    _finalise_selection_audit(audit, reasons, selected)
    boundaries = (sizes.train, sizes.train + sizes.dev)
    return {
        "train": selected[: boundaries[0]],
        "dev": selected[boundaries[0] : boundaries[1]],
        "test": selected[boundaries[1] :],
    }


def validate_no_overlap(
    classifier_splits: dict[str, list[dict[str, Any]]],
    existing_records: Iterable[dict[str, str]],
) -> dict[str, Any]:
    """Prove disjointness against MT/SFT splits and among classifier splits."""
    existing = _exclusion_sets(existing_records)
    report: dict[str, Any] = {"existing_mt_sft": {}, "between_classifier_splits": {}}
    split_sets: dict[str, dict[str, set[Any]]] = {}
    for split, records in classifier_splits.items():
        keys = {
            "source_id": {str(record["source_id"]) for record in records},
            "source": {str(record["source"]) for record in records},
            "target": {str(record["target"]) for record in records},
            "pair": {
                _pair_key(str(record["source"]), str(record["target"])) for record in records
            },
        }
        split_sets[split] = keys
        overlaps = {name: len(values & existing[name]) for name, values in keys.items()}
        report["existing_mt_sft"][split] = overlaps
        if any(overlaps.values()):
            raise ClassifierDataError(f"Classifier {split} overlaps existing MT/SFT data: {overlaps}")
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlaps = {
            name: len(split_sets[left][name] & split_sets[right][name]) for name in split_sets[left]
        }
        report["between_classifier_splits"][f"{left}_{right}"] = overlaps
        if any(overlaps.values()):
            raise ClassifierDataError(f"Classifier split overlap {left}/{right}: {overlaps}")
    report["passed"] = True
    return report


def _reference_examples(dataset_key: str, split: str, records: list[dict[str, Any]], language: str) -> list[dict[str, Any]]:
    return [
        {
            "id": f"{dataset_key}_cls_{split}_{index:06d}_ref",
            "source_id": record["source_id"],
            "text": record["target"],
            "label": 1,
            "origin": "reference",
            "language": language,
            "split": split,
            **(
                {"positive_cometkiwi_score": float(record["positive_cometkiwi_score"])}
                if "positive_cometkiwi_score" in record
                else {}
            ),
        }
        for index, record in enumerate(records, start=1)
    ]


def prepare_classifier_source_data(
    dataset_key: str,
    size_mode: str = "matched",
    datasets_config_path: str | Path = "configs/datasets.yaml",
    preprocessing_config_path: str | Path = "configs/preprocessing.yaml",
    classifier_config_path: str | Path = "configs/classifier.yaml",
    processed_dir: str | Path = "data/processed",
    force: bool = False,
) -> dict[str, str]:
    """Create ordered unused source/reference pairs and reference positives."""
    classifier_config = load_classifier_config(classifier_config_path)
    preprocessing_config = load_preprocessing_config(preprocessing_config_path)
    dataset_entry = get_dataset_entry(dataset_key, datasets_config_path)
    sizes = classifier_sizes(classifier_config, size_mode)
    existing = _existing_records(dataset_key, processed_dir)
    raw_frame = load_dataset(dataset_key, datasets_config_path)
    language_model = load_fasttext_model(str(preprocessing_config["language_id"]["model_path"]))
    quality_config = dict(
        classifier_config["classifier"].get("positive_quality_filter", {}).get(dataset_key, {})
    )
    selection_audit: dict[str, Any] = {}
    quality_enabled = bool(quality_config.get("enabled", False))
    if quality_enabled:
        splits = select_quality_filtered_classifier_pairs(
            raw_frame,
            dataset_entry,
            preprocessing_config,
            existing,
            sizes,
            language_model,
            quality_config,
            selection_audit,
        )
    else:
        splits = select_disjoint_classifier_pairs(
            raw_frame,
            dataset_entry,
            preprocessing_config,
            existing,
            sizes,
            language_model,
            selection_audit,
        )
    overlap_report = validate_no_overlap(splits, existing)

    root = Path(classifier_config["classifier"]["output_dir"]) / dataset_key
    pairs_root = Path(classifier_config["classifier"]["source_pairs_dir"]) / dataset_key
    paths: dict[str, str] = {}
    for split, records in splits.items():
        pair_path = pairs_root / f"{split}.jsonl"
        reference_path = root / f"{split}.reference.jsonl"
        if not force and (pair_path.exists() or reference_path.exists()):
            raise FileExistsError(f"Classifier split already exists: {pair_path} or {reference_path}")
        for record in records:
            record["split"] = split
        save_jsonl(records, pair_path)
        save_jsonl(_reference_examples(dataset_key, split, records, dataset_entry["target_lang"]), reference_path)
        paths[f"{split}_pairs"] = str(pair_path)
        paths[f"{split}_reference"] = str(reference_path)
    overlap_path = pairs_root / "overlap_audit.json"
    save_json(
        {
            "dataset_key": dataset_key,
            "size_mode": size_mode,
            "sizes": {"train": sizes.train, "dev": sizes.dev, "test": sizes.test},
            "selection_order": "original corpus order; no shuffle before split selection",
            "positive_quality_filter": {
                "enabled": quality_enabled,
                **(
                    {
                        key: value
                        for key, value in selection_audit.items()
                        if key
                        in {
                            "eligible_rows_scored",
                            "below_threshold",
                            "accepted",
                            "model",
                            "threshold",
                            "allowed_domains",
                            "accepted_score_min",
                            "accepted_score_mean",
                            "accepted_score_max",
                        }
                    }
                    if quality_enabled
                    else {}
                ),
            },
            "selection_audit": selection_audit,
            "overlap_checks": overlap_report,
        },
        overlap_path,
    )
    paths["overlap_audit"] = str(overlap_path)
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build disjoint ordered classifier source/reference splits.")
    parser.add_argument("--dataset", required=True, choices=("en_eu", "en_ca"))
    parser.add_argument("--size-mode", choices=("matched",), default="matched")
    parser.add_argument("--datasets-config", default="configs/datasets.yaml")
    parser.add_argument("--preprocessing-config", default="configs/preprocessing.yaml")
    parser.add_argument("--classifier-config", default="configs/classifier.yaml")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    common = {
        "dataset_key": args.dataset,
        "size_mode": args.size_mode,
        "datasets_config_path": args.datasets_config,
        "preprocessing_config_path": args.preprocessing_config,
        "classifier_config_path": args.classifier_config,
        "processed_dir": args.processed_dir,
    }
    paths = prepare_classifier_source_data(**common, force=args.force)
    logger.info("Classifier source data written: %s", paths)


if __name__ == "__main__":
    main()
