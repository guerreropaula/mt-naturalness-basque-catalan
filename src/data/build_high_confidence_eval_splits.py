"""Build immutable high-confidence EN->target evaluation splits.

The builder never changes data/processed, data/training, classifier data, or
existing experiment results. It selects new, disjoint pairs from the raw corpus,
using language/length checks and COMETKiwi QE at a configurable threshold.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.data.loaders import load_dataset
from src.data.normalization import normalize_text
from src.data.preprocessing import (
    _basic_text_reason,
    _length_reason,
    _load_fasttext_model,
    _predict_language,
    _score_alignment_batch,
)
from src.utils.config import get_dataset_entry, load_preprocessing_config
from src.utils.io import save_json, save_jsonl

logger = logging.getLogger(__name__)


class HighConfidenceEvalError(RuntimeError):
    """Raised when a clean evaluation pool cannot be constructed safely."""


def _source_id(prefix: str, original_index: int) -> str:
    return f"{prefix}_{original_index + 1:06d}"


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    except (OSError, json.JSONDecodeError) as exc:
        raise HighConfidenceEvalError(f"Could not read JSONL file {path}: {exc}") from exc


def _used_data_paths(dataset_key: str) -> list[Path]:
    roots = (
        Path("data/processed") / dataset_key,
        Path("data/training") / dataset_key,
        Path("data/classifier_source_pairs") / dataset_key,
        Path("data/classifiers") / dataset_key,
    )
    return sorted(path for root in roots if root.exists() for path in root.rglob("*.jsonl"))


def _collect_exclusions(dataset_key: str) -> tuple[dict[str, set[Any]], dict[str, int]]:
    """Collect ID/text/pair exclusions from every currently allocated split."""
    exclusions: dict[str, set[Any]] = {"id": set(), "source": set(), "target": set(), "pair": set()}
    rows_by_root: Counter[str] = Counter()
    for path in _used_data_paths(dataset_key):
        for row in _read_jsonl(path):
            row_id = row.get("source_id", row.get("sentence_id", row.get("id")))
            source = row.get("source")
            target = row.get("target", row.get("reference", row.get("target_text")))
            if row_id is not None:
                exclusions["id"].add(str(row_id))
            if source is not None:
                source_value = str(source)
                exclusions["source"].add(source_value)
                if target is not None:
                    target_value = str(target)
                    exclusions["target"].add(target_value)
                    exclusions["pair"].add((source_value, target_value))
            rows_by_root[path.parts[1] if len(path.parts) > 1 else str(path.parent)] += 1
    return exclusions, dict(rows_by_root)


def _candidate_reason(
    source: str,
    target: str,
    *,
    dataset_entry: dict[str, Any],
    preprocessing_config: dict[str, Any],
    language_model: Any,
) -> str | None:
    reason = _basic_text_reason(source, preprocessing_config["text_filtering"], "source")
    reason = reason or _basic_text_reason(target, preprocessing_config["text_filtering"], "target")
    if reason is not None:
        return reason
    language_config = preprocessing_config["language_id"]
    source_lang, source_score = _predict_language(language_model, source)
    target_lang, target_score = _predict_language(language_model, target)
    confidence = float(language_config["confidence_threshold"])
    if source_lang != dataset_entry["source_lang"] or source_score < confidence:
        return "source_language_mismatch"
    if target_lang != dataset_entry["target_lang"] or target_score < confidence:
        return "target_language_mismatch"
    return _length_reason(source, target, preprocessing_config["length_filtering"])


def _domain_allowed(row: Any, dataset_entry: dict[str, Any], preprocessing_config: dict[str, Any]) -> bool:
    domain_config = dict(preprocessing_config.get("domain_filtering", {}))
    active = bool(domain_config.get("enabled", False)) and str(dataset_entry["corpus_id"]) in {
        str(value) for value in domain_config.get("corpus_ids", [])
    }
    if not active:
        return True
    allowed = {str(value) for value in domain_config.get("allowed_values", [])}
    return str(getattr(row, str(domain_config.get("column", "domain")), "")) in allowed


def build_high_confidence_eval_splits(
    dataset_key: str,
    *,
    threshold: float = 0.70,
    dev_size: int = 2000,
    test_size: int = 2000,
    candidate_pool_size: int = 8000,
    score_chunk_size: int = 2048,
    seed: int = 42,
    output_root: str | Path = "data/evaluation_high_confidence",
    datasets_config_path: str | Path = "configs/datasets.yaml",
    preprocessing_config_path: str | Path = "configs/preprocessing.yaml",
    force: bool = False,
) -> dict[str, str]:
    """Select a scored, disjoint high-confidence evaluation pool without mutation."""
    if not 0.0 <= threshold <= 1.0:
        raise HighConfidenceEvalError("threshold must be in [0, 1].")
    total_size = dev_size + test_size
    if candidate_pool_size < total_size:
        raise HighConfidenceEvalError("candidate_pool_size must be at least dev_size + test_size.")
    if score_chunk_size < 1:
        raise HighConfidenceEvalError("score_chunk_size must be positive.")

    dataset_entry = get_dataset_entry(dataset_key, datasets_config_path)
    preprocessing_config = load_preprocessing_config(preprocessing_config_path)
    exclusions, exclusions_by_root = _collect_exclusions(dataset_key)
    raw_frame = load_dataset(dataset_key, datasets_config_path)
    language_model = _load_fasttext_model(str(preprocessing_config["language_id"]["model_path"]))
    normalization = preprocessing_config["normalization"]
    prefix = str(dataset_entry.get("id_prefix", dataset_entry["corpus_id"]))
    qe_config = {
        "model": "Unbabel/wmt23-cometkiwi-da-xl",
        "batch_size": 16,
        "gpus": 1,
        "progress_bar": False,
    }

    accepted: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()

    def flush_pending() -> None:
        if not pending:
            return
        scores = _score_alignment_batch(
            [record["source"] for record in pending],
            [record["target"] for record in pending],
            qe_config,
        )
        for record, score in zip(pending, scores, strict=True):
            stats["scored"] += 1
            if float(score) < threshold:
                stats["cometkiwi_below_threshold"] += 1
                continue
            record["cometkiwi_score"] = float(score)
            accepted.append(record)
            stats["accepted"] += 1
        pending.clear()

    for row in raw_frame.itertuples(index=False):
        stats["rows_seen"] += 1
        if not _domain_allowed(row, dataset_entry, preprocessing_config):
            stats["domain_not_allowed"] += 1
            continue
        original_index = int(getattr(row, "original_index"))
        source_value = normalize_text(getattr(row, "source"), normalization)
        target_value = normalize_text(getattr(row, "target"), normalization)
        if pd.isna(source_value) or pd.isna(target_value):
            stats["empty_after_normalization"] += 1
            continue
        source, target = str(source_value), str(target_value)
        row_id = _source_id(prefix, original_index)
        pair = (source, target)
        if (
            row_id in exclusions["id"]
            or source in exclusions["source"]
            or target in exclusions["target"]
            or pair in exclusions["pair"]
        ):
            stats["already_used"] += 1
            continue
        reason = _candidate_reason(
            source,
            target,
            dataset_entry=dataset_entry,
            preprocessing_config=preprocessing_config,
            language_model=language_model,
        )
        if reason is not None:
            stats[reason] += 1
            continue
        record: dict[str, Any] = {
            "id": row_id,
            "sentence_id": row_id,
            "source": source,
            "target": target,
            "original_index": original_index,
        }
        for name in ("domain", "type", "corpus_alignment"):
            if hasattr(row, name):
                record[name] = getattr(row, name)
        pending.append(record)
        # COMETKiwi batches internally for GPU memory; score larger chunks so
        # Lightning is not initialized once per 16 candidate pairs.
        if len(pending) >= score_chunk_size:
            flush_pending()
            if len(accepted) >= candidate_pool_size:
                break
    flush_pending()

    if len(accepted) < total_size:
        raise HighConfidenceEvalError(
            f"Only {len(accepted)} unused pairs meet COMETKiwi >= {threshold}; need {total_size}."
        )

    rng = random.Random(seed)
    rng.shuffle(accepted)
    selected = accepted[:total_size]
    selected_ids = {record["id"] for record in selected}
    selected_sources = {record["source"] for record in selected}
    selected_targets = {record["target"] for record in selected}
    selected_pairs = {(record["source"], record["target"]) for record in selected}
    overlaps = {
        key: len(values & exclusions[key])
        for key, values in {
            "id": selected_ids,
            "source": selected_sources,
            "target": selected_targets,
            "pair": selected_pairs,
        }.items()
    }
    if any(overlaps.values()):
        raise HighConfidenceEvalError(f"New evaluation split overlaps existing data: {overlaps}")

    output_dir = Path(output_root) / dataset_key
    paths = {
        "global_dev": output_dir / "global_dev.jsonl",
        "global_test": output_dir / "global_test.jsonl",
        "metadata": output_dir / "selection_metadata.json",
    }
    if not force and any(path.exists() for path in paths.values()):
        raise FileExistsError("High-confidence evaluation output already exists: " + ", ".join(map(str, paths.values())))
    save_jsonl(selected[:dev_size], paths["global_dev"])
    save_jsonl(selected[dev_size:], paths["global_test"])
    scores = [float(record["cometkiwi_score"]) for record in selected]
    save_json(
        {
            "dataset_key": dataset_key,
            "selection_version": "cometkiwi_ge_0.70_v1",
            "source_corpus": dataset_entry["name"],
            "threshold": threshold,
            "model": qe_config["model"],
            "seed": seed,
            "candidate_pool_size": candidate_pool_size,
            "sizes": {"global_dev": dev_size, "global_test": test_size},
            "score_summary": {"min": min(scores), "mean": sum(scores) / len(scores), "max": max(scores)},
            "selection_stats": dict(stats),
            "excluded_existing_rows_by_root": exclusions_by_root,
            "overlap_validation": {"passed": True, **overlaps},
            "output_files": {key: str(value) for key, value in paths.items()},
        },
        paths["metadata"],
    )
    return {key: str(value) for key, value in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("en_ca", "en_eu"))
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--dev-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--candidate-pool-size", type=int, default=8000)
    parser.add_argument("--score-chunk-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default="data/evaluation_high_confidence")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    paths = build_high_confidence_eval_splits(
        args.dataset,
        threshold=args.threshold,
        dev_size=args.dev_size,
        test_size=args.test_size,
        candidate_pool_size=args.candidate_pool_size,
        score_chunk_size=args.score_chunk_size,
        seed=args.seed,
        output_root=args.output_root,
        force=args.force,
    )
    logger.info("High-confidence evaluation splits written: %s", paths)


if __name__ == "__main__":
    main()
