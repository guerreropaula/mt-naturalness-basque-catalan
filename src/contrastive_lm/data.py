"""Data loading and invariants for contrastive HT-versus-MT language models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ContrastiveLMDataError(RuntimeError):
    """Raised when the classifier-style HT/MT source data is unsafe to use."""


_SPLITS = ("train", "dev", "test")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise ContrastiveLMDataError(f"Contrastive-LM data file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ContrastiveLMDataError(f"Contrastive-LM data file is empty: {path}")
    return records


def split_path(data_root: str | Path, dataset_key: str, split: str) -> Path:
    if split not in _SPLITS:
        raise ContrastiveLMDataError(f"Unsupported split {split!r}; expected one of {_SPLITS}.")
    return Path(data_root) / dataset_key / f"{split}.jsonl"


def load_labeled_texts(
    data_root: str | Path,
    dataset_key: str,
    split: str,
    label: int,
) -> list[str]:
    """Load one class of target-side texts from classifier-compatible JSONL."""
    records = read_jsonl(split_path(data_root, dataset_key, split))
    required = {"text", "label"}
    missing = required - set(records[0])
    if missing:
        raise ContrastiveLMDataError(
            f"{dataset_key}/{split} lacks required fields: {sorted(missing)}"
        )
    texts = [
        str(record["text"]).strip()
        for record in records
        if int(record["label"]) == label and str(record["text"]).strip()
    ]
    if not texts:
        raise ContrastiveLMDataError(
            f"No non-empty label={label} texts in {dataset_key}/{split}."
        )
    return texts


def validate_classifier_style_splits(data_root: str | Path, dataset_key: str) -> dict[str, int]:
    """Verify balanced labels and source-group disjointness before LM training."""
    source_groups: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    for split in _SPLITS:
        records = read_jsonl(split_path(data_root, dataset_key, split))
        labels = {int(record["label"]) for record in records if "label" in record}
        if labels != {0, 1}:
            raise ContrastiveLMDataError(
                f"{dataset_key}/{split} must contain exactly HT label 1 and MT label 0; got {labels}."
            )
        if any("text" not in record for record in records):
            raise ContrastiveLMDataError(f"{dataset_key}/{split} contains a record without text.")
        source_groups[split] = {
            str(record["source_id"])
            for record in records
            if record.get("source_id") is not None
        }
        counts[split] = len(records)
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap = source_groups[left] & source_groups[right]
        if overlap:
            raise ContrastiveLMDataError(
                f"Contrastive-LM source overlap {left}/{right}: {len(overlap)} groups."
            )
    return counts
