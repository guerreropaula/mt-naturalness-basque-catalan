"""Create ordered SFT, GRPO, and in-domain evaluation splits from processed data."""

from __future__ import annotations

import argparse
import itertools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.prompting._shared import load_processed_split
from src.utils.io import save_dataframe_jsonl, save_json

logger = logging.getLogger(__name__)


class TrainingSplitError(RuntimeError):
    """Raised when SFT/GRPO split construction is unsafe or inconsistent."""


@dataclass(frozen=True)
class TrainingSplitSizes:
    sft_train: int = 76000
    sft_dev: int = 2000
    grpo_train: int = 16000
    grpo_dev: int = 2000

    @property
    def required_train_rows(self) -> int:
        return sum(
            (
                self.sft_train,
                self.sft_dev,
                self.grpo_train,
                self.grpo_dev,
            )
        )


DEFAULT_SIZES = TrainingSplitSizes()


def _validate_columns(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    result = frame.copy()
    if "id" not in result and "sentence_id" in result:
        result["id"] = result["sentence_id"].astype(str)
    required = {"id", "source", "target"}
    missing = required - set(result.columns)
    if missing:
        raise TrainingSplitError(f"{label} is missing required columns: {sorted(missing)}")
    core_columns = ["id", "source", "target"]
    metadata_columns = [
        column
        for column in result.columns
        if column not in {*core_columns, "sentence_id"}
    ]
    return result[core_columns + metadata_columns].copy()


def build_ordered_training_splits(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    sizes: TrainingSplitSizes = DEFAULT_SIZES,
) -> dict[str, pd.DataFrame]:
    """Partition the processed training file sequentially without shuffling."""
    train = _validate_columns(train_frame, "train")
    test = _validate_columns(test_frame, "test")
    if len(train) < sizes.required_train_rows:
        raise TrainingSplitError(
            f"Expected at least {sizes.required_train_rows} processed train rows, found {len(train)}."
        )

    sft_end = sizes.sft_train + sizes.sft_dev
    grpo_train_end = sft_end + sizes.grpo_train
    grpo_dev_end = grpo_train_end + sizes.grpo_dev
    splits = {
        "sft_train": train.iloc[: sizes.sft_train].copy(),
        "sft_dev": train.iloc[sizes.sft_train : sizes.sft_train + sizes.sft_dev].copy(),
        "grpo_train": train.iloc[sft_end:grpo_train_end].copy(),
        "grpo_dev": train.iloc[grpo_train_end:grpo_dev_end].copy(),
        "test": test,
    }
    expected = {
        "sft_train": sizes.sft_train,
        "sft_dev": sizes.sft_dev,
        "grpo_train": sizes.grpo_train,
        "grpo_dev": sizes.grpo_dev,
    }
    for name, count in expected.items():
        if len(splits[name]) != count:
            raise TrainingSplitError(f"{name} contains {len(splits[name])} rows, expected {count}.")
    return splits


def deduplicate_training_splits(
    splits: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Keep strict cross-split uniqueness while preserving each split's row order.

    The in-domain test set has highest priority because it is never used for training. Only collisions with an earlier split are removed; duplicates within a
    single split do not create cross-split leakage and remain in their original
    order.
    """
    priority = (
        "test",
        "sft_train",
        "sft_dev",
        "grpo_train",
        "grpo_dev",
    )
    if set(priority) != set(splits):
        raise TrainingSplitError("Deduplication received an unexpected split set.")
    seen = {"id": set(), "source": set(), "target": set(), "source_target_pair": set()}
    retained: dict[str, pd.DataFrame] = {}
    report: dict[str, Any] = {"priority": list(priority), "splits": {}}
    for name in priority:
        frame = splits[name]
        keep_rows: list[int] = []
        dropped = {"id": 0, "source": 0, "target": 0, "source_target_pair": 0}
        for index, row in frame.iterrows():
            keys = {
                "id": str(row["id"]),
                "source": str(row["source"]),
                "target": str(row["target"]),
                "source_target_pair": (str(row["source"]), str(row["target"])),
            }
            reasons = [field for field, value in keys.items() if value in seen[field]]
            if reasons:
                for reason in reasons:
                    dropped[reason] += 1
                continue
            keep_rows.append(index)
        retained[name] = frame.loc[keep_rows].copy()
        # Add this whole split only after deciding it, so duplicate text inside
        # one split is preserved while future splits remain disjoint from it.
        retained_keys = _split_keys(retained[name])
        for field, values in retained_keys.items():
            seen[field].update(values)
        report["splits"][name] = {
            "input_rows": int(len(frame)),
            "retained_rows": int(len(retained[name])),
            "dropped_rows": int(len(frame) - len(retained[name])),
            "drop_reasons": dropped,
        }
    return retained, report


def refill_deduplicated_splits(
    splits: dict[str, pd.DataFrame],
    reserve_frame: pd.DataFrame,
    expected_sizes: dict[str, int],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Restore fixed split sizes from an ordered, disjoint preprocessing reserve."""
    reserve = _validate_columns(reserve_frame, "evaluation_reserve")
    priority = (
        "test",
        "sft_train",
        "sft_dev",
        "grpo_train",
        "grpo_dev",
    )
    if set(priority) != set(splits) or set(priority) != set(expected_sizes):
        raise TrainingSplitError("Reserve refill received unexpected split names.")

    seen = {"id": set(), "source": set(), "target": set(), "source_target_pair": set()}
    refilled: dict[str, pd.DataFrame] = {}
    used_reserve_ids: list[str] = []
    cursor = 0
    for name in priority:
        frame = splits[name].copy()
        target_size = expected_sizes[name]
        if len(frame) > target_size:
            raise TrainingSplitError(f"{name} has {len(frame)} rows, expected at most {target_size}.")
        while len(frame) < target_size:
            replacement = None
            while cursor < len(reserve):
                candidate = reserve.iloc[cursor]
                cursor += 1
                keys = {
                    "id": str(candidate["id"]),
                    "source": str(candidate["source"]),
                    "target": str(candidate["target"]),
                    "source_target_pair": (str(candidate["source"]), str(candidate["target"])),
                }
                if not any(value in seen[field] for field, value in keys.items()):
                    replacement = candidate.to_frame().T
                    used_reserve_ids.append(str(candidate["id"]))
                    break
            if replacement is None:
                raise TrainingSplitError(
                    f"Evaluation reserve exhausted while restoring {name} to {target_size} rows."
                )
            frame = pd.concat([frame, replacement], ignore_index=True)
        refilled[name] = frame
        for field, values in _split_keys(frame).items():
            seen[field].update(values)

    return refilled, {
        "input_rows": int(len(reserve)),
        "used_rows": int(len(used_reserve_ids)),
        "used_ids": used_reserve_ids,
        "unused_rows": int(len(reserve) - cursor),
    }


def _split_keys(frame: pd.DataFrame) -> dict[str, set[Any]]:
    return {
        "id": set(frame["id"].astype(str)),
        "source": set(frame["source"].astype(str)),
        "target": set(frame["target"].astype(str)),
        "source_target_pair": set(zip(frame["source"].astype(str), frame["target"].astype(str))),
    }


def validate_training_splits(splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Check all SFT, GRPO, and in-domain test combinations for leakage."""
    keys = {name: _split_keys(frame) for name, frame in splits.items()}
    pairwise: dict[str, dict[str, int]] = {}
    for left, right in itertools.combinations(splits, 2):
        overlap = {field: len(keys[left][field] & keys[right][field]) for field in keys[left]}
        pairwise[f"{left}__{right}"] = overlap
        if any(overlap.values()):
            raise TrainingSplitError(f"Overlap between {left} and {right}: {overlap}")
    return {"passed": True, "pairwise_overlap": pairwise}


def _length_summary(values: pd.Series) -> dict[str, float | int]:
    if values.empty:
        return {"min": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "max": 0}
    return {
        "min": int(values.min()),
        "mean": float(values.mean()),
        "median": float(values.quantile(0.5)),
        "p95": float(values.quantile(0.95)),
        "p99": float(values.quantile(0.99)),
        "max": int(values.max()),
    }


def split_length_statistics(frame: pd.DataFrame) -> dict[str, Any]:
    """Return transparent source/target character and whitespace-token statistics."""
    source = frame["source"].astype(str)
    target = frame["target"].astype(str)
    return {
        "rows": int(len(frame)),
        "source_tokens": _length_summary(source.str.split().str.len()),
        "target_tokens": _length_summary(target.str.split().str.len()),
        "source_characters": _length_summary(source.str.len()),
        "target_characters": _length_summary(target.str.len()),
    }


def persist_training_splits(
    dataset_key: str,
    splits: dict[str, pd.DataFrame],
    output_dir: str | Path = "data/training",
    force: bool = False,
) -> dict[str, str]:
    """Persist the train/development splits and the in-domain test split."""
    root = Path(output_dir) / dataset_key
    paths = {
        "sft_train": root / "sft" / "train.jsonl",
        "sft_dev": root / "sft" / "dev.jsonl",
        "grpo_train": root / "grpo" / "train.jsonl",
        "grpo_dev": root / "grpo" / "dev.jsonl",
        "test": root / "eval" / "test.jsonl",
    }
    if not force:
        existing = [path for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError("Training split output already exists: " + ", ".join(map(str, existing)))
    for name, path in paths.items():
        save_dataframe_jsonl(splits[name], path)
    return {name: str(path) for name, path in paths.items()}


def prepare_training_splits(
    dataset_key: str,
    processed_dir: str | Path = "data/processed",
    output_dir: str | Path = "data/training",
    force: bool = False,
) -> dict[str, str]:
    """Construct SFT/GRPO train-dev splits and one held-out in-domain test set."""
    if dataset_key not in {"en_eu", "en_ca"}:
        raise TrainingSplitError(f"Unsupported dataset for training split preparation: {dataset_key}")
    train = load_processed_split(dataset_key, "train", processed_dir)
    test = load_processed_split(dataset_key, "test", processed_dir)
    reserve = load_processed_split(dataset_key, "evaluation_reserve", processed_dir)
    nominal_splits = build_ordered_training_splits(train, test)
    splits, deduplication = deduplicate_training_splits(nominal_splits)
    expected_sizes = {name: int(len(frame)) for name, frame in nominal_splits.items()}
    splits, reserve_refill = refill_deduplicated_splits(splits, reserve, expected_sizes)
    validation = validate_training_splits(splits)
    paths = persist_training_splits(dataset_key, splits, output_dir, force)
    metadata_path = Path(output_dir) / dataset_key / "split_metadata.json"
    save_json(
        {
            "dataset_key": dataset_key,
            "source_files": {
                "train": str(Path(processed_dir) / dataset_key / "train.jsonl"),
                "test": str(Path(processed_dir) / dataset_key / "test.jsonl"),
                "evaluation_reserve": str(Path(processed_dir) / dataset_key / "evaluation_reserve.jsonl"),
            },
            "source_files_untouched": True,
            "selection_order": "processed train.jsonl original order; no shuffle before splitting",
            "deduplication_policy": (
                "Strict cross-split uniqueness by id, source, target, and source-target pair. "
                "Priority is the in-domain test set, then SFT and GRPO; later duplicate rows are dropped."
            ),
            "training_policy": {
                "sft": "cold start on sft_train; validate on sft_dev",
                "grpo": "start from SFT adapter; train on grpo_train; monitor grpo_dev",
                "in_domain_evaluation": "P0-P5 use test only; it is never used for training",
            },
            "nominal_counts": {name: int(len(frame)) for name, frame in nominal_splits.items()},
            "counts": {name: int(len(frame)) for name, frame in splits.items()},
            "deduplication": deduplication,
            "reserve_refill": reserve_refill,
            "length_statistics": {name: split_length_statistics(frame) for name, frame in splits.items()},
            "overlap_validation": validation,
            "outputs": paths,
        },
        metadata_path,
    )
    paths["metadata"] = str(metadata_path)
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build ordered SFT, GRPO, and in-domain test splits.")
    parser.add_argument("--dataset", required=True, choices=("en_eu", "en_ca", "all"))
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default="data/training")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    datasets = ("en_eu", "en_ca") if args.dataset == "all" else (args.dataset,)
    for dataset_key in datasets:
        paths = prepare_training_splits(
            dataset_key,
            processed_dir=args.processed_dir,
            output_dir=args.output_dir,
            force=args.force,
        )
        logger.info("Training splits written for %s: %s", dataset_key, paths)


if __name__ == "__main__":
    main()
