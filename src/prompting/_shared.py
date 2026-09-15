"""Shared helpers for configurable experiment runners."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.loaders import DatasetLoadError


class ExperimentError(RuntimeError):
    """Raised when an experiment configuration or runtime step is invalid."""


def load_processed_split(
    dataset_key: str,
    split: str,
    processed_dir: str | Path = "data/processed",
) -> pd.DataFrame:
    """Load a processed split saved as Parquet or JSONL."""
    base_dir = Path(processed_dir) / dataset_key
    parquet_path = base_dir / f"{split}.parquet"
    jsonl_path = base_dir / f"{split}.jsonl"
    if jsonl_path.exists():
        df = pd.read_json(jsonl_path, lines=True)
    elif parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    else:
        raise DatasetLoadError(
            f"Could not find processed split '{split}' for dataset '{dataset_key}' under {base_dir}"
        )
    if "sentence_id" not in df.columns and "id" in df.columns:
        df["sentence_id"] = df["id"].astype(str)
    return df


def _load_jsonl_or_parquet(jsonl_path: Path, parquet_path: Path, label: str) -> pd.DataFrame:
    if jsonl_path.exists():
        df = pd.read_json(jsonl_path, lines=True)
    elif parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    else:
        raise DatasetLoadError(f"Could not find split file for {label}")
    if "sentence_id" not in df.columns and "id" in df.columns:
        df["sentence_id"] = df["id"].astype(str)
    return df


def load_experiment_split(
    dataset_key: str,
    split: str,
    processed_dir: str | Path = "data/processed",
    training_dir: str | Path = "data/training",
) -> pd.DataFrame:
    """Load the canonical split for experiments.

    Development and final evaluation should use the deduplicated global eval
    files created by the training-split builder. If those files are absent
    (for tests, smoke data, or freshly preprocessed corpora), fall back to the
    processed split. Training split construction must keep using
    ``load_processed_split`` directly.
    """
    global_split = {"dev": "global_dev", "test": "global_test"}.get(split)
    if global_split is not None:
        eval_dir = Path(training_dir) / dataset_key / "eval"
        eval_jsonl = eval_dir / f"{global_split}.jsonl"
        eval_parquet = eval_dir / f"{global_split}.parquet"
        if eval_jsonl.exists() or eval_parquet.exists():
            return _load_jsonl_or_parquet(
                eval_jsonl,
                eval_parquet,
                f"global evaluation split '{global_split}' for dataset '{dataset_key}'",
            )
    return load_processed_split(dataset_key, split, processed_dir)


def resolve_generation_profile(
    generation_config: dict[str, Any],
    profile_path: str,
) -> dict[str, Any]:
    """Resolve a dotted generation-profile path merged with baseline defaults."""
    value: Any = generation_config
    for part in profile_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ExperimentError(
                f"Unknown generation profile '{profile_path}' in generation config"
            )
        value = value[part]
    if not isinstance(value, dict):
        raise ExperimentError(f"Generation profile '{profile_path}' must resolve to a mapping")

    resolved = copy.deepcopy(value)
    if profile_path == "baseline":
        return resolved

    baseline = generation_config.get("baseline")
    if not isinstance(baseline, dict):
        return resolved

    merged = copy.deepcopy(baseline)
    merged.update(resolved)
    return merged


def experiment_output_dir(
    results_dir: str | Path,
    experiment_entry: dict[str, Any],
    experiment_key: str,
    dataset_key: str,
    model_key: str,
) -> Path:
    """Build a traceable output directory for an experiment run."""
    label = experiment_key.split("_", 1)[0]
    if not (label.startswith("p") and label[1:].isdigit()):
        raise ExperimentError(f"Experiment key must start with a phase label: {experiment_key}")
    return Path(results_dir) / label / dataset_key / model_key


def load_existing_records(path: Path) -> list[dict[str, Any]]:
    """Load newline-delimited records if they already exist."""
    if not path.exists():
        return []
    return pd.read_json(path, lines=True).to_dict(orient="records")
