"""Loaders for the training and evaluation corpora."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset as hf_load_dataset

from src.utils.config import ConfigError, get_dataset_entry
from src.utils.hf_auth import get_hf_token

logger = logging.getLogger(__name__)


class DatasetLoadError(RuntimeError):
    """Raised when a corpus cannot be loaded as aligned bilingual rows."""


def _with_metadata(df: pd.DataFrame, dataset_config: dict[str, Any]) -> pd.DataFrame:
    df.attrs["dataset_metadata"] = {
        "dataset_name": dataset_config["name"],
        "corpus_id": dataset_config["corpus_id"],
        "source_lang": dataset_config["source_lang"],
        "target_lang": dataset_config["target_lang"],
        "input_rows": len(df),
    }
    return df


def load_ehu_hac(dataset_config: dict[str, Any], limit: int | None = None) -> pd.DataFrame:
    """Load aligned EHU-HAC files in their original order."""
    paths = dataset_config["paths"]
    root = Path(paths["root_dir"])
    source_path = root / paths["source_file"]
    target_path = root / paths["target_file"]
    if not source_path.exists() or not target_path.exists():
        raise DatasetLoadError(f"EHU-HAC files not found: {source_path}, {target_path}")

    with source_path.open(encoding="utf-8") as source_file:
        source_lines = [line.rstrip("\n") for line in source_file]
    with target_path.open(encoding="utf-8") as target_file:
        target_lines = [line.rstrip("\n") for line in target_file]
    if len(source_lines) != len(target_lines):
        raise DatasetLoadError(
            f"EHU-HAC source/target line-count mismatch: {len(source_lines)} vs {len(target_lines)}"
        )
    if limit is not None:
        source_lines = source_lines[:limit]
        target_lines = target_lines[:limit]

    return _with_metadata(
        pd.DataFrame(
            {
                "original_index": range(len(source_lines)),
                "source": source_lines,
                "target": target_lines,
                "corpus": dataset_config["corpus_id"],
            }
        ),
        dataset_config,
    )


def load_local_jsonl(dataset_config: dict[str, Any], limit: int | None = None) -> pd.DataFrame:
    """Load a normalized local JSONL parallel corpus without reordering rows."""
    path = Path(dataset_config["paths"]["file"])
    if not path.exists():
        raise DatasetLoadError(f"Local JSONL dataset not found: {path}")
    frame = pd.read_json(path, lines=True)
    required = {"source", "target"}
    missing = required - set(frame.columns)
    if missing:
        raise DatasetLoadError(f"Local JSONL dataset {path} is missing columns: {sorted(missing)}")
    if limit is not None:
        frame = frame.head(int(limit)).copy()
    frame = frame.copy()
    frame.insert(0, "original_index", range(len(frame)))
    frame["corpus"] = dataset_config["corpus_id"]
    return _with_metadata(frame, dataset_config)


def load_hf_dataset(dataset_config: dict[str, Any], limit: int | None = None) -> pd.DataFrame:
    """Load configured bilingual and metadata columns in source order."""
    mapping = dataset_config["column_mapping"]
    source_column = mapping["source"]
    target_column = mapping["target"]
    metadata_mapping = dict(dataset_config.get("metadata_mapping", {}))
    try:
        dataset = hf_load_dataset(
            dataset_config["hf_repo_id"],
            name=dataset_config.get("subset"),
            split=dataset_config.get("split", "train"),
            token=get_hf_token(),
        )
    except Exception as exc:  # pragma: no cover - remote backend errors vary
        raise DatasetLoadError(f"Could not load {dataset_config['hf_repo_id']}: {exc}") from exc

    selected_columns = [source_column, target_column, *metadata_mapping.values()]
    missing = set(selected_columns) - set(dataset.column_names)
    if missing:
        raise DatasetLoadError(f"Dataset is missing required columns: {sorted(missing)}")
    dataset = dataset.select_columns(selected_columns)
    if limit is not None:
        dataset = dataset.select(range(min(int(limit), len(dataset))))
    records = dataset.to_pandas()
    records = records.rename(
        columns={
            source_column: "source",
            target_column: "target",
            **{source_name: output_name for output_name, source_name in metadata_mapping.items()},
        }
    )
    records.insert(0, "original_index", range(len(records)))
    records["corpus"] = dataset_config["corpus_id"]
    return _with_metadata(records, dataset_config)


_LOADERS = {"ehu_hac": load_ehu_hac, "hf_dataset": load_hf_dataset, "local_jsonl": load_local_jsonl}


def load_dataset(
    dataset_key: str,
    datasets_config_path: str | Path = "configs/datasets.yaml",
    limit: int | None = None,
) -> pd.DataFrame:
    """Load one configured corpus without filtering or reordering rows."""
    config = get_dataset_entry(dataset_key, datasets_config_path)
    loader = _LOADERS.get(config["loader"])
    if loader is None:
        raise ConfigError(f"Unsupported loader: {config['loader']}")
    logger.info("Loading %s in original corpus order", dataset_key)
    return loader(config, limit=limit)
