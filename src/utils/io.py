"""Write experiment outputs and compute file hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_parent_dir(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def save_json(data: Any, path: str | Path, indent: int = 2) -> Path:
    output_path = ensure_parent_dir(path)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=indent, sort_keys=True)
        handle.write("\n")
    return output_path


def save_jsonl(records: list[dict[str, Any]], path: str | Path) -> Path:
    output_path = ensure_parent_dir(path)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return output_path


def save_dataframe_jsonl(df: pd.DataFrame, path: str | Path) -> Path:
    records = df.where(pd.notna(df), None).to_dict(orient="records")
    return save_jsonl(records, path)


def save_dataframe_csv(df: pd.DataFrame, path: str | Path) -> Path:
    output_path = ensure_parent_dir(path)
    df.to_csv(output_path, index=False)
    return output_path


def save_dataframe_tsv(df: pd.DataFrame, path: str | Path) -> Path:
    output_path = ensure_parent_dir(path)
    df.to_csv(output_path, index=False, sep="\t")
    return output_path


def file_sha256(path: str | Path) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
