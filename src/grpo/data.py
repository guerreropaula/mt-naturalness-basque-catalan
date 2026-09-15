"""Prepare ordered parallel GRPO records with prompts and gold references."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.sft.data import load_jsonl_records
from src.utils.config import ModelEntry
from src.utils.model_adapters import build_prompt_text


class GRPODataError(RuntimeError):
    """Raised when the GRPO source data is malformed."""


def build_grpo_records(
    records: list[Mapping[str, Any]],
    tokenizer: Any,
    model_entry: ModelEntry,
    target_lang: str,
    prompt_spec: Mapping[str, Any] | None = None,
    enable_thinking: bool = False,
) -> list[dict[str, str]]:
    """Create TRL-ready records while preserving source/reference provenance."""
    prepared: list[dict[str, str]] = []
    for index, record in enumerate(records):
        missing = {"id", "source", "target"} - set(record)
        if missing:
            raise GRPODataError(f"GRPO record {index} is missing fields: {sorted(missing)}")
        source = str(record["source"])
        prepared.append(
            {
                "sentence_id": str(record["id"]),
                "prompt": build_prompt_text(
                    model_entry,
                    source,
                    target_lang,
                    tokenizer=tokenizer,
                    enable_thinking=enable_thinking,
                    prompt_spec=prompt_spec,
                ),
                "source": source,
                "reference": str(record["target"]),
            }
        )
    if not prepared:
        raise GRPODataError("GRPO split is empty.")
    return prepared


def load_grpo_records(data_dir: str | Path, dataset_key: str, split: str) -> list[dict[str, Any]]:
    """Load one fixed ordered GRPO split from the training layout."""
    if split not in {"train", "dev"}:
        raise GRPODataError(f"Unsupported GRPO split '{split}'.")
    return load_jsonl_records(Path(data_dir) / dataset_key / "grpo" / f"{split}.jsonl")
