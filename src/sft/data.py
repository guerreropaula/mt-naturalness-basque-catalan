"""Build and tokenize chat examples for translation SFT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import Dataset

from src.utils.config import ModelEntry
from src.utils.errors import PipelineError
from src.utils.model_adapters import (
    build_prompt_text,
    build_supervised_translation_text,
    build_translation_messages,
)


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise PipelineError(f"SFT data file not found: {file_path}")
    with file_path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise PipelineError(f"SFT data file is empty: {file_path}")
    return records


def build_chat_record(
    record: Mapping[str, Any],
    target_lang: str,
    prompt_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert one parallel example into a chat record."""
    required = {"id", "source", "target"}
    missing = required - set(record)
    if missing:
        raise PipelineError(f"Raw SFT record is missing fields: {sorted(missing)}")
    messages = build_translation_messages(
        str(record["source"]), target_lang=target_lang, prompt_spec=prompt_spec
    )
    messages.append({"role": "assistant", "content": str(record["target"])})
    return {
        "id": str(record["id"]),
        "source": str(record["source"]),
        "target": str(record["target"]),
        "language": target_lang,
        "messages": messages,
    }


def _common_prefix_length(left: list[int], right: list[int]) -> int:
    length = 0
    for first, second in zip(left, right):
        if first != second:
            break
        length += 1
    return length


class TranslationChatDataset(Dataset[dict[str, torch.Tensor]]):
    """Tokenize chats and mask prompt tokens from the training loss."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: Any,
        model_entry: ModelEntry,
        target_lang: str,
        max_seq_length: int,
        enable_thinking: bool = False,
        prompt_spec: Mapping[str, Any] | None = None,
    ) -> None:
        if not records:
            raise PipelineError("SFT dataset is empty.")
        if max_seq_length <= 0:
            raise PipelineError("max_seq_length must be positive.")
        self.records = records
        self.tokenizer = tokenizer
        self.model_entry = model_entry
        self.target_lang = target_lang
        self.max_seq_length = max_seq_length
        self.enable_thinking = enable_thinking
        self.prompt_spec = prompt_spec

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        source = str(record["source"])
        target = str(record["target"])
        prompt = build_prompt_text(
            self.model_entry,
            source,
            self.target_lang,
            tokenizer=self.tokenizer,
            enable_thinking=self.enable_thinking,
            prompt_spec=self.prompt_spec,
        )
        full_text = build_supervised_translation_text(
            self.model_entry,
            source,
            target,
            self.target_lang,
            tokenizer=self.tokenizer,
            enable_thinking=self.enable_thinking,
            prompt_spec=self.prompt_spec,
        )
        full_ids = list(self.tokenizer(full_text, add_special_tokens=False)["input_ids"])
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is not None and (not full_ids or full_ids[-1] != eos_token_id):
            full_ids.append(eos_token_id)
        full_ids = full_ids[: self.max_seq_length]
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        supervised_start = _common_prefix_length(prompt_ids, full_ids)
        labels = list(full_ids)
        labels[:supervised_start] = [-100] * supervised_start
        if not any(token != -100 for token in labels):
            raise PipelineError(
                f"Example {record.get('id', index)} has no assistant tokens within max_seq_length. "
                "Increase chat_format.max_seq_length."
            )
        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class PackedTranslationChatDataset(Dataset[dict[str, torch.Tensor]]):
    """Pack consecutive complete SFT examples into sequences no longer than ``max_seq_length``."""

    def __init__(self, dataset: Dataset[dict[str, torch.Tensor]], max_seq_length: int) -> None:
        if max_seq_length <= 0:
            raise PipelineError("max_seq_length must be positive for sequence packing.")
        self.original_examples = len(dataset)
        self.max_seq_length = max_seq_length
        self.features = self._pack(dataset)
        if not self.features:
            raise PipelineError("Sequence packing produced an empty SFT dataset.")

    def _pack(self, dataset: Dataset[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
        packed: list[dict[str, torch.Tensor]] = []
        current: dict[str, list[int]] = {"input_ids": [], "attention_mask": [], "labels": []}
        for index in range(len(dataset)):
            feature = dataset[index]
            values = {name: feature[name].tolist() for name in current}
            length = len(values["input_ids"])
            if length == 0:
                raise PipelineError(f"SFT feature {index} is empty and cannot be packed.")
            if length > self.max_seq_length:
                raise PipelineError(
                    f"SFT feature {index} has {length} tokens, above max_seq_length={self.max_seq_length}."
                )
            if current["input_ids"] and len(current["input_ids"]) + length > self.max_seq_length:
                packed.append(self._as_feature(current))
                current = {"input_ids": [], "attention_mask": [], "labels": []}
            for name, values_for_name in values.items():
                current[name].extend(values_for_name)
        if current["input_ids"]:
            packed.append(self._as_feature(current))
        return packed

    @staticmethod
    def _as_feature(values: dict[str, list[int]]) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(values["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(values["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(values["labels"], dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.features[index]


def pack_translation_dataset(
    dataset: Dataset[dict[str, torch.Tensor]], max_seq_length: int
) -> PackedTranslationChatDataset:
    """Apply complete-example packing while retaining the existing loss masks and EOS boundaries."""
    return PackedTranslationChatDataset(dataset, max_seq_length)


class CausalLMCollator:
    """Right-pad input IDs while preserving -100 label padding."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_length = max(int(feature["input_ids"].shape[0]) for feature in features)
        batch: dict[str, list[torch.Tensor]] = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding = max_length - int(feature["input_ids"].shape[0])
            batch["input_ids"].append(
                torch.nn.functional.pad(feature["input_ids"], (0, padding), value=self.pad_token_id)
            )
            batch["attention_mask"].append(
                torch.nn.functional.pad(feature["attention_mask"], (0, padding), value=0)
            )
            batch["labels"].append(
                torch.nn.functional.pad(feature["labels"], (0, padding), value=-100)
            )
        return {name: torch.stack(values) for name, values in batch.items()}
