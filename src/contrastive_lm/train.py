"""Train target-side HT and MT causal language-model adapters."""

from __future__ import annotations

import argparse
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from src.contrastive_lm.data import load_labeled_texts, validate_classifier_style_splits
from src.utils.config import load_contrastive_lm_config
from src.utils.errors import PipelineError
from src.utils.hf_auth import get_hf_token
from src.utils.io import save_json

logger = logging.getLogger(__name__)


_SIDE_TO_KEY = {"ht": "ht", "mt": "mt"}


class CausalTextDataset(Dataset[dict[str, torch.Tensor]]):
    """Causal-LM data with loss on every non-padding target token."""

    def __init__(self, texts: list[str], tokenizer: Any, max_length: int) -> None:
        if not texts:
            raise PipelineError("Causal-LM dataset is empty.")
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[index],
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
        )
        input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids.clone(),
        }


@dataclass(frozen=True)
class CausalLMCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_length = max(int(feature["input_ids"].shape[0]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_tokenizer(model_name: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=get_hf_token(), use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise PipelineError("Tokenizer must provide an EOS or PAD token.")
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(model_name: str, load_in_4bit: bool, training: bool) -> Any:
    kwargs: dict[str, Any] = {"token": get_hf_token()}
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if training:
        model.config.use_cache = False
    return model


def _prepare_lora_model(model: Any, config: dict[str, Any]) -> Any:
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:  # pragma: no cover - project dependency
        raise PipelineError("peft is required for contrastive-LM training.") from exc
    if bool(config["model"]["load_in_4bit"]):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(config["training"]["gradient_checkpointing"]),
        )
    lora = config["model"]["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(lora["r"]),
            lora_alpha=int(lora["lora_alpha"]),
            lora_dropout=float(lora["lora_dropout"]),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(lora["target_modules"]),
        ),
    )
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable == 0:
        raise PipelineError("LoRA setup produced no trainable parameters.")
    return model


def train_contrastive_lm(
    dataset_key: str,
    side: str,
    config_path: str | Path = "configs/contrastive_lm.yaml",
) -> dict[str, str]:
    """Train either the HT or the MT adapter."""
    if side not in _SIDE_TO_KEY:
        raise PipelineError("side must be 'ht' or 'mt'.")
    config = load_contrastive_lm_config(config_path)["contrastive_lm"]
    if dataset_key not in config["languages"]:
        raise PipelineError(f"No contrastive-LM language config for {dataset_key}.")
    split_counts = validate_classifier_style_splits(config["data_root"], dataset_key)
    label = int(config["labels"][side])
    train_texts = load_labeled_texts(config["data_root"], dataset_key, "train", label)
    dev_texts = load_labeled_texts(config["data_root"], dataset_key, "dev", label)
    language = config["languages"][dataset_key]
    model_name = str(language["base_model"])
    training = config["training"]
    _set_seed(int(training["seed"]))

    tokenizer = _load_tokenizer(model_name)
    model = _prepare_lora_model(
        load_base_model(model_name, bool(config["model"]["load_in_4bit"]), training=True), config
    )
    train_dataset = CausalTextDataset(train_texts, tokenizer, int(config["model"]["max_length"]))
    dev_dataset = CausalTextDataset(dev_texts, tokenizer, int(config["model"]["max_length"]))
    collator = CausalLMCollator(int(tokenizer.pad_token_id))
    run_dir = Path(config["adapter_root"]) / dataset_key / side
    run_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(run_dir / "trainer"),
        num_train_epochs=float(training["num_train_epochs"]),
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        warmup_ratio=float(training["warmup_ratio"]),
        lr_scheduler_type=str(training["lr_scheduler_type"]),
        optim=str(training["optim"]),
        logging_steps=int(training["logging_steps"]),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=int(training["save_total_limit"]),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=bool(training["bf16"])
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=bool(training["gradient_checkpointing"]),
        report_to=[],
        remove_unused_columns=False,
        seed=int(training["seed"]),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    train_output = trainer.train()
    dev_metrics = trainer.evaluate()
    model.save_pretrained(run_dir)
    tokenizer.save_pretrained(run_dir)
    metadata_path = run_dir / "training_metadata.json"
    save_json(
        {
            "method": "contrastive_ht_mt_causal_lm",
            "dataset_key": dataset_key,
            "target_language": language["target_language"],
            "side": side,
            "label": label,
            "base_model": model_name,
            "adapter_dir": str(run_dir),
            "data_root": str(config["data_root"]),
            "classifier_split_rows": split_counts,
            "target_texts": {"train": len(train_texts), "dev": len(dev_texts)},
            "training": dict(training),
            "model": dict(config["model"]),
            "train_metrics": train_output.metrics,
            "dev_metrics": dev_metrics,
        },
        metadata_path,
    )
    return {"adapter_dir": str(run_dir), "metadata": str(metadata_path)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("en_eu", "en_ca"))
    parser.add_argument("--side", choices=("ht", "mt"), default=None)
    parser.add_argument("--all-sides", action="store_true")
    parser.add_argument("--config", default="configs/contrastive_lm.yaml")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if bool(args.side) == bool(args.all_sides):
        raise PipelineError("Choose exactly one of --side or --all-sides.")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    sides = (args.side,) if args.side else ("ht", "mt")
    for side in sides:
        logger.info("Training contrastive %s LM for %s", side, args.dataset)
        logger.info("Completed: %s", train_contrastive_lm(args.dataset, side, args.config))


if __name__ == "__main__":
    main()
