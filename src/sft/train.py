"""Registry-driven QLoRA/LoRA supervised fine-tuning for translation."""

from __future__ import annotations

import argparse
import copy
import logging
from datetime import timedelta
import os
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

os.environ.pop("TRANSFORMERS_CACHE", None)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import Trainer, TrainingArguments, set_seed

from src.sft.data import (
    CausalLMCollator,
    TranslationChatDataset,
    load_jsonl_records,
    pack_translation_dataset,
)
from src.utils.config import ModelEntry, get_model_entry, load_sft_config
from src.utils.hf_auth import get_hf_token
from src.utils.io import save_json

logger = logging.getLogger(__name__)


class SFTTrainingError(RuntimeError):
    """Raised when an SFT run cannot be configured safely."""


class ModelParallelCausalLMTrainer(Trainer):
    """Compute causal loss after dispatched logits return to the input device."""

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> Any:
        model_inputs = dict(inputs)
        labels = model_inputs.pop("labels")
        outputs = model(**model_inputs)
        logits = outputs.logits
        labels = labels.to(logits.device)
        shift_labels = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
        supervised = shift_labels[shift_labels.ne(-100)]
        vocab_size = int(logits.shape[-1])
        if supervised.numel() and (
            int(supervised.min()) < 0 or int(supervised.max()) >= vocab_size
        ):
            raise SFTTrainingError(
                f"SFT label range [{int(supervised.min())}, {int(supervised.max())}] "
                f"is outside model vocabulary [0, {vocab_size})."
            )
        reduction = "sum" if num_items_in_batch is not None else "mean"
        loss = F.cross_entropy(
            logits.float().view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction=reduction,
        )
        if num_items_in_batch is not None:
            loss = loss / num_items_in_batch.to(loss.device)
        return (loss, outputs) if return_outputs else loss


def _merge_mapping(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_mapping(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def resolve_sft_settings(config: Mapping[str, Any], model_key: str) -> dict[str, Any]:
    overrides = config.get("model_overrides", {})
    override = overrides.get(model_key, {})
    if not isinstance(override, Mapping):
        raise SFTTrainingError(f"sft.model_overrides.{model_key} must be a mapping")
    return _merge_mapping(config, override)


def _torch_dtype(use_bf16: bool) -> torch.dtype:
    if use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else torch.float32


def align_fsdp_qlora_parameter_dtypes(
    model: Any,
    target_dtype: torch.dtype,
) -> int:
    """Give FSDP one floating dtype after PEFT's k-bit preparation.

    PEFT promotes regular low-precision parameters such as layer norms to
    float32. FSDP1 cannot flatten those together with QLoRA parameters whose
    4-bit storage dtype is bfloat16, so distributed QLoRA must restore one
    storage dtype before wrapping.
    """
    converted = 0
    converted_numel = 0
    for name, parameter in model.named_parameters():
        if parameter.__class__.__name__ == "Params4bit":
            if parameter.dtype != target_dtype:
                raise SFTTrainingError(
                    f"FSDP QLoRA parameter {name} uses {parameter.dtype}, not "
                    f"configured storage dtype {target_dtype}."
                )
            continue
        if parameter.is_floating_point() and parameter.dtype != target_dtype:
            parameter.data = parameter.data.to(target_dtype)
            converted += 1
            converted_numel += int(parameter.numel())
    remaining_dtypes = {
        parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()
    }
    if remaining_dtypes != {target_dtype}:
        raise SFTTrainingError(
            "FSDP QLoRA still has mixed floating parameter dtypes after alignment: "
            f"{sorted(map(str, remaining_dtypes))}."
        )
    logger.info(
        "Aligned %d FSDP QLoRA tensors (%d parameters) to %s",
        converted,
        converted_numel,
        target_dtype,
    )
    return converted


def prepare_model_for_distributed_qlora_training(
    model: Any,
    use_gradient_checkpointing: bool,
    gradient_checkpointing_kwargs: Mapping[str, Any],
) -> Any:
    """Freeze a k-bit base model without PEFT's memory-heavy fp32 upcast."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    if not use_gradient_checkpointing:
        return model

    checkpointing_kwargs = dict(gradient_checkpointing_kwargs)
    if checkpointing_kwargs.get("use_reentrant", True):
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(_module: Any, _inputs: Any, output: Any) -> None:
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=checkpointing_kwargs)
    except TypeError:
        if checkpointing_kwargs:
            logger.warning(
                "This model does not accept gradient_checkpointing_kwargs; "
                "enabling gradient checkpointing without them."
            )
        model.gradient_checkpointing_enable()
    return model


def distributed_world_size() -> int:
    """Return the process count set by torchrun or Accelerate."""
    return max(1, int(os.environ.get("WORLD_SIZE", "1")))


def distributed_rank() -> int:
    """Return the global process rank, defaulting to the only process."""
    return int(os.environ.get("RANK", "0"))


def local_process_index() -> int:
    """Return the CUDA index assigned to this process."""
    return int(os.environ.get("LOCAL_RANK", "0"))


def _validate_trainable_lora_parameters(model: Any) -> int:
    """Refuse an SFT run unless the newly attached LoRA adapter can learn."""
    lora_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" in name.lower()
    ]
    if not lora_parameters:
        raise SFTTrainingError(
            "P4 attached no trainable LoRA parameters. Refusing to train an inert adapter."
        )
    parameter_count = sum(int(parameter.numel()) for _, parameter in lora_parameters)
    if parameter_count <= 0:  # pragma: no cover - tensors cannot have negative size
        raise SFTTrainingError("P4 LoRA adapter has no trainable parameters.")
    logger.info(
        "P4 LoRA trainable parameters: %d tensors, %d parameters",
        len(lora_parameters),
        parameter_count,
    )
    return parameter_count


def load_sft_tokenizer(model_entry: ModelEntry) -> Any:
    token = get_hf_token()
    if model_entry.gated and not token:
        raise SFTTrainingError(f"Model {model_entry.key} is gated and requires HF_TOKEN.")
    tokenizer = AutoTokenizer.from_pretrained(
        model_entry.hf_id, token=token, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_sft_base_model(
    model_entry: ModelEntry, settings: Mapping[str, Any], training: bool
) -> Any:
    token = get_hf_token()
    if model_entry.gated and not token:
        raise SFTTrainingError(f"Model {model_entry.key} is gated and requires HF_TOKEN.")
    quantization = settings["quantization"]
    use_bf16 = bool(settings["training"].get("bf16", True))
    world_size = distributed_world_size()
    process_index = local_process_index()
    if world_size > 1:
        if not torch.cuda.is_available():
            raise SFTTrainingError("Distributed QLoRA requires CUDA.")
        from accelerate import PartialState

        timeout_minutes = int(settings["training"].get("distributed_timeout_minutes", 10))
        state = PartialState(timeout=timedelta(minutes=timeout_minutes))
        process_index = int(state.process_index)
        torch.cuda.set_device(int(state.local_process_index))
    if world_size > 1:
        device_map: str | dict[str, int] = {"": process_index}
    elif int(model_entry.tensor_parallel_size) > 1:
        # A large single-process training model must be balanced across its
        # assigned GPUs so rollout and reward models retain memory headroom.
        device_map = "balanced"
    else:
        device_map = "auto"
    load_kwargs: dict[str, Any] = {
        "token": token,
        "trust_remote_code": True,
        "device_map": device_map,
        "dtype": _torch_dtype(use_bf16),
    }
    if bool(quantization.get("load_in_4bit", False)):
        storage_name = str(quantization.get("bnb_4bit_quant_storage", "uint8"))
        storage_dtype = getattr(torch, storage_name, None)
        if not isinstance(storage_dtype, torch.dtype):
            raise SFTTrainingError(
                f"Unknown quantization.bnb_4bit_quant_storage dtype: {storage_name}"
            )
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(quantization.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(quantization.get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_compute_dtype=_torch_dtype(use_bf16),
            bnb_4bit_quant_storage=storage_dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(model_entry.hf_id, **load_kwargs)
    if training and bool(settings["training"].get("gradient_checkpointing", False)):
        model.config.use_cache = False
    return model


def _dataset(
    records: list[dict[str, Any]],
    tokenizer: Any,
    model_entry: ModelEntry,
    settings: Mapping[str, Any],
    target_lang: str,
) -> TranslationChatDataset:
    chat = settings["chat_format"]
    return TranslationChatDataset(
        records,
        tokenizer,
        model_entry,
        target_lang,
        max_seq_length=int(chat["max_seq_length"]),
        enable_thinking=bool(chat.get("enable_thinking", False)),
        prompt_spec=chat.get("prompt_spec"),
    )


def train_sft(
    dataset_key: str,
    model_key: str,
    config_path: str | Path = "configs/sft.yaml",
    models_config_path: str | Path = "configs/models.yaml",
    data_dir: str | Path | None = None,
    adapter_root: str | Path | None = None,
    results_root: str | Path | None = None,
    resume_from_checkpoint: str | None = None,
    max_train_examples: int | None = None,
    max_dev_examples: int | None = None,
) -> dict[str, str]:
    """Fine-tune one registered causal LM with sft_train and sft_dev only."""
    config = load_sft_config(config_path)["sft"]
    if dataset_key not in config["target_languages"]:
        raise SFTTrainingError(f"Unsupported SFT dataset: {dataset_key}")
    settings = resolve_sft_settings(config, model_key)
    model_entry = get_model_entry(model_key, models_config_path)
    target_lang = str(settings["target_languages"][dataset_key])
    training = settings["training"]
    world_size = distributed_world_size()
    required_world_size = int(training.get("required_world_size", world_size))
    if world_size != required_world_size:
        raise SFTTrainingError(
            f"{model_key} SFT requires {required_world_size} distributed processes; "
            f"found {world_size}."
        )
    effective_batch_size = (
        int(training["per_device_train_batch_size"])
        * int(training["gradient_accumulation_steps"])
        * world_size
    )
    configured_batch_size = int(training.get("effective_batch_size", effective_batch_size))
    if effective_batch_size != configured_batch_size:
        raise SFTTrainingError(
            f"Effective SFT batch is {effective_batch_size}, not configured "
            f"{configured_batch_size}."
        )
    set_seed(int(training["seed"]))

    data_root = Path(data_dir or settings["data_dir"]) / dataset_key / "sft"
    chat_root = data_root / str(settings["chat_format"]["directory"])
    records = {
        split: load_jsonl_records(chat_root / f"{split}.jsonl") for split in ("train", "dev")
    }
    for split, limit in (("train", max_train_examples), ("dev", max_dev_examples)):
        if limit is not None:
            if limit <= 0:
                raise SFTTrainingError(f"max_{split}_examples must be positive.")
            records[split] = records[split][:limit]
    adapter_dir = Path(adapter_root or settings["adapter_root"]) / dataset_key / model_key
    if adapter_dir.exists() and not resume_from_checkpoint:
        raise FileExistsError(f"Adapter directory already exists: {adapter_dir}")
    run_dir = Path(results_root or settings["results_root"]) / dataset_key / model_key
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_sft_tokenizer(model_entry)
    model = load_sft_base_model(model_entry, settings, training=True)
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:  # pragma: no cover
        raise SFTTrainingError("peft is required for SFT.") from exc
    if bool(settings["quantization"].get("load_in_4bit", False)):
        gradient_checkpointing = bool(training.get("gradient_checkpointing", False))
        gradient_checkpointing_kwargs = dict(training.get("gradient_checkpointing_kwargs", {}))
        if world_size > 1:
            model = prepare_model_for_distributed_qlora_training(
                model,
                gradient_checkpointing,
                gradient_checkpointing_kwargs,
            )
        else:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=gradient_checkpointing,
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
            )
    lora = settings["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(lora["r"]),
            lora_alpha=int(lora["lora_alpha"]),
            lora_dropout=float(lora["lora_dropout"]),
            bias=str(lora["bias"]),
            task_type=str(lora["task_type"]),
            target_modules=list(lora["target_modules"]),
        ),
        autocast_adapter_dtype=bool(lora.get("autocast_adapter_dtype", True)),
    )
    if world_size > 1 and bool(settings["quantization"].get("load_in_4bit", False)):
        align_fsdp_qlora_parameter_dtypes(
            model,
            _torch_dtype(bool(training.get("bf16", True))),
        )
    trainable_lora_parameters = _validate_trainable_lora_parameters(model)
    train_dataset = _dataset(records["train"], tokenizer, model_entry, settings, target_lang)
    dev_dataset = _dataset(records["dev"], tokenizer, model_entry, settings, target_lang)
    sequence_packing = bool(training["packing"])
    if sequence_packing:
        max_seq_length = int(settings["chat_format"]["max_seq_length"])
        train_dataset = pack_translation_dataset(train_dataset, max_seq_length)
        dev_dataset = pack_translation_dataset(dev_dataset, max_seq_length)
    collator = CausalLMCollator(int(tokenizer.pad_token_id))

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
        eval_strategy=str(training["eval_strategy"]),
        eval_steps=int(training["eval_steps"]),
        save_strategy=str(training["save_strategy"]),
        save_steps=int(training["save_steps"]),
        save_total_limit=int(training["save_total_limit"]),
        save_only_model=bool(training.get("save_only_model", False)),
        gradient_checkpointing=bool(training["gradient_checkpointing"]),
        gradient_checkpointing_kwargs=dict(training.get("gradient_checkpointing_kwargs", {})),
        bf16=bool(training["bf16"])
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported(),
        tf32=bool(training["tf32"]) and torch.cuda.is_available(),
        report_to=[],
        load_best_model_at_end=bool(training.get("load_best_model_at_end", True)),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
    )
    dispatched_devices = {
        device
        for device in getattr(model, "hf_device_map", {}).values()
        if device not in {"cpu", "disk"}
    }
    trainer_class = ModelParallelCausalLMTrainer if len(dispatched_devices) > 1 else Trainer
    if trainer_class is ModelParallelCausalLMTrainer:
        logger.info(
            "Using explicit causal loss for model-parallel SFT across %d GPUs.",
            len(dispatched_devices),
        )
    trainer = trainer_class(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    train_output = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.accelerator.wait_for_everyone()
    trainer.save_model(str(adapter_dir))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(adapter_dir)
    trainer.accelerator.wait_for_everyone()
    metadata = {
        "experiment_key": "p4_sft",
        "stage": "post_training",
        "dataset_key": dataset_key,
        "model_key": model_key,
        "base_model": model_entry.hf_id,
        "prompt_adapter": model_entry.prompt_adapter,
        "adapter_dir": str(adapter_dir),
        "trainable_lora_parameters": trainable_lora_parameters,
        "distributed": {
            "world_size": distributed_world_size(),
            "fsdp": bool(trainer.is_fsdp_enabled),
        },
        "training_metrics": dict(train_output.metrics),
        "optimizer_steps": int(trainer.state.global_step),
        "max_train_examples": max_train_examples,
        "max_dev_examples": max_dev_examples,
        "data": {
            "train": len(records["train"]),
            "dev": len(records["dev"]),
            "training_sequences": len(train_dataset),
            "dev_sequences": len(dev_dataset),
            "sequence_packing": sequence_packing,
            "training_selection": "sft_train; validation on sft_dev",
        },
        "settings": settings,
    }
    metadata_path = run_dir / "training_metadata.json"
    if trainer.is_world_process_zero():
        save_json(metadata, metadata_path)
    trainer.accelerator.wait_for_everyone()
    return {
        "adapter_dir": str(adapter_dir),
        "training_metadata": str(metadata_path),
        "trainer_dir": str(run_dir / "trainer"),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a LoRA SFT translation adapter.")
    parser.add_argument("--dataset", required=True, choices=("en_eu", "en_ca"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--adapter-root", default=None)
    parser.add_argument("--results-root", default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-dev-examples", type=int, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    paths = train_sft(
        args.dataset,
        args.model,
        args.config,
        args.models_config,
        args.data_dir,
        args.adapter_root,
        args.results_root,
        args.resume_from_checkpoint,
        args.max_train_examples,
        args.max_dev_examples,
    )
    logger.info("SFT training complete: %s", paths)


if __name__ == "__main__":
    main()
