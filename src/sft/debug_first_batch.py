"""Run one SFT forward/backward pass with runtime shape diagnostics."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from src.sft.data import CausalLMCollator, load_jsonl_records, pack_translation_dataset
from src.sft.train import (
    _dataset,
    load_sft_base_model,
    load_sft_tokenizer,
    resolve_sft_settings,
)
from src.utils.config import get_model_entry, load_sft_config

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("en_ca", "en_eu"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--records", type=int, default=16)
    parser.add_argument("--internal-loss", action="store_true")
    args = parser.parse_args()

    root = load_sft_config()["sft"]
    settings = resolve_sft_settings(root, args.model)
    model_entry = get_model_entry(args.model)
    tokenizer = load_sft_tokenizer(model_entry)
    data_path = (
        Path(settings["data_dir"])
        / args.dataset
        / "sft"
        / str(settings["chat_format"]["directory"])
        / "train.jsonl"
    )
    records = load_jsonl_records(data_path)[: args.records]
    dataset = _dataset(
        records,
        tokenizer,
        model_entry,
        settings,
        str(settings["target_languages"][args.dataset]),
    )
    dataset = pack_translation_dataset(
        dataset, int(settings["chat_format"]["max_seq_length"])
    )
    batch = CausalLMCollator(int(tokenizer.pad_token_id))([dataset[0]])

    model = load_sft_base_model(model_entry, settings, training=True)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if bool(settings["quantization"].get("load_in_4bit", False)):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(
                settings["training"].get("gradient_checkpointing", False)
            ),
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
    )
    model.train()

    input_device = model.get_input_embeddings().weight.device
    output_rows = int(model.get_output_embeddings().weight.shape[0])
    labels = batch["labels"]
    supervised = labels[labels.ne(-100)]
    logger.info("hf_device_map=%s", getattr(model, "hf_device_map", None))
    logger.info(
        "tokenizer=%d output_rows=%d input_device=%s labels=[%d,%d]",
        len(tokenizer),
        output_rows,
        input_device,
        int(supervised.min()),
        int(supervised.max()),
    )
    if int(supervised.min()) < 0 or int(supervised.max()) >= output_rows:
        raise RuntimeError("Supervised label is outside the output vocabulary.")

    device_batch = {name: value.to(input_device) for name, value in batch.items()}
    labels = device_batch.pop("labels")
    if args.internal_loss:
        outputs = model(**device_batch, labels=labels, use_cache=False)
        logger.info("internal loss=%f; starting backward", float(outputs.loss.detach()))
        outputs.loss.backward()
        logger.info("Internal SFT loss forward/backward pass completed successfully.")
        return

    outputs = model(**device_batch, use_cache=False)
    logits = outputs.logits
    logger.info(
        "logits shape=%s dtype=%s device=%s",
        tuple(logits.shape),
        logits.dtype,
        logits.device,
    )
    shift_labels = F.pad(labels, (0, 1), value=-100)[..., 1:].to(logits.device)
    loss = F.cross_entropy(
        logits.float().view(-1, logits.shape[-1]),
        shift_labels.contiguous().view(-1),
        ignore_index=-100,
    )
    logger.info("manual loss=%f; starting backward", float(loss.detach()))
    loss.backward()
    logger.info("First SFT forward/backward pass completed successfully.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
