"""Diagnose one P5 policy rollout before an expensive GRPO run."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from src.grpo.data import build_grpo_records, load_grpo_records
from src.grpo.train import (
    _peft_adapter_loading_compatibility,
    resolve_grpo_settings,
)
from src.sft.train import load_sft_base_model, load_sft_tokenizer, resolve_sft_settings
from src.utils.config import get_model_entry, load_grpo_config, load_sft_config


logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("en_eu", "en_ca"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--cache-mode",
        choices=("cached", "uncached", "both"),
        default="cached",
    )
    args = parser.parse_args()

    grpo = resolve_grpo_settings(load_grpo_config()["grpo"], args.model)
    sft = resolve_sft_settings(load_sft_config()["sft"], args.model)
    model_entry = get_model_entry(args.model)
    tokenizer = load_sft_tokenizer(model_entry)
    tokenizer.padding_side = "left"
    record = build_grpo_records(
        load_grpo_records(grpo["data_dir"], args.dataset, "train")[:1],
        tokenizer,
        model_entry,
        str(grpo["target_languages"][args.dataset]),
        prompt_spec=grpo["chat_format"]["prompt_spec"],
        enable_thinking=bool(grpo["chat_format"].get("enable_thinking", False)),
    )[0]
    model = load_sft_base_model(model_entry, sft, training=False)
    model.eval()
    input_device = model.get_input_embeddings().weight.device
    encoded = tokenizer(record["prompt"], return_tensors="pt").to(input_device)
    prompt_length = int(encoded["input_ids"].shape[-1])
    logger.info("source=%r", record["source"])
    logger.info("reference=%r", record["reference"])
    logger.info("input_device=%s hf_device_map=%s", input_device, model.hf_device_map)

    cache_modes = {
        "cached": (True,),
        "uncached": (False,),
        "both": (False, True),
    }

    def generate(label: str) -> None:
        for use_cache in cache_modes[args.cache_mode]:
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=use_cache,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(
                generated[0][prompt_length:], skip_special_tokens=True
            )
            logger.info("model=%s use_cache=%s output=%r", label, use_cache, text)

    generate("base_before_peft")

    from peft import PeftModel

    with _peft_adapter_loading_compatibility(model):
        model = PeftModel.from_pretrained(model, Path(args.adapter_dir))
    model.eval()
    generate("p4_adapter")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
