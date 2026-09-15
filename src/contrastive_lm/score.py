"""Score target text with paired HT and MT causal language-model adapters."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.contrastive_lm.train import load_base_model
from src.utils.config import load_contrastive_lm_config
from src.utils.hf_auth import get_hf_token
from src.utils.io import save_json, save_jsonl

logger = logging.getLogger(__name__)


class ContrastiveLMScoringError(RuntimeError):
    """Raised when target-side HT/MT scoring cannot be completed safely."""


def _input_device(model: Any) -> torch.device:
    return model.get_input_embeddings().weight.device


def load_adapter_lm(base_model: str, adapter_dir: Path, load_in_4bit: bool) -> tuple[Any, Any]:
    if not adapter_dir.exists():
        raise ContrastiveLMScoringError(f"Contrastive-LM adapter not found: {adapter_dir}")
    try:
        from peft import PeftModel
    except ImportError as exc:  # pragma: no cover - project dependency
        raise ContrastiveLMScoringError("peft is required for contrastive-LM scoring.") from exc
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, token=get_hf_token(), use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ContrastiveLMScoringError("Tokenizer must provide an EOS or PAD token.")
    tokenizer.padding_side = "right"
    model = PeftModel.from_pretrained(
        load_base_model(base_model, load_in_4bit=load_in_4bit, training=False), adapter_dir
    )
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def average_token_logprobs(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    *,
    batch_size: int,
    max_length: int,
) -> list[float | None]:
    """Return mean next-token log probability per text, ignoring padding tokens."""
    if batch_size < 1 or max_length < 2:
        raise ContrastiveLMScoringError("batch_size must be positive and max_length at least two.")
    scores: list[float | None] = []
    device = _input_device(model)
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        logits = model(**encoded).logits[:, :-1, :]
        targets = encoded["input_ids"][:, 1:]
        valid = encoded["attention_mask"][:, 1:].bool()
        token_logprobs = F.log_softmax(logits.float(), dim=-1).gather(
            dim=-1, index=targets.unsqueeze(-1)
        ).squeeze(-1)
        for values, mask in zip(token_logprobs, valid, strict=True):
            token_count = int(mask.sum().item())
            scores.append(float(values[mask].mean().item()) if token_count else None)
    return scores


def annotate_records(
    records: Iterable[dict[str, Any]],
    ht_logprobs: Iterable[float | None],
    mt_logprobs: Iterable[float | None],
    *,
    threshold: float | None,
) -> list[dict[str, Any]]:
    """Preserve each JSONL record while adding HT/MT contrastive scores."""
    annotated: list[dict[str, Any]] = []
    for record, ht_value, mt_value in zip(records, ht_logprobs, mt_logprobs, strict=True):
        result = dict(record)
        result["ht_logprob"] = ht_value
        result["mt_logprob"] = mt_value
        score = None if ht_value is None or mt_value is None else float(ht_value - mt_value)
        result["htmt_score"] = score
        if threshold is not None:
            result["htmt_label"] = None if score is None else ("ht_like" if score >= threshold else "mt_like")
        annotated.append(result)
    return annotated


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ContrastiveLMScoringError(f"Input JSONL not found: {path}")
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ContrastiveLMScoringError("Input JSONL is empty.")
    return records


def score_jsonl(
    dataset_key: str,
    input_path: str | Path,
    output_path: str | Path,
    *,
    text_field: str,
    threshold: float | None = None,
    config_path: str | Path = "configs/contrastive_lm.yaml",
) -> dict[str, str]:
    """Score a JSONL file with the HT-versus-MT contrastive-LM difference."""
    config = load_contrastive_lm_config(config_path)["contrastive_lm"]
    if dataset_key not in config["languages"]:
        raise ContrastiveLMScoringError(f"No contrastive-LM language config for {dataset_key}.")
    records = _read_jsonl(Path(input_path))
    if any(text_field not in record for record in records):
        raise ContrastiveLMScoringError(f"Every record must contain text field {text_field!r}.")
    texts = [str(record[text_field]) for record in records]
    language = config["languages"][dataset_key]
    adapter_root = Path(config["adapter_root"]) / dataset_key
    model_config = config["model"]
    scoring_config = config["scoring"]
    ht_model, ht_tokenizer = load_adapter_lm(
        str(language["base_model"]), adapter_root / "ht", bool(model_config["load_in_4bit"])
    )
    ht_scores = average_token_logprobs(
        ht_model, ht_tokenizer, texts,
        batch_size=int(scoring_config["batch_size"]), max_length=int(model_config["max_length"]),
    )
    del ht_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mt_model, mt_tokenizer = load_adapter_lm(
        str(language["base_model"]), adapter_root / "mt", bool(model_config["load_in_4bit"])
    )
    mt_scores = average_token_logprobs(
        mt_model, mt_tokenizer, texts,
        batch_size=int(scoring_config["batch_size"]), max_length=int(model_config["max_length"]),
    )
    del mt_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    annotated = annotate_records(records, ht_scores, mt_scores, threshold=threshold)
    output_path = Path(output_path)
    save_jsonl(annotated, output_path)
    metadata_path = output_path.with_suffix(".metadata.json")
    save_json(
        {
            "method": "contrastive_ht_mt_causal_lm",
            "dataset_key": dataset_key,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "text_field": text_field,
            "threshold": threshold,
            "formula": "avg_logprob_ht - avg_logprob_mt",
            "adapters": {"ht": str(adapter_root / "ht"), "mt": str(adapter_root / "mt")},
            "base_model": language["base_model"],
            "scoring": dict(scoring_config),
        },
        metadata_path,
    )
    return {"scored_jsonl": str(output_path), "metadata": str(metadata_path)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("en_eu", "en_ca"))
    parser.add_argument("--input", required=True, dest="input_path")
    parser.add_argument("--output", required=True, dest="output_path")
    parser.add_argument("--text-field", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--config", default="configs/contrastive_lm.yaml")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_contrastive_lm_config(args.config)["contrastive_lm"]
    text_field = args.text_field or str(config["scoring"]["text_field"])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.info(
        "Contrastive HT/MT scoring complete: %s",
        score_jsonl(
            args.dataset, args.input_path, args.output_path,
            text_field=text_field, threshold=args.threshold, config_path=args.config,
        ),
    )


if __name__ == "__main__":
    main()
