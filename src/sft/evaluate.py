"""Generate and score SFT adapters on a held-out global evaluation split."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.evaluation.metrics.automatic import compute_automatic_metrics
from src.sft.data import load_jsonl_records
from src.sft.train import load_sft_base_model, load_sft_tokenizer, resolve_sft_settings
from src.utils.config import get_model_entry, load_sft_config
from src.utils.io import save_dataframe_jsonl, save_json
from src.utils.model_adapters import build_prompt_text
from src.utils.model_loader import (
    generate_batch,
    load_inference_model,
    release_inference_model,
)

logger = logging.getLogger(__name__)

_BOUNDARY_TAG_ALIASES = ("s", "sentence", "sententzia")


def _build_boundary_prompt_spec(expected_count: int) -> dict[str, str]:
    """Build a language-neutral fixed-unit output contract."""
    output_shape = "\n".join(
        f"<s{index}>translation {index}</s{index}>"
        for index in range(1, expected_count + 1)
    )
    return {
        "label": "fixed_sentence_boundary_translation_v2",
        "single_user_prompt_template": (
            f"The input contains {expected_count} separately tagged English sentences. "
            "Translate each sentence into {target_lang} without merging, omitting, "
            "adding, or renumbering sentences. Copy the short tags exactly and write "
            "nothing outside them. Return exactly this structure:\n"
            f"{output_shape}\n\n{{source}}"
        ),
    }


class SFTEvaluationError(RuntimeError):
    """Raised when held-out SFT evaluation cannot be run safely."""


def _input_device(model: Any) -> torch.device:
    return model.get_input_embeddings().weight.device


def _tag_fixed_sentence_units(source: str, expected_count: int) -> str:
    """Render double-newline sentence units with explicit, stable tags."""
    if expected_count < 1:
        raise SFTEvaluationError("sentence_boundary_count must be positive.")
    units = [unit.strip() for unit in re.split(r"\n\s*\n", source) if unit.strip()]
    if len(units) != expected_count:
        raise SFTEvaluationError(
            f"Boundary-preserving evaluation expected {expected_count} source units, "
            f"but found {len(units)}."
        )
    return "\n".join(
        f"<s{index}>{unit}</s{index}>"
        for index, unit in enumerate(units, start=1)
    )


def _strip_boundary_tags(output: str) -> str:
    """Remove evaluation-only markers without changing translation content."""
    aliases = "|".join(re.escape(alias) for alias in _BOUNDARY_TAG_ALIASES)
    marker = re.compile(
        rf"</?\s*(?:{aliases})_?\d+\s*>?", flags=re.IGNORECASE
    )
    return marker.sub("", output).strip()


def _extract_fixed_sentence_units(
    output: str,
    expected_count: int,
) -> tuple[str, bool, str | None]:
    """Extract an exact tagged translation contract without inventing boundaries."""
    aliases = "|".join(re.escape(alias) for alias in _BOUNDARY_TAG_ALIASES)
    pattern = re.compile(
        rf"<\s*(?:{aliases})_?(?P<index>\d+)\s*>?\s*"
        rf"(?P<unit>.*?)\s*</\s*(?:{aliases})_?(?P=index)\s*>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    matches = list(pattern.finditer(output))
    indexes = [int(match.group("index")) for match in matches]
    expected_indexes = list(range(1, expected_count + 1))
    if indexes != expected_indexes:
        return _strip_boundary_tags(output), False, (
            f"expected tagged sentence indexes {expected_indexes}, found {indexes}"
        )
    units = [match.group("unit").strip() for match in matches]
    if any(not unit for unit in units):
        return _strip_boundary_tags(output), False, "one or more tagged translations are empty"
    return "\n\n".join(units), True, None


@torch.inference_mode()
def _generate(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    settings: dict[str, Any],
    *,
    max_input_length: int | None = None,
    max_new_tokens: int | None = None,
) -> list[str]:
    evaluation = settings["evaluation"]
    encoded = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
        max_length=int(max_input_length or settings["chat_format"]["max_seq_length"]),
    )
    encoded = {key: value.to(_input_device(model)) for key, value in encoded.items()}
    do_sample = bool(evaluation["do_sample"])
    num_beams = int(evaluation["num_beams"])
    if not do_sample and (
        num_beams != 1
        or evaluation.get("temperature") is not None
        or evaluation.get("top_p") is not None
    ):
        raise SFTEvaluationError(
            "Deterministic P4 evaluation requires do_sample=false, num_beams=1, "
            "temperature=null, and top_p=null."
        )
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens or evaluation["max_new_tokens"]),
        "do_sample": do_sample,
        "num_beams": num_beams,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = float(evaluation["temperature"])
        top_p = evaluation.get("top_p")
        if top_p is not None:
            generation_kwargs["top_p"] = float(top_p)
    output_ids = model.generate(**encoded, **generation_kwargs)
    continuation_ids = output_ids[:, encoded["input_ids"].shape[1] :]
    return [text.strip() for text in tokenizer.batch_decode(continuation_ids, skip_special_tokens=True)]


VALID_EVAL_SPLITS = ("test",)


def evaluate_sft_split(
    dataset_key: str,
    model_key: str,
    config_path: str | Path = "configs/sft.yaml",
    models_config_path: str | Path = "configs/models.yaml",
    data_dir: str | Path | None = None,
    adapter_root: str | Path | None = None,
    results_root: str | Path | None = None,
    split: str = "test",
    limit: int | None = None,
    experiment_key: str = "p4_sft",
    evaluation_file: str | Path | None = None,
    output_dataset_key: str | None = None,
    max_input_length: int | None = None,
    max_new_tokens: int | None = None,
    resume: bool = True,
    sentence_boundary_count: int | None = None,
) -> dict[str, str]:
    """Evaluate an adapter on the in-domain test set or an explicit immutable file."""
    if evaluation_file is None and split not in VALID_EVAL_SPLITS:
        raise SFTEvaluationError(
            f"Unsupported SFT evaluation split: {split}. Choose one of {', '.join(VALID_EVAL_SPLITS)}, "
            "or provide evaluation_file."
        )
    config = load_sft_config(config_path)["sft"]
    if dataset_key not in config["target_languages"]:
        raise SFTEvaluationError(f"Unsupported SFT dataset: {dataset_key}")
    settings = resolve_sft_settings(config, model_key)
    model_entry = get_model_entry(model_key, models_config_path)
    target_lang = str(settings["target_languages"][dataset_key])
    root = Path(data_dir or settings["data_dir"]) / dataset_key
    eval_path = (
        Path(evaluation_file)
        if evaluation_file is not None
        else root / "eval" / f"{split}.jsonl"
    )
    records = load_jsonl_records(eval_path)
    if limit is not None:
        records = records[:limit]
    if not records:
        raise SFTEvaluationError(f"{split} is empty.")
    for index, record in enumerate(records, start=1):
        if not record.get("id"):
            record["id"] = str(
                record.get("sentence_id") or f"{output_dataset_key or dataset_key}_{index:06d}"
            )

    adapter_dir = Path(adapter_root or settings["adapter_root"]) / dataset_key / model_key
    if not adapter_dir.exists():
        raise SFTEvaluationError(f"SFT adapter not found: {adapter_dir}")

    result_dataset_key = output_dataset_key or dataset_key
    output_dir = Path(results_root or settings["results_root"]) / result_dataset_key / model_key / split
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "automatic_metrics.json"
    metadata_path = output_dir / "evaluation_metadata.json"

    records_by_id = {str(record["id"]): record for record in records}
    if len(records_by_id) != len(records):
        raise SFTEvaluationError(f"{eval_path} contains duplicate evaluation IDs.")
    predictions_by_id: dict[str, dict[str, Any]] = {}
    if resume and predictions_path.is_file():
        for prediction in load_jsonl_records(predictions_path):
            prediction_id = str(prediction.get("id") or prediction.get("sentence_id") or "")
            if prediction_id not in records_by_id:
                raise SFTEvaluationError(
                    f"Existing prediction ID {prediction_id!r} is not present in {eval_path}."
                )
            if prediction_id in predictions_by_id:
                raise SFTEvaluationError(
                    f"Existing predictions contain duplicate ID {prediction_id!r}."
                )
            record = records_by_id[prediction_id]
            if str(prediction.get("source", "")) != str(record["source"]):
                raise SFTEvaluationError(
                    f"Existing prediction source does not match evaluation row {prediction_id}."
                )
            prediction_record: dict[str, Any] = {
                "id": prediction_id,
                "sentence_id": prediction_id,
                "source": str(record["source"]),
                "reference": str(record["target"]),
                "prediction": str(prediction.get("prediction", "")),
            }
            if sentence_boundary_count is not None:
                saved_count = prediction.get("sentence_boundary_count")
                if saved_count != sentence_boundary_count:
                    raise SFTEvaluationError(
                        f"Existing prediction {prediction_id!r} does not use the requested "
                        f"{sentence_boundary_count}-sentence boundary contract."
                    )
                raw_prediction = str(
                    prediction.get("raw_prediction") or prediction.get("prediction", "")
                )
                reparsed_prediction, boundary_valid, boundary_error = (
                    _extract_fixed_sentence_units(
                        raw_prediction, sentence_boundary_count
                    )
                )
                prediction_record["prediction"] = reparsed_prediction
                prediction_record.update(
                    {
                        "raw_prediction": raw_prediction,
                        "sentence_boundary_count": sentence_boundary_count,
                        "sentence_boundary_contract_valid": boundary_valid,
                        "sentence_boundary_contract_error": boundary_error,
                    }
                )
            predictions_by_id[prediction_id] = prediction_record
        logger.info(
            "Resuming SFT %s/%s %s from %d/%d predictions",
            dataset_key,
            model_key,
            split,
            len(predictions_by_id),
            len(records),
        )

    pending_records = [record for record in records if str(record["id"]) not in predictions_by_id]
    prompt_spec = (
        _build_boundary_prompt_spec(sentence_boundary_count)
        if sentence_boundary_count is not None
        else settings["chat_format"].get("prompt_spec")
    )
    inference_backend = model_entry.backend
    loaded_model = None
    tokenizer = None
    transformer_model = None
    if pending_records and model_entry.backend == "vllm":
        decoding = dict(settings["evaluation"])
        decoding["max_new_tokens"] = int(max_new_tokens or decoding["max_new_tokens"])
        loaded_model = load_inference_model(
            model_key,
            str(models_config_path),
            decoding,
            adapter_path=adapter_dir,
        )
        tokenizer = loaded_model.tokenizer
        if tokenizer is None:
            raise SFTEvaluationError("vLLM did not expose a tokenizer for prompt rendering.")
        batch_size = int(settings["evaluation"].get("vllm_batch_size", 128))
    elif pending_records:
        tokenizer = load_sft_tokenizer(model_entry)
        tokenizer.padding_side = "left"
        transformer_model = load_sft_base_model(model_entry, settings, training=False)
        try:
            from peft import PeftModel
        except ImportError as exc:  # pragma: no cover
            raise SFTEvaluationError("peft is required for adapter evaluation.") from exc
        transformer_model = PeftModel.from_pretrained(transformer_model, adapter_dir)
        transformer_model.eval()
        transformer_model.generation_config.do_sample = bool(
            settings["evaluation"]["do_sample"]
        )
        transformer_model.generation_config.num_beams = int(
            settings["evaluation"]["num_beams"]
        )
        transformer_model.generation_config.temperature = settings["evaluation"].get(
            "temperature"
        )
        transformer_model.generation_config.top_p = settings["evaluation"].get("top_p")
        transformer_model.generation_config.pad_token_id = tokenizer.pad_token_id
        batch_size = int(settings["evaluation"]["batch_size"])
    else:
        batch_size = int(settings["evaluation"].get("vllm_batch_size", 128))

    try:
        for start_index in range(0, len(pending_records), batch_size):
            batch = pending_records[start_index : start_index + batch_size]
            prompt_sources = [
                _tag_fixed_sentence_units(str(record["source"]), sentence_boundary_count)
                if sentence_boundary_count is not None
                else str(record["source"])
                for record in batch
            ]
            prompts = [
                build_prompt_text(
                    model_entry,
                    prompt_source,
                    target_lang,
                    tokenizer=tokenizer,
                    enable_thinking=bool(
                        settings["chat_format"].get("enable_thinking", False)
                    ),
                    prompt_spec=prompt_spec,
                )
                for record, prompt_source in zip(batch, prompt_sources, strict=True)
            ]
            if loaded_model is not None:
                decoding = dict(settings["evaluation"])
                decoding["max_new_tokens"] = int(
                    max_new_tokens or decoding["max_new_tokens"]
                )
                outputs = generate_batch(loaded_model, prompts, decoding)
            elif transformer_model is not None:
                outputs = _generate(
                    transformer_model,
                    tokenizer,
                    prompts,
                    settings,
                    max_input_length=max_input_length,
                    max_new_tokens=max_new_tokens,
                )
            else:  # pragma: no cover - pending records guarantee one loaded backend
                raise SFTEvaluationError("No inference backend was loaded.")
            for record, output in zip(batch, outputs, strict=True):
                prediction_id = str(record["id"])
                prediction = output
                boundary_valid = None
                boundary_error = None
                if sentence_boundary_count is not None:
                    prediction, boundary_valid, boundary_error = _extract_fixed_sentence_units(
                        output,
                        sentence_boundary_count,
                    )
                prediction_record: dict[str, Any] = {
                    "id": prediction_id,
                    "sentence_id": prediction_id,
                    "source": str(record["source"]),
                    "reference": str(record["target"]),
                    "prediction": prediction,
                }
                if sentence_boundary_count is not None:
                    prediction_record.update(
                        {
                            "raw_prediction": output,
                            "sentence_boundary_count": sentence_boundary_count,
                            "sentence_boundary_contract_valid": boundary_valid,
                            "sentence_boundary_contract_error": boundary_error,
                        }
                    )
                predictions_by_id[prediction_id] = prediction_record
            ordered_partial = [
                predictions_by_id[str(record["id"])]
                for record in records
                if str(record["id"]) in predictions_by_id
            ]
            save_dataframe_jsonl(pd.DataFrame(ordered_partial), predictions_path)
            logger.info(
                "SFT %s/%s %s via %s: %d/%d",
                dataset_key,
                model_key,
                split,
                inference_backend,
                len(predictions_by_id),
                len(records),
            )
    finally:
        if loaded_model is not None:
            release_inference_model(loaded_model)

    if len(predictions_by_id) != len(records):
        raise SFTEvaluationError(
            f"Evaluation produced {len(predictions_by_id)}/{len(records)} predictions."
        )
    predictions = [predictions_by_id[str(record["id"])] for record in records]
    prediction_frame = pd.DataFrame(predictions)
    save_dataframe_jsonl(prediction_frame, predictions_path)
    save_json(compute_automatic_metrics(prediction_frame), metrics_path)
    decoding = dict(settings["evaluation"])
    decoding["max_input_length"] = int(
        max_input_length or settings["chat_format"]["max_seq_length"]
    )
    decoding["max_new_tokens"] = int(max_new_tokens or decoding["max_new_tokens"])
    save_json(
        {
            "experiment_key": experiment_key,
            "stage": "post_training",
            "dataset_key": dataset_key,
            "result_dataset_key": result_dataset_key,
            "model_key": model_key,
            "adapter_dir": str(adapter_dir),
            "evaluation_split": split,
            "in_domain_test_read": evaluation_file is None and split == "test",
            "explicit_evaluation_file": str(eval_path) if evaluation_file is not None else None,
            "input_path": str(eval_path),
            "examples": len(predictions),
            "limit": limit,
            "resume": resume,
            "inference_backend": inference_backend,
            "tensor_parallel_size": model_entry.tensor_parallel_size,
            "decoding": decoding,
            "sentence_boundary_contract": {
                "enabled": sentence_boundary_count is not None,
                "expected_count": sentence_boundary_count,
                "valid_outputs": (
                    sum(
                        bool(item.get("sentence_boundary_contract_valid"))
                        for item in predictions
                    )
                    if sentence_boundary_count is not None
                    else None
                ),
                "prompt_spec": prompt_spec if sentence_boundary_count is not None else None,
            },
        },
        metadata_path,
    )
    return {
        "predictions": str(predictions_path),
        "automatic_metrics": str(metrics_path),
        "metadata": str(metadata_path),
    }



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an SFT adapter on the in-domain test set.")
    parser.add_argument("--dataset", required=True, choices=("en_eu", "en_ca"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--adapter-root", default=None)
    parser.add_argument("--results-root", default=None)
    parser.add_argument("--split", default="test", help="Canonical split label, or a label for --evaluation-file.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--experiment-key", default="p4_sft")
    parser.add_argument("--evaluation-file", default=None, help="Immutable JSONL with id/source/target fields.")
    parser.add_argument("--output-dataset-key", default=None, help="Dataset label used only for result paths.")
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=None,
        help="Optional evaluation-only tokenizer limit; training configuration is unchanged.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Optional evaluation-only generation cap.",
    )
    parser.add_argument(
        "--sentence-boundary-count",
        type=int,
        default=None,
        help="Require this many tagged translation units and preserve them as blank-line-separated segments.",
    )
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    paths = evaluate_sft_split(
        args.dataset, args.model, args.config, args.models_config, args.data_dir,
        args.adapter_root, args.results_root, args.split, args.limit, args.experiment_key,
        args.evaluation_file, args.output_dataset_key, args.max_input_length,
        args.max_new_tokens, args.resume,
        sentence_boundary_count=args.sentence_boundary_count,
    )
    logger.info("SFT %s evaluation complete: %s", args.split, paths)


if __name__ == "__main__":
    main()
