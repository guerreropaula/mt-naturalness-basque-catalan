"""Registry-driven model loading and batched text generation."""

from __future__ import annotations

import gc
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from src.utils.config import ModelEntry, get_model_entry
from src.utils.errors import PipelineError
from src.utils.hf_auth import get_hf_token
from src.utils.model_adapters import build_prompt_text


@dataclass
class LoadedModel:
    model_entry: ModelEntry
    backend: str
    model: Any
    tokenizer: Any | None = None
    dtype: str | None = None
    device_map: str | None = None
    quantization: str | None = None
    lora_request: Any | None = None
    runtime_metadata: dict[str, Any] = field(default_factory=dict)


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    try:  # pragma: no cover - torch is environment-dependent in tests
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        return


def _resolve_torch_dtype(dtype_name: str) -> Any | None:
    try:  # pragma: no cover - torch import depends on runtime
        import torch
    except Exception:
        return None
    dtype_lookup = {
        "auto": None,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return dtype_lookup.get(dtype_name)


def _resolve_vllm_dtype(dtype_name: str) -> str:
    dtype_lookup = {
        "auto": "auto",
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }
    return dtype_lookup.get(dtype_name, dtype_name)


def _build_quantization_config(quantization: str | None) -> Any | None:
    if quantization != "4bit":
        return None
    try:  # pragma: no cover - bitsandbytes may be unavailable in tests/runtime
        from transformers import BitsAndBytesConfig
    except Exception as exc:  # pragma: no cover
        raise PipelineError(
            "4-bit quantization was requested but BitsAndBytesConfig is unavailable"
        ) from exc
    return BitsAndBytesConfig(load_in_4bit=True)


def load_inference_model(
    model_key: str,
    models_config_path: str = "configs/models.yaml",
    generation_config: dict[str, Any] | None = None,
    adapter_path: str | os.PathLike[str] | None = None,
) -> LoadedModel:
    """Load a registry model with its configured inference backend."""
    model_entry = get_model_entry(model_key, models_config_path)
    if not model_entry.is_id_confirmed:
        raise PipelineError(
            f"Model '{model_key}' has hf_id=TO_BE_CONFIRMED and cannot be loaded yet."
        )
    hf_token = get_hf_token()
    if model_entry.gated and not hf_token:
        raise PipelineError(
            f"Model '{model_key}' is gated and requires HF_TOKEN in the environment."
        )

    generation_config = generation_config or {}
    seed = int(generation_config.get("seed", 42))
    if model_entry.backend == "transformers":
        if adapter_path is not None:
            raise PipelineError(
                "Adapter-aware shared inference is only implemented for the vLLM backend."
            )
        _set_random_seed(seed)
        return _load_transformers_model(model_entry)
    if model_entry.backend == "vllm":
        return _load_vllm_model(model_entry, seed=seed, adapter_path=adapter_path)
    raise PipelineError(
        f"Backend '{model_entry.backend}' is not supported for local baseline inference."
    )


def _load_transformers_model(model_entry: ModelEntry) -> LoadedModel:
    if model_entry.family == "gemma3":
        return _load_gemma3_model(model_entry)
    try:  # pragma: no cover - actual imports are mocked in tests
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover
        raise PipelineError("transformers is required for the transformers backend") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_entry.hf_id,
        token=get_hf_token(),
        trust_remote_code=True,
        padding_side="left",
    )
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only generation needs left padding. Some tokenizer versions ignore
    # the loader argument.
    tokenizer.padding_side = "left"

    load_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "token": get_hf_token(),
        "trust_remote_code": True,
    }
    torch_dtype = _resolve_torch_dtype(model_entry.dtype)
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype
    quantization_config = _build_quantization_config(model_entry.quantization)
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(model_entry.hf_id, **load_kwargs)

    # Clear checkpoint sampling defaults before greedy decoding.
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    return LoadedModel(
        model_entry=model_entry,
        backend="transformers",
        model=model,
        tokenizer=tokenizer,
        dtype=model_entry.dtype,
        device_map="auto",
        quantization=model_entry.quantization,
        runtime_metadata={
            "hf_id": model_entry.hf_id,
            "family": model_entry.family,
            "backend": model_entry.backend,
        },
    )


def _load_gemma3_model(model_entry: ModelEntry) -> LoadedModel:
    """Load Gemma 3 through its documented processor/model API."""
    try:  # pragma: no cover - actual imports are mocked in tests
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration
    except Exception as exc:  # pragma: no cover
        raise PipelineError(
            "Gemma 3 requires AutoProcessor and Gemma3ForConditionalGeneration"
        ) from exc

    load_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "token": get_hf_token(),
    }
    torch_dtype = _resolve_torch_dtype(model_entry.dtype)
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype
    quantization_config = _build_quantization_config(model_entry.quantization)
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config

    processor = AutoProcessor.from_pretrained(model_entry.hf_id, token=get_hf_token())
    model = Gemma3ForConditionalGeneration.from_pretrained(model_entry.hf_id, **load_kwargs).eval()
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    return LoadedModel(
        model_entry=model_entry,
        backend="transformers",
        model=model,
        tokenizer=processor,
        dtype=model_entry.dtype,
        device_map="auto",
        quantization=model_entry.quantization,
        runtime_metadata={
            "hf_id": model_entry.hf_id,
            "family": model_entry.family,
            "backend": model_entry.backend,
            "inference_api": "Gemma3ForConditionalGeneration+AutoProcessor",
        },
    )


def _load_vllm_model(
    model_entry: ModelEntry,
    seed: int,
    adapter_path: str | os.PathLike[str] | None = None,
) -> LoadedModel:
    # Select spawn before CUDA initialization because vLLM starts a CUDA worker process.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    try:  # pragma: no cover - optional dependency
        from vllm import LLM
    except Exception as exc:  # pragma: no cover
        raise PipelineError("vLLM backend requested but vllm is not installed") from exc

    llm_kwargs: dict[str, Any] = {}
    lora_request = None
    resolved_adapter_path: Path | None = None
    if adapter_path is not None:
        resolved_adapter_path = Path(adapter_path).resolve()
        config_path = resolved_adapter_path / "adapter_config.json"
        if not config_path.is_file():
            raise PipelineError(f"LoRA adapter configuration not found: {config_path}")
        try:
            adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
            lora_rank = int(adapter_config["r"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise PipelineError(f"Invalid LoRA adapter configuration: {config_path}") from exc
        if lora_rank <= 0:
            raise PipelineError(f"LoRA rank must be positive in {config_path}")
        try:
            from vllm.lora.request import LoRARequest
        except Exception as exc:  # pragma: no cover - optional dependency
            raise PipelineError("The installed vLLM build does not provide LoRA inference") from exc
        llm_kwargs.update(
            enable_lora=True,
            max_lora_rank=lora_rank,
            max_loras=1,
            max_cpu_loras=1,
            fully_sharded_loras=model_entry.tensor_parallel_size > 1,
        )
        lora_request = LoRARequest(
            "thesis_adapter",
            1,
            str(resolved_adapter_path),
            base_model_name=model_entry.hf_id,
        )

    memory_utilization = os.environ.get("VLLM_GPU_MEMORY_UTILIZATION")
    if memory_utilization:
        try:
            value = float(memory_utilization)
        except ValueError as exc:
            raise PipelineError(
                "VLLM_GPU_MEMORY_UTILIZATION must be a floating-point value in (0, 1)."
            ) from exc
        if not 0.0 < value < 1.0:
            raise PipelineError("VLLM_GPU_MEMORY_UTILIZATION must be in (0, 1).")
        llm_kwargs["gpu_memory_utilization"] = value

    enforce_eager = os.environ.get("VLLM_ENFORCE_EAGER")
    if enforce_eager is not None:
        normalized = enforce_eager.strip().lower()
        if normalized not in {"0", "1", "false", "true"}:
            raise PipelineError("VLLM_ENFORCE_EAGER must be one of: 0, 1, false, true.")
        llm_kwargs["enforce_eager"] = normalized in {"1", "true"}

    llm = LLM(
        model=model_entry.hf_id,
        dtype=_resolve_vllm_dtype(model_entry.dtype),
        tensor_parallel_size=model_entry.tensor_parallel_size,
        disable_custom_all_reduce=model_entry.tensor_parallel_size > 1,
        max_model_len=model_entry.max_context_length,
        trust_remote_code=True,
        seed=seed,
        **llm_kwargs,
    )
    tokenizer = llm.get_tokenizer() if hasattr(llm, "get_tokenizer") else None
    return LoadedModel(
        model_entry=model_entry,
        backend="vllm",
        model=llm,
        tokenizer=tokenizer,
        dtype=model_entry.dtype,
        device_map="auto",
        quantization=model_entry.quantization,
        lora_request=lora_request,
        runtime_metadata={
            "hf_id": model_entry.hf_id,
            "family": model_entry.family,
            "backend": model_entry.backend,
            "tensor_parallel_size": model_entry.tensor_parallel_size,
            "max_context_length": model_entry.max_context_length,
            "adapter_path": (
                str(resolved_adapter_path) if resolved_adapter_path is not None else None
            ),
        },
    )


def release_inference_model(loaded_model: LoadedModel) -> None:
    """Release model and CUDA memory before another model is loaded."""
    if loaded_model.backend == "vllm":
        engine = getattr(loaded_model.model, "llm_engine", None)
        client = getattr(engine, "engine_core", None)
        shutdown = getattr(client, "shutdown", None)
        if callable(shutdown):
            shutdown()
    loaded_model.model = None
    loaded_model.tokenizer = None
    gc.collect()
    try:  # pragma: no cover - CUDA is runtime-dependent
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def render_translation_prompts(
    loaded_model: LoadedModel,
    source_texts: list[str],
    target_lang: str,
    enable_thinking: bool,
    prompt_spec: Mapping[str, Any] | None = None,
    extra_template_contexts: list[Mapping[str, Any] | None] | None = None,
) -> list[str]:
    """Render translation prompts for a batch."""
    if extra_template_contexts is not None and len(extra_template_contexts) != len(source_texts):
        raise PipelineError("extra_template_contexts must align one-to-one with source_texts")
    return [
        build_prompt_text(
            loaded_model.model_entry,
            source_text=source_text,
            target_lang=target_lang,
            tokenizer=loaded_model.tokenizer,
            enable_thinking=enable_thinking,
            prompt_spec=prompt_spec,
            extra_template_context=(
                None if extra_template_contexts is None else extra_template_contexts[index]
            ),
        )
        for index, source_text in enumerate(source_texts)
    ]


def generate_batch(
    loaded_model: LoadedModel,
    prompts: list[str],
    generation_config: dict[str, Any],
) -> list[str]:
    """Generate one translation for each prompt."""
    start_time = time.perf_counter()
    try:
        if loaded_model.backend == "transformers":
            outputs = _generate_batch_transformers(loaded_model, prompts, generation_config)
        elif loaded_model.backend == "vllm":
            outputs = _generate_batch_vllm(loaded_model, prompts, generation_config)
        else:
            raise PipelineError(f"Unsupported backend for generation: {loaded_model.backend}")
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise PipelineError(
                "Generation failed due to out-of-memory. Reduce batch_size or max_new_tokens."
            ) from exc
        raise
    elapsed = time.perf_counter() - start_time
    loaded_model.runtime_metadata["last_batch_runtime_seconds"] = elapsed
    return outputs


def _generate_batch_transformers(
    loaded_model: LoadedModel,
    prompts: list[str],
    generation_config: dict[str, Any],
) -> list[str]:
    tokenizer = loaded_model.tokenizer
    if tokenizer is None:
        raise PipelineError("Transformers backend requires a tokenizer")

    if loaded_model.model_entry.family == "gemma3":
        batch = tokenizer(text=prompts, return_tensors="pt", padding=True, truncation=True)
    else:
        batch = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            # Pass left padding explicitly because some tokenizer versions ignore
            # the instance setting.
            padding_side="left",
        )

    model_inputs = dict(batch)
    if hasattr(loaded_model.model, "device"):
        try:
            model_inputs = {
                key: value.to(loaded_model.model.device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }
        except Exception:
            model_inputs = dict(batch)

    input_ids = batch["input_ids"]
    # With left padding, generated tokens begin after the shared padded prompt length.
    prompt_length = (
        int(input_ids.shape[1])
        if hasattr(input_ids, "shape") and len(input_ids.shape) > 1
        else len(input_ids[0])
    )

    generate_kwargs = {
        "do_sample": bool(generation_config.get("do_sample", False)),
        "num_beams": int(generation_config.get("num_beams", 1)),
        "max_new_tokens": int(generation_config["max_new_tokens"]),
        "temperature": generation_config.get("temperature"),
        "top_p": generation_config.get("top_p"),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
    }
    generate_kwargs = {key: value for key, value in generate_kwargs.items() if value is not None}

    output_ids = loaded_model.model.generate(**model_inputs, **generate_kwargs)
    predictions: list[str] = []
    for sequence in output_ids:
        new_tokens = sequence[prompt_length:]
        predictions.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return predictions


def _generate_batch_vllm(
    loaded_model: LoadedModel,
    prompts: list[str],
    generation_config: dict[str, Any],
) -> list[str]:
    from vllm import SamplingParams  # pragma: no cover - optional dependency

    sampling_params = SamplingParams(
        temperature=0.0
        if generation_config.get("temperature") is None
        else generation_config["temperature"],
        top_p=1.0 if generation_config.get("top_p") is None else generation_config["top_p"],
        max_tokens=int(generation_config["max_new_tokens"]),
    )
    generate_kwargs: dict[str, Any] = {}
    if loaded_model.lora_request is not None:
        generate_kwargs["lora_request"] = loaded_model.lora_request
    outputs = loaded_model.model.generate(prompts, sampling_params, **generate_kwargs)
    return [output.outputs[0].text.strip() for output in outputs]
