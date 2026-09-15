"""P5 GRPO training from a P4 SFT LoRA adapter."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from datasets import Dataset
from transformers import TrainerCallback

from src.grpo.data import build_grpo_records, load_grpo_records
from src.grpo.health import HealthThresholds, assess_translation_health
from src.grpo.reward import build_translation_grpo_reward
from src.sft.train import (
    _torch_dtype,
    align_fsdp_qlora_parameter_dtypes,
    distributed_rank,
    distributed_world_size,
    load_sft_base_model,
    load_sft_tokenizer,
    prepare_model_for_distributed_qlora_training,
    resolve_sft_settings,
)
from src.utils.config import get_model_entry, load_grpo_config, load_sft_config
from src.utils.io import save_json

logger = logging.getLogger(__name__)


class GRPOTrainingError(RuntimeError):
    """Raised when P5 GRPO cannot be configured or launched safely."""


def _merge_mapping(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_mapping(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def resolve_grpo_settings(config: Mapping[str, Any], model_key: str) -> dict[str, Any]:
    overrides = config.get("model_overrides", {})
    override = overrides.get(model_key, {})
    if not isinstance(override, Mapping):
        raise GRPOTrainingError(f"grpo.model_overrides.{model_key} must be a mapping")
    return _merge_mapping(config, override)


def _prepare_trl_vllm_import() -> None:
    """Bridge an import-only vLLM API rename while P5 uses Transformers generation.

    TRL 0.19 imports the removed GuidedDecodingParams symbol whenever vLLM is
    installed, even with use_vllm=False. The placeholder is never instantiated
    by this protocol; it gives a clear error if a future configuration enables
    the incompatible guided-decoding integration.
    """
    try:  # pragma: no cover - depends on optional vLLM installation
        import vllm.sampling_params as sampling_params
    except ImportError:
        return
    if hasattr(sampling_params, "GuidedDecodingParams"):
        return

    class _UnsupportedGuidedDecodingParams:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise GRPOTrainingError(
                "P5 disables vLLM generation. Guided decoding is incompatible with the installed "
                "vLLM API; keep training.use_vllm=false."
            )

    sampling_params.GuidedDecodingParams = _UnsupportedGuidedDecodingParams


@contextmanager
def _temporary_generation_cache(*configs: Any):
    """Enable KV caching only while sampling, then restore training settings."""
    previous: list[tuple[Any, Any]] = []
    for config in configs:
        if config is None:
            continue
        previous.append((config, getattr(config, "use_cache", None)))
        config.use_cache = True
    try:
        yield
    finally:
        for config, use_cache in previous:
            config.use_cache = use_cache


def _load_trl(
    vllm_model_kwargs: Mapping[str, Any] | None = None,
    vllm_lora_adapter: Path | None = None,
    vllm_lora_sync_dir: Path | None = None,
) -> tuple[Any, Any]:
    import trl.import_utils as trl_import_utils

    # Import GRPOTrainer without eagerly importing vLLM. The real vLLM module
    # is loaded only after Trainer has FSDP-sharded the 70B policy.
    original_vllm_available = trl_import_utils._vllm_available
    trl_import_utils._vllm_available = False
    try:
        from trl import GRPOConfig, GRPOTrainer
    except (ImportError, RuntimeError) as exc:  # pragma: no cover - optional runtime
        raise GRPOTrainingError(
            "P5 GRPO requires a compatible TRL installation. Install the project GRPO extra."
        ) from exc
    finally:
        trl_import_utils._vllm_available = original_vllm_available

    import trl.trainer.grpo_trainer as grpo_trainer_module

    class LazyLLM:
        def __new__(cls, *args: Any, **kwargs: Any) -> Any:
            from vllm import LLM

            return LLM(*args, **kwargs)

    class LazySamplingParams:
        def __new__(cls, *args: Any, **kwargs: Any) -> Any:
            from vllm import SamplingParams

            guided_decoding = kwargs.pop("guided_decoding", None)
            if guided_decoding is not None:
                raise GRPOTrainingError("Guided decoding is not supported by the installed vLLM API.")
            return SamplingParams(*args, **kwargs)

    class UnsupportedGuidedDecodingParams:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise GRPOTrainingError("Guided decoding is not supported by the installed vLLM API.")

    grpo_trainer_module.is_vllm_available = lambda: True
    grpo_trainer_module.LLM = LazyLLM
    grpo_trainer_module.SamplingParams = LazySamplingParams
    grpo_trainer_module.GuidedDecodingParams = UnsupportedGuidedDecodingParams

    class TranslationGRPOTrainer(GRPOTrainer):
        """Generate rollouts in eval mode, then restore train mode for optimization."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if not vllm_model_kwargs:
                super().__init__(*args, **kwargs)
                return

            import trl.trainer.grpo_trainer as grpo_trainer_module

            original_llm = grpo_trainer_module.LLM

            def configured_llm(*llm_args: Any, **llm_kwargs: Any) -> Any:
                llm_kwargs.update(dict(vllm_model_kwargs))
                return original_llm(*llm_args, **llm_kwargs)

            grpo_trainer_module.LLM = configured_llm
            try:
                super().__init__(*args, **kwargs)
            finally:
                grpo_trainer_module.LLM = original_llm
            if self.use_vllm and bool(vllm_model_kwargs.get("enable_sleep_mode", False)):
                self.llm.sleep(level=1, mode="abort")
                self._vllm_is_sleeping = True

        def _generate_and_score_completions(self, inputs: Any) -> Any:
            if not hasattr(self, "_fsdp_rollout_unit_count"):
                self._fsdp_rollout_unit_count = _validate_fsdp_rollout_wrapping(self)
            was_training = bool(self.model.training)
            self.model.eval()
            try:
                if self.use_vllm:
                    if getattr(self, "_vllm_is_sleeping", False):
                        self.llm.wake_up()
                        self._vllm_is_sleeping = False
                    try:
                        return super()._generate_and_score_completions(inputs)
                    finally:
                        self.llm.sleep(level=1, mode="abort")
                        self._vllm_is_sleeping = True
                with _temporary_generation_cache(self.model.config, self.generation_config):
                    return super()._generate_and_score_completions(inputs)
            finally:
                if was_training:
                    self.model.train()

        def _move_model_to_vllm(self) -> None:
            if vllm_lora_adapter is None:
                return super()._move_model_to_vllm()
            _sync_lora_adapter_to_vllm(
                self,
                initial_adapter=vllm_lora_adapter,
                sync_root=vllm_lora_sync_dir,
            )

    return GRPOConfig, TranslationGRPOTrainer


def _normalise_peft_lora_key(name: str) -> str:
    for wrapper in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module."):
        name = name.replace(wrapper, "")
    return name.replace(".default.", ".")


def _save_fsdp_lora_adapter(model: Any, adapter_dir: Path, source_adapter: Path) -> None:
    """Gather only LoRA tensors and write a vLLM-compatible PEFT adapter."""
    from safetensors.torch import save_file
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    tensors: dict[str, torch.Tensor] = {}
    visited: set[str] = set()

    def visit(module: Any, prefix: str = "") -> None:
        for child_name, child in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            visit(child, child_prefix)
        if not isinstance(module, FSDP):
            return
        with FSDP.summon_full_params(module, recurse=False, writeback=False):
            for param_name, parameter in module.named_parameters():
                full_name = f"{prefix}.{param_name}" if prefix else param_name
                full_name = _normalise_peft_lora_key(full_name)
                if "lora_" not in full_name or full_name in visited:
                    continue
                visited.add(full_name)
                if distributed_rank() == 0:
                    tensors[full_name] = (
                        parameter.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
                    )

    visit(model)
    if distributed_rank() == 0:
        if not tensors:
            raise GRPOTrainingError("FSDP adapter synchronization found no LoRA tensors.")
        adapter_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_adapter / "adapter_config.json", adapter_dir / "adapter_config.json")
        save_file(tensors, adapter_dir / "adapter_model.safetensors")
    torch.distributed.barrier()


def _sync_lora_adapter_to_vllm(
    trainer: Any, initial_adapter: Path, sync_root: Path | None
) -> None:
    """Generate with the active LoRA and refresh it after each policy update."""
    from vllm.lora.request import LoRARequest

    step = int(trainer.state.global_step)
    if step == 0:
        adapter_path = initial_adapter
    else:
        if sync_root is None:
            raise GRPOTrainingError("A vLLM LoRA synchronization directory is required.")
        adapter_path = sync_root / f"step_{step:06d}"
        _save_fsdp_lora_adapter(trainer.model, adapter_path, initial_adapter)

    old_id = getattr(trainer, "_active_vllm_lora_id", None)
    if old_id is not None:
        trainer.llm.llm_engine.remove_lora(int(old_id))
    adapter_config = json.loads(
        (initial_adapter / "adapter_config.json").read_text(encoding="utf-8")
    )
    request = LoRARequest(
        f"p5_step_{step}",
        step + 1,
        str(adapter_path.resolve()),
        base_model_name=str(adapter_config.get("base_model_name_or_path") or ""),
    )
    if not hasattr(trainer, "_vllm_generate_without_lora"):
        trainer._vllm_generate_without_lora = trainer.llm.generate
    generate = trainer._vllm_generate_without_lora

    def generate_with_lora(*args: Any, **kwargs: Any) -> Any:
        kwargs["lora_request"] = request
        return generate(*args, **kwargs)

    trainer.llm.generate = generate_with_lora
    trainer._active_vllm_lora_id = request.lora_int_id
    trainer.llm.reset_prefix_cache()


@contextmanager
def _peft_adapter_loading_compatibility(model: Any):
    """Skip PEFT's unavailable TP hook only for a non-TP FSDP model."""
    from transformers.integrations import tensor_parallel

    if hasattr(tensor_parallel, "EmbeddingParallel"):
        yield
        return
    if any(getattr(module, "_hf_device_mesh", None) is not None for module in model.modules()):
        raise GRPOTrainingError(
            "PEFT requires Transformers to provide EmbeddingParallel for an active "
            "tensor-parallel model. Use compatible PEFT/Transformers versions."
        )

    from peft.utils import save_and_load

    original_hook = save_and_load._maybe_shard_state_dict_for_tp
    save_and_load._maybe_shard_state_dict_for_tp = lambda *_args, **_kwargs: None
    logger.warning(
        "PEFT/Transformers do not share the EmbeddingParallel API; skipping "
        "PEFT tensor-parallel adapter sharding for this FSDP model."
    )
    try:
        yield
    finally:
        save_and_load._maybe_shard_state_dict_for_tp = original_hook


def _validate_parent_adapter_lora(sft_adapter_dir: Path, lora: dict[str, Any]) -> None:
    """Refuse to continue a P4 adapter with a LoRA rank/alpha outside the P5 protocol."""
    config_path = sft_adapter_dir / "adapter_config.json"
    if not config_path.exists():
        raise GRPOTrainingError(f"P4 adapter configuration not found: {config_path}")
    try:
        adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GRPOTrainingError(
            f"P4 adapter configuration is not valid JSON: {config_path}"
        ) from exc
    expected_r = int(lora["r"])
    expected_alpha = int(lora["lora_alpha"])
    actual_r = adapter_config.get("r")
    actual_alpha = adapter_config.get("lora_alpha")
    if actual_r != expected_r or actual_alpha != expected_alpha:
        raise GRPOTrainingError(
            "P5 requires its P4 parent adapter to use LoRA r={} and alpha={}; found r={} and alpha={} "
            "in {}. Retrain P4 with configs/sft.yaml before launching P5.".format(
                expected_r, expected_alpha, actual_r, actual_alpha, config_path
            )
        )


def _validate_trainable_lora_parameters(model: Any) -> int:
    """Prove that the loaded P4 adapter exposes trainable LoRA weights."""
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if bool(parameter.requires_grad)
    ]
    lora_parameters = [
        (name, parameter) for name, parameter in trainable if "lora_" in name.lower()
    ]
    if not lora_parameters:
        raise GRPOTrainingError(
            "P5 loaded no trainable LoRA parameters. Refusing to run GRPO without an adapter."
        )
    parameter_count = sum(int(parameter.numel()) for _, parameter in lora_parameters)
    if parameter_count <= 0:  # pragma: no cover - a tensor cannot have negative size
        raise GRPOTrainingError("P5 LoRA adapter has no trainable parameters.")
    logger.info(
        "P5 LoRA trainable parameters: %d tensors, %d parameters",
        len(lora_parameters),
        parameter_count,
    )
    return parameter_count


def _validate_fsdp_rollout_wrapping(trainer: Any) -> int:
    """Require nested FSDP units before Transformers-based GRPO rollouts.

    TRL temporarily materialises only the outer FSDP unit during generation.
    That is memory-safe for a 70B model only when decoder layers are separate
    child FSDP units and gather their parameters one layer at a time.
    """
    if not bool(trainer.is_fsdp_enabled):
        return 0

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    units = [module for module in trainer.model_wrapped.modules() if isinstance(module, FSDP)]
    if len(units) < 2:
        raise GRPOTrainingError(
            "FSDP created only one root unit. GRPO rollout generation would gather "
            "the complete model on every GPU. Configure "
            "fsdp_transformer_layer_cls_to_wrap for the model's decoder layer."
        )
    logger.info(
        "P5 FSDP rollout layout: %d units (%d nested decoder units)",
        len(units),
        len(units) - 1,
    )
    return len(units)


@dataclass(frozen=True)
class GRPOBatchPlan:
    """TRL batch sizes expressed in completions and distinct source prompts."""

    world_size: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    num_generations: int
    generation_batch_size: int
    effective_completion_batch_size: int
    effective_prompt_batch_size: int
    prompts_per_generation_batch: int


def resolve_grpo_batch_plan(training: dict[str, Any], world_size: int = 1) -> GRPOBatchPlan:
    """Resolve TRL's distributed completion batches into source-prompt batches."""
    if world_size <= 0:
        raise GRPOTrainingError("world_size must be positive.")
    per_device = int(training["per_device_train_batch_size"])
    accumulation = int(training["gradient_accumulation_steps"])
    generations = int(training["num_generations"])
    generation_batch = int(training["generation_batch_size"])
    global_micro_batch = per_device * world_size
    effective_completions = global_micro_batch * accumulation
    if effective_completions % generations:
        raise GRPOTrainingError(
            f"Effective completion batch {effective_completions} is not divisible by G={generations}."
        )
    if generation_batch % generations:
        raise GRPOTrainingError(
            f"Generation batch {generation_batch} is not divisible by G={generations}."
        )
    if generation_batch % global_micro_batch:
        raise GRPOTrainingError(
            f"Generation batch {generation_batch} is not divisible by distributed "
            f"micro-batch {global_micro_batch}."
        )
    effective_prompts = effective_completions // generations
    configured_prompts = int(training["effective_prompt_batch_size"])
    if effective_prompts != configured_prompts:
        raise GRPOTrainingError(
            f"Effective prompt batch is {effective_prompts}, not configured {configured_prompts}."
        )
    return GRPOBatchPlan(
        world_size=world_size,
        per_device_batch_size=per_device,
        gradient_accumulation_steps=accumulation,
        num_generations=generations,
        generation_batch_size=generation_batch,
        effective_completion_batch_size=effective_completions,
        effective_prompt_batch_size=effective_prompts,
        prompts_per_generation_batch=generation_batch // generations,
    )


class GRPOHealthCallback(TrainerCallback):
    """Abort training when a fixed greedy development sample clearly collapses."""

    def __init__(
        self,
        tokenizer: Any,
        records: list[dict[str, str]],
        health_config: dict[str, Any],
        output_dir: Path,
        max_prompt_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.records = records
        self.health_config = health_config
        self.output_dir = output_dir
        self.max_prompt_length = int(max_prompt_length)
        self.thresholds = HealthThresholds.from_mapping(health_config)
        self.last_checked_step = -1

    def _generate(self, model: Any) -> tuple[list[str], list[list[int]], list[bool]]:
        predictions: list[str] = []
        completion_ids: list[list[int]] = []
        terminated: list[bool] = []
        batch_size = int(self.health_config["batch_size"])
        eos_token_id = int(self.tokenizer.eos_token_id)
        pad_token_id = int(self.tokenizer.pad_token_id)
        was_training = bool(model.training)
        model.eval()
        try:
            for start in range(0, len(self.records), batch_size):
                batch = self.records[start : start + batch_size]
                encoded = self.tokenizer(
                    [record["prompt"] for record in batch],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_prompt_length,
                )
                device = model.get_input_embeddings().weight.device
                encoded = {name: value.to(device) for name, value in encoded.items()}
                with torch.inference_mode():
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=int(self.health_config["max_new_tokens"]),
                        do_sample=False,
                        num_beams=1,
                        pad_token_id=pad_token_id,
                        eos_token_id=eos_token_id,
                    )
                continuation = generated[:, encoded["input_ids"].shape[1] :].detach().cpu()
                for row in continuation.tolist():
                    did_terminate = eos_token_id in row
                    if did_terminate:
                        row = row[: row.index(eos_token_id)]
                    clean_ids = [token_id for token_id in row if token_id != pad_token_id]
                    completion_ids.append(clean_ids)
                    terminated.append(did_terminate)
                    predictions.append(
                        self.tokenizer.decode(clean_ids, skip_special_tokens=True).strip()
                    )
        finally:
            if was_training:
                model.train()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return predictions, completion_ids, terminated

    def _check(self, state: Any, model: Any) -> None:
        step = int(state.global_step)
        if step == self.last_checked_step:
            return
        predictions, completion_ids, terminated = self._generate(model)
        report = assess_translation_health(
            predictions,
            [record["reference"] for record in self.records],
            completion_ids,
            terminated,
            self.thresholds,
        )
        report.update({"step": step, "thresholds": dict(self.health_config)})
        if distributed_rank() == 0:
            save_json(report, self.output_dir / "health" / f"step_{step:06d}.json")
        self.last_checked_step = step
        logger.info(
            "P5 health step=%d passed=%s chrF++=%.4f repeated=%d/%d unterminated=%d/%d",
            step,
            report["passed"],
            report["mean_chrfpp"],
            report["repeated_rows"],
            report["examples"],
            report["unterminated_rows"],
            report["examples"],
        )
        if not report["passed"]:
            raise GRPOTrainingError(
                f"P5 policy-collapse check failed at step {step}: " + "; ".join(report["failures"])
            )

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._check(state, kwargs["model"])

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        step = int(state.global_step)
        interval = int(self.health_config["interval_steps"])
        if step > 0 and (step % interval == 0 or step == int(state.max_steps)):
            self._check(state, kwargs["model"])


def train_grpo(
    dataset_key: str,
    model_key: str,
    ablation: str | None = None,
    config_path: str | Path = "configs/grpo.yaml",
    sft_config_path: str | Path = "configs/sft.yaml",
    models_config_path: str | Path = "configs/models.yaml",
    resume_from_checkpoint: str | None = None,
    max_train_examples: int | None = None,
    adapter_root: str | Path | None = None,
    results_root: str | Path | None = None,
    sft_adapter_root: str | Path | None = None,
    disable_health_check: bool = False,
) -> dict[str, str]:
    """Run P5 from the configured P4 adapter using a selected GRPO ablation."""
    grpo_config = load_grpo_config(config_path)["grpo"]
    grpo_root = resolve_grpo_settings(grpo_config, model_key)
    if dataset_key not in grpo_root["target_languages"]:
        raise GRPOTrainingError(f"Unsupported P5 dataset: {dataset_key}")
    selected_ablation = str(ablation or grpo_root["reward"]["active_ablation"])
    if selected_ablation not in grpo_root["reward"]["ablations"]:
        raise GRPOTrainingError(f"Unknown P5 ablation: {selected_ablation}")

    model_entry = get_model_entry(model_key, models_config_path)
    sft_root = load_sft_config(sft_config_path)["sft"]
    sft_settings = resolve_sft_settings(sft_root, model_key)
    target_lang = str(grpo_root["target_languages"][dataset_key])
    training = grpo_root["training"]
    blocked_reason = training.get("blocked_reason")
    if blocked_reason:
        raise GRPOTrainingError(str(blocked_reason))
    world_size = distributed_world_size()
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        logger.info("Bound distributed rank %d to CUDA device %d", distributed_rank(), local_rank)
    required_world_size = int(training.get("required_world_size", world_size))
    if world_size != required_world_size:
        raise GRPOTrainingError(
            f"{model_key} GRPO requires {required_world_size} distributed processes; "
            f"found {world_size}."
        )
    required_cuda_devices = int(training.get("required_cuda_devices", 0))
    if torch.cuda.device_count() < required_cuda_devices:
        raise GRPOTrainingError(
            f"{model_key} GRPO requires {required_cuda_devices} visible CUDA devices; "
            f"found {torch.cuda.device_count()}."
        )
    reward = build_translation_grpo_reward(grpo_root, dataset_key, ablation=selected_ablation)
    batch_plan = resolve_grpo_batch_plan(training, world_size=world_size)
    if batch_plan.num_generations != reward.group_size:
        raise GRPOTrainingError("training.num_generations must equal reward.group_size.")
    global_eval_batch = int(training["per_device_eval_batch_size"]) * batch_plan.world_size
    if global_eval_batch % batch_plan.num_generations != 0:
        raise GRPOTrainingError(
            "The global evaluation batch must be divisible by training.num_generations."
        )

    tokenizer = load_sft_tokenizer(model_entry)
    tokenizer.padding_side = "left"
    base_model = load_sft_base_model(model_entry, sft_settings, training=True)
    try:
        from peft import PeftModel, prepare_model_for_kbit_training
    except ImportError as exc:  # pragma: no cover - project dependency
        raise GRPOTrainingError("P5 GRPO requires peft.") from exc
    if bool(sft_settings["quantization"].get("load_in_4bit", False)):
        gradient_checkpointing = bool(training["gradient_checkpointing"])
        gradient_checkpointing_kwargs = dict(training.get("gradient_checkpointing_kwargs", {}))
        if world_size > 1:
            base_model = prepare_model_for_distributed_qlora_training(
                base_model,
                gradient_checkpointing,
                gradient_checkpointing_kwargs,
            )
        else:
            base_model = prepare_model_for_kbit_training(
                base_model,
                use_gradient_checkpointing=gradient_checkpointing,
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
            )
    sft_adapter_dir = (
        Path(sft_adapter_root or grpo_root["sft_adapter_roots"][dataset_key])
        / dataset_key
        / model_key
    )
    if not sft_adapter_dir.exists():
        raise GRPOTrainingError(f"P4 SFT adapter not found: {sft_adapter_dir}")
    _validate_parent_adapter_lora(sft_adapter_dir, dict(grpo_root["lora"]))
    native_vllm_lora = (
        sft_adapter_dir if bool(training.get("vllm_native_lora", False)) else None
    )
    vllm_sync_dir = (
        Path(results_root or grpo_root["results_root"])
        / selected_ablation
        / dataset_key
        / model_key
        / "vllm_adapter_sync"
    )
    GRPOConfig, GRPOTrainer = _load_trl(
        training.get("vllm_model_kwargs"),
        vllm_lora_adapter=native_vllm_lora,
        vllm_lora_sync_dir=vllm_sync_dir,
    )
    with _peft_adapter_loading_compatibility(base_model):
        model = PeftModel.from_pretrained(
            base_model,
            sft_adapter_dir,
            is_trainable=True,
            autocast_adapter_dtype=bool(grpo_root["lora"].get("autocast_adapter_dtype", True)),
        )
    if world_size > 1 and bool(sft_settings["quantization"].get("load_in_4bit", False)):
        align_fsdp_qlora_parameter_dtypes(
            model,
            _torch_dtype(bool(sft_settings["training"].get("bf16", True))),
        )
    trainable_lora_parameters = _validate_trainable_lora_parameters(model)
    model_devices = {
        str(device)
        for device in getattr(model, "hf_device_map", {}).values()
        if str(device) not in {"cpu", "disk"}
    }
    required_model_devices = int(training.get("required_model_devices", 0))
    if len(model_devices) < required_model_devices:
        raise GRPOTrainingError(
            f"{model_key} must be sharded across {required_model_devices} CUDA devices; "
            f"the resolved device map uses {sorted(model_devices)}."
        )
    logger.info("P5 policy device map uses: %s", sorted(model_devices))
    model.config.use_cache = False

    prompt_spec = grpo_root["chat_format"].get("prompt_spec")
    data_dir = grpo_root["data_dir"]
    train_records = build_grpo_records(
        load_grpo_records(data_dir, dataset_key, "train"),
        tokenizer,
        model_entry,
        target_lang,
        prompt_spec=prompt_spec,
        enable_thinking=bool(grpo_root["chat_format"].get("enable_thinking", False)),
    )
    if max_train_examples is not None:
        if max_train_examples <= 0:
            raise GRPOTrainingError("max_train_examples must be positive when provided.")
        train_records = train_records[:max_train_examples]
    health_config = dict(training["health_check"])
    if disable_health_check:
        health_config["enabled"] = False
    health_records: list[dict[str, str]] = []
    if bool(health_config["enabled"]):
        health_records = build_grpo_records(
            load_grpo_records(data_dir, dataset_key, "dev")[: int(health_config["examples"])],
            tokenizer,
            model_entry,
            target_lang,
            prompt_spec=prompt_spec,
            enable_thinking=bool(grpo_root["chat_format"].get("enable_thinking", False)),
        )
    evaluation_enabled = str(training["eval_strategy"]).lower() != "no"
    dev_records = []
    if evaluation_enabled:
        dev_records = build_grpo_records(
            load_grpo_records(data_dir, dataset_key, "dev"),
            tokenizer,
            model_entry,
            target_lang,
            prompt_spec=prompt_spec,
            enable_thinking=bool(grpo_root["chat_format"].get("enable_thinking", False)),
        )

    adapter_dir = (
        Path(adapter_root or grpo_root["adapter_root"])
        / selected_ablation
        / dataset_key
        / model_key
    )
    run_dir = (
        Path(results_root or grpo_root["results_root"])
        / selected_ablation
        / dataset_key
        / model_key
    )
    if adapter_dir.exists() and resume_from_checkpoint is None:
        raise FileExistsError(f"P5 adapter already exists: {adapter_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    audit_config = dict(training["completion_audit"])
    if bool(audit_config["enabled"]):
        audit_name = (
            "completion_audit.jsonl"
            if distributed_rank() == 0
            else f"completion_audit.rank_{distributed_rank()}.jsonl"
        )
        reward.configure_audit(
            run_dir / audit_name,
            interval_calls=int(audit_config["interval_reward_calls"]),
            max_groups=int(audit_config["max_groups_per_write"]),
            min_mean_chrfpp=float(audit_config["min_mean_chrfpp"]),
            min_mean_cometkiwi=float(audit_config["min_mean_cometkiwi"]),
        )
    callbacks: list[Any] = []
    if health_records:
        callbacks.append(
            GRPOHealthCallback(
                tokenizer,
                health_records,
                health_config,
                run_dir,
                int(training["max_prompt_length"]),
            )
        )
    generation_kwargs = dict(training["generation_kwargs"])
    if bool(training.get("use_vllm", False)):
        generation_kwargs.pop("do_sample", None)
        generation_kwargs.pop("num_beams", None)
        if "min_new_tokens" in generation_kwargs:
            generation_kwargs["min_tokens"] = generation_kwargs.pop("min_new_tokens")

    trainer_args = GRPOConfig(
        output_dir=str(run_dir / "trainer"),
        run_name=f"p5_grpo_{selected_ablation}_{dataset_key}_{model_key}",
        num_train_epochs=float(training["num_train_epochs"]),
        learning_rate=float(training["learning_rate"]),
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        generation_batch_size=batch_plan.generation_batch_size,
        warmup_ratio=float(training["warmup_ratio"]),
        weight_decay=float(training["weight_decay"]),
        lr_scheduler_type=str(training["lr_scheduler_type"]),
        optim=str(training["optim"]),
        logging_steps=int(training["logging_steps"]),
        eval_strategy=str(training["eval_strategy"]),
        eval_steps=int(training["eval_steps"]),
        save_strategy=str(training["save_strategy"]),
        save_steps=int(training["save_steps"]),
        save_total_limit=int(training["save_total_limit"]),
        save_only_model=bool(training.get("save_only_model", False)),
        disable_dropout=bool(training["disable_dropout"]),
        max_grad_norm=float(training["max_grad_norm"]),
        gradient_checkpointing=bool(training["gradient_checkpointing"]),
        gradient_checkpointing_kwargs=dict(training.get("gradient_checkpointing_kwargs", {})),
        bf16=bool(training["bf16"])
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported(),
        tf32=bool(training["tf32"]) and torch.cuda.is_available(),
        use_vllm=bool(training.get("use_vllm", False)),
        vllm_mode=str(training.get("vllm_mode", "server")),
        vllm_tensor_parallel_size=int(training.get("vllm_tensor_parallel_size", 1)),
        vllm_gpu_memory_utilization=float(training.get("vllm_gpu_memory_utilization", 0.3)),
        report_to=[],
        log_completions=bool(audit_config["print_to_console"]),
        num_completions_to_print=int(audit_config["num_completions_to_print"]),
        num_generations=int(training["num_generations"]),
        max_prompt_length=int(training["max_prompt_length"]),
        max_completion_length=int(training["max_completion_length"]),
        temperature=float(training["temperature"]),
        beta=float(training["beta"]),
        mask_truncated_completions=bool(training["mask_truncated_completions"]),
        generation_kwargs=generation_kwargs,
        remove_unused_columns=False,
        seed=int(training["seed"]),
    )
    # Prevent faster ranks from reserving colocated vLLM memory while another rank is still loading.
    if world_size > 1:
        torch.distributed.barrier()

    trainer = GRPOTrainer(
        model=model,
        args=trainer_args,
        reward_funcs=reward,
        train_dataset=Dataset.from_list(train_records),
        eval_dataset=Dataset.from_list(dev_records) if evaluation_enabled else None,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    expected_steps_per_generation = batch_plan.generation_batch_size // (
        batch_plan.per_device_batch_size * batch_plan.world_size
    )
    if int(trainer.args.steps_per_generation) != expected_steps_per_generation:
        raise GRPOTrainingError(
            "TRL resolved an unexpected steps_per_generation: "
            f"{trainer.args.steps_per_generation} != {expected_steps_per_generation}."
        )
    expected_optimizer_steps = math.ceil(
        len(train_records) / batch_plan.effective_prompt_batch_size
    )
    logger.info(
        "P5 batch plan: %s; expected optimizer steps: %d", batch_plan, expected_optimizer_steps
    )
    train_output = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    fsdp_unit_count = int(getattr(trainer, "_fsdp_rollout_unit_count", 0))
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.accelerator.wait_for_everyone()
    trainer.save_model(str(adapter_dir))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(adapter_dir)
    trainer.accelerator.wait_for_everyone()
    metadata_path = run_dir / "training_metadata.json"
    metadata = {
        "experiment_key": "p5_grpo",
        "stage": "post_training",
        "dataset_key": dataset_key,
        "model_key": model_key,
        "ablation": selected_ablation,
        "p4_sft_adapter": str(sft_adapter_dir),
        "p5_adapter": str(adapter_dir),
        "resume_from_checkpoint": resume_from_checkpoint,
        "train_examples": len(train_records),
        "max_train_examples": max_train_examples,
        "dev_examples": len(dev_records),
        "in_training_evaluation": evaluation_enabled,
        "reward": grpo_root["reward"]["ablations"][selected_ablation],
        "cometkiwi": grpo_root["reward"]["cometkiwi"],
        "comet": grpo_root["reward"]["comet"],
        "training": training,
        "effective_completion_batch_size": batch_plan.effective_completion_batch_size,
        "effective_prompt_batch_size": batch_plan.effective_prompt_batch_size,
        "generation_batch_size": batch_plan.generation_batch_size,
        "prompts_per_generation_batch": batch_plan.prompts_per_generation_batch,
        "expected_optimizer_steps": expected_optimizer_steps,
        "health_examples": len(health_records),
        "trainable_lora_parameters": trainable_lora_parameters,
        "sequence_packing": bool(training["sequence_packing"]),
        "distributed": {
            "world_size": batch_plan.world_size,
            "fsdp": bool(trainer.is_fsdp_enabled),
            "fsdp_unit_count": fsdp_unit_count,
            "model_parallel": len(model_devices) > 1,
            "model_devices": sorted(model_devices),
        },
        "training_metrics": dict(train_output.metrics),
        "optimizer_steps": int(trainer.state.global_step),
    }
    if trainer.is_world_process_zero():
        save_json(metadata, metadata_path)
    trainer.accelerator.wait_for_everyone()
    return {"adapter_dir": str(adapter_dir), "training_metadata": str(metadata_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P5 GRPO from a P4 SFT adapter.")
    parser.add_argument("--dataset", required=True, choices=("en_ca", "en_eu"))
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--ablation",
        choices=("a0", "a1", "a2", "a3", "a4", "a3v2", "a3v3", "a3v4", "a3v5", "a5"),
        default=None,
    )
    parser.add_argument("--config", default="configs/grpo.yaml")
    parser.add_argument("--sft-config", default="configs/sft.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--adapter-root", default=None)
    parser.add_argument("--results-root", default=None)
    parser.add_argument("--sft-adapter-root", default=None)
    parser.add_argument(
        "--disable-health-check",
        action="store_true",
        help="Disable the separate greedy policy-health callback for a bounded smoke run.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logger.info(
        "P5 GRPO complete: %s",
        train_grpo(
            args.dataset,
            args.model,
            ablation=args.ablation,
            config_path=args.config,
            sft_config_path=args.sft_config,
            models_config_path=args.models_config,
            resume_from_checkpoint=args.resume_from_checkpoint,
            max_train_examples=args.max_train_examples,
            adapter_root=args.adapter_root,
            results_root=args.results_root,
            sft_adapter_root=args.sft_adapter_root,
            disable_health_check=args.disable_health_check,
        ),
    )


if __name__ == "__main__":
    main()
