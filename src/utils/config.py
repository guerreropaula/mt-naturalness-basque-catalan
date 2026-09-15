"""Configuration loading utilities.

All experiment scripts must load their settings through this module rather
than hard-coding values. Configs are plain YAML files under `configs/`.

"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

logger = logging.getLogger(__name__)

_ALLOWED_MODEL_BACKENDS = {"transformers", "vllm", "api"}
_ALLOWED_MODEL_DTYPES = {"auto", "bf16", "fp16", "fp32"}
_ALLOWED_MODEL_QUANTIZATION = {None, "4bit", "8bit"}
_ALLOWED_PROMPT_ADAPTERS = {"chat", "salamandrata_translation"}
_ALLOWED_DATASET_LOADERS = {"ehu_hac", "hf_dataset", "local_jsonl"}
_ALLOWED_OUTPUT_FORMATS = {"parquet", "jsonl"}
_ALLOWED_REVIEW_STATUSES = {"accepted", "rejected", "manual_review"}
_ALLOWED_EXPERIMENT_FAMILIES = {"baseline", "prompting", "advanced", "post_training"}
_ALLOWED_EXPERIMENT_MODES = {
    "single_pass",
    "self_polish",
    "grpo_sampling",
    "step_by_step_pipeline",
    "sft_training",
    "grpo_training",
}


class ConfigError(RuntimeError):
    """Raised for missing files, malformed YAML, or missing required keys."""


@dataclass(frozen=True)
class ModelEntry:
    key: str
    hf_id: str
    family: str
    parameter_scale: str
    instruction_tuned: bool
    comparison_group: str
    backend: str
    dtype: str
    quantization: str | None
    chat_template_options: Mapping[str, Any]
    prompt_adapter: str
    enable_thinking: bool
    max_context_length: int
    tensor_parallel_size: int
    gated: bool

    @property
    def is_id_confirmed(self) -> bool:
        return self.hf_id != "TO_BE_CONFIRMED"


def _read_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        try:
            data = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ConfigError(f"Failed to parse YAML in {path}: {exc}") from exc
    if data is None:
        raise ConfigError(f"Config file is empty: {path}")
    return data


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a single YAML config file and return it as a plain dict."""
    config = _read_yaml(path)
    logger.debug("Loaded config from %s", path)
    return config


def load_classifier_config(path: str | Path = "configs/classifier.yaml") -> dict[str, Any]:
    """Load the target-side reference-likeness classifier configuration."""
    config = _read_yaml(path)
    classifier = _ensure_mapping(config.get("classifier"), f"'classifier' section in {path}")
    required = {
        "output_dir",
        "source_pairs_dir",
        "results_dir",
        "size_modes",
        "backbones",
        "training",
    }
    _require_fields(classifier, required, f"'classifier' section in {path}")
    return config



def load_contrastive_lm_config(
    path: str | Path = "configs/contrastive_lm.yaml",
) -> dict[str, Any]:
    """Load target-side contrastive human-versus-machine LM settings."""
    config = _read_yaml(path)
    contrastive = _ensure_mapping(
        config.get("contrastive_lm"), f"'contrastive_lm' section in {path}"
    )
    _require_fields(
        contrastive,
        {"data_root", "adapter_root", "results_root", "labels", "languages", "model", "training", "scoring"},
        f"'contrastive_lm' section in {path}",
    )
    labels = _ensure_mapping(contrastive["labels"], f"contrastive_lm.labels in {path}")
    if {"ht", "mt"} - set(labels) or int(labels["ht"]) != 1 or int(labels["mt"]) != 0:
        raise ConfigError(f"contrastive_lm.labels in {path} must map ht to 1 and mt to 0.")
    languages = _ensure_mapping(contrastive["languages"], f"contrastive_lm.languages in {path}")
    for dataset_key in ("en_eu", "en_ca"):
        language = _ensure_mapping(
            languages.get(dataset_key), f"contrastive_lm.languages.{dataset_key} in {path}"
        )
        _require_fields(language, {"target_language", "base_model"}, f"contrastive_lm.languages.{dataset_key} in {path}")
    model = _ensure_mapping(contrastive["model"], f"contrastive_lm.model in {path}")
    _require_fields(model, {"load_in_4bit", "max_length", "lora"}, f"contrastive_lm.model in {path}")
    if int(model["max_length"]) < 2:
        raise ConfigError(f"contrastive_lm.model.max_length in {path} must be at least 2")
    lora = _ensure_mapping(model["lora"], f"contrastive_lm.model.lora in {path}")
    _require_fields(lora, {"r", "lora_alpha", "lora_dropout", "target_modules"}, f"contrastive_lm.model.lora in {path}")
    if int(lora["r"]) <= 0 or int(lora["lora_alpha"]) <= 0 or not lora["target_modules"]:
        raise ConfigError(f"contrastive_lm.model.lora in {path} has invalid LoRA settings")
    training = _ensure_mapping(contrastive["training"], f"contrastive_lm.training in {path}")
    _require_fields(
        training,
        {"seed", "num_train_epochs", "per_device_train_batch_size", "per_device_eval_batch_size", "gradient_accumulation_steps", "learning_rate", "weight_decay", "warmup_ratio", "lr_scheduler_type", "optim", "logging_steps", "save_total_limit", "bf16", "gradient_checkpointing"},
        f"contrastive_lm.training in {path}",
    )
    if int(training["per_device_train_batch_size"]) <= 0 or int(training["gradient_accumulation_steps"]) <= 0:
        raise ConfigError(f"contrastive_lm.training in {path} requires positive batch settings")
    scoring = _ensure_mapping(contrastive["scoring"], f"contrastive_lm.scoring in {path}")
    _require_fields(scoring, {"batch_size", "text_field"}, f"contrastive_lm.scoring in {path}")
    if int(scoring["batch_size"]) <= 0 or not str(scoring["text_field"]):
        raise ConfigError(f"contrastive_lm.scoring in {path} requires a positive batch size and text field")
    return config

def load_grpo_config(path: str | Path = "configs/grpo.yaml") -> dict[str, Any]:
    """Load and validate the P5 GRPO reward, data, and training configuration."""
    config = _read_yaml(path)
    grpo = _ensure_mapping(config.get("grpo"), f"'grpo' section in {path}")
    _require_fields(
        grpo,
        {
            "data_dir",
            "adapter_root",
            "results_root",
            "sft_adapter_roots",
            "target_languages",
            "chat_format",
            "lora",
            "reward",
            "training",
        },
        f"'grpo' section in {path}",
    )
    reward = _ensure_mapping(grpo["reward"], f"grpo.reward in {path}")
    _require_fields(
        reward,
        {"active_ablation", "group_size", "cometkiwi", "comet", "classifier", "ablations"},
        f"grpo.reward in {path}",
    )
    group_size = int(reward["group_size"])
    if group_size < 2:
        raise ConfigError(f"grpo.reward.group_size in {path} must be at least 2")
    cometkiwi = _ensure_mapping(reward["cometkiwi"], f"grpo.reward.cometkiwi in {path}")
    _require_fields(cometkiwi, {"model", "batch_size", "gpus"}, f"grpo.reward.cometkiwi in {path}")
    if not cometkiwi["model"] or int(cometkiwi["batch_size"]) <= 0 or int(cometkiwi["gpus"]) < 0:
        raise ConfigError(f"grpo.reward.cometkiwi in {path} has invalid model, batch_size, or gpus")
    comet = _ensure_mapping(reward["comet"], f"grpo.reward.comet in {path}")
    _require_fields(comet, {"model", "batch_size", "gpus"}, f"grpo.reward.comet in {path}")
    if not comet["model"] or int(comet["batch_size"]) <= 0 or int(comet["gpus"]) < 0:
        raise ConfigError(f"grpo.reward.comet in {path} has invalid model, batch_size, or gpus")
    ablations = _ensure_mapping(reward["ablations"], f"grpo.reward.ablations in {path}")
    if reward["active_ablation"] not in ablations:
        raise ConfigError(f"grpo.reward.active_ablation in {path} is not defined in ablations")
    allowed_weights = {"chrfpp", "bleu", "cometkiwi", "comet", "self_bleu", "length", "reference_likeness"}
    for name, entry in ablations.items():
        entry = _ensure_mapping(entry, f"grpo.reward.ablations.{name} in {path}")
        weights = _ensure_mapping(
            entry.get("weights"), f"grpo.reward.ablations.{name}.weights in {path}"
        )
        unknown = set(weights) - allowed_weights
        if unknown:
            raise ConfigError(
                f"grpo.reward.ablations.{name} in {path} has unknown weights: {sorted(unknown)}"
            )
        numeric = [float(value) for value in weights.values()]
        if any(value < 0.0 for value in numeric) or abs(sum(numeric) - 1.0) > 1e-8:
            raise ConfigError(
                f"grpo.reward.ablations.{name} in {path} must use non-negative weights summing to 1"
            )
    training = _ensure_mapping(grpo["training"], f"grpo.training in {path}")
    _require_fields(
        training,
        {
            "num_generations",
            "learning_rate",
            "num_train_epochs",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "warmup_ratio",
            "lr_scheduler_type",
            "optim",
            "max_prompt_length",
            "max_completion_length",
            "temperature",
            "beta",
            "sequence_packing",
            "generation_kwargs",
            "effective_prompt_batch_size",
            "generation_batch_size",
            "max_grad_norm",
            "mask_truncated_completions",
            "health_check",
        },
        f"grpo.training in {path}",
    )
    if int(training["num_generations"]) != group_size:
        raise ConfigError(
            f"grpo.training.num_generations in {path} must equal grpo.reward.group_size"
        )
    if (
        int(training["per_device_train_batch_size"]) <= 0
        or int(training["gradient_accumulation_steps"]) <= 0
    ):
        raise ConfigError(
            f"grpo.training in {path} requires positive batch and accumulation values"
        )
    completion_batch_size = int(training["per_device_train_batch_size"]) * int(
        training["gradient_accumulation_steps"]
    )
    if completion_batch_size % group_size:
        raise ConfigError(
            f"grpo.training completion batch {completion_batch_size} in {path} must be "
            f"divisible by G={group_size}"
        )
    effective_prompt_batch_size = completion_batch_size // group_size
    if effective_prompt_batch_size != int(training["effective_prompt_batch_size"]):
        raise ConfigError(
            f"grpo.training effective prompt batch in {path} is "
            f"{effective_prompt_batch_size}, not {training['effective_prompt_batch_size']}"
        )
    generation_batch_size = int(training["generation_batch_size"])
    if generation_batch_size <= 0 or generation_batch_size % group_size:
        raise ConfigError(
            f"grpo.training.generation_batch_size in {path} must be positive and "
            f"divisible by G={group_size}"
        )
    if generation_batch_size % int(training["per_device_train_batch_size"]):
        raise ConfigError(
            f"grpo.training.generation_batch_size in {path} must be divisible by "
            "per_device_train_batch_size"
        )
    if float(training["max_grad_norm"]) <= 0.0:
        raise ConfigError(f"grpo.training.max_grad_norm in {path} must be positive")
    prompt_length = int(training["max_prompt_length"])
    completion_length = int(training["max_completion_length"])
    if prompt_length <= 0 or completion_length <= 0:
        raise ConfigError(
            f"grpo.training in {path} requires positive prompt and completion lengths"
        )
    if prompt_length + completion_length != 512:
        raise ConfigError(
            f"grpo.training in {path} must use a 512-token total window; found "
            f"{prompt_length} prompt + {completion_length} completion tokens"
        )
    generation_kwargs = _ensure_mapping(
        training["generation_kwargs"], f"grpo.training.generation_kwargs in {path}"
    )
    if (
        generation_kwargs.get("do_sample") is not True
        or int(generation_kwargs.get("num_beams", 1)) != 1
    ):
        raise ConfigError(
            f"grpo.training.generation_kwargs in {path} must use multinomial sampling with one beam"
        )
    if float(training["temperature"]) <= 0.0:
        raise ConfigError(
            f"grpo.training.temperature in {path} must be positive for sampled GRPO rollouts"
        )
    health = _ensure_mapping(training["health_check"], f"grpo.training.health_check in {path}")
    _require_fields(
        health,
        {
            "enabled",
            "interval_steps",
            "examples",
            "batch_size",
            "max_new_tokens",
            "min_mean_chrfpp",
            "max_repeated_token_ratio",
            "max_repeated_rows_fraction",
            "max_unterminated_fraction",
            "max_empty_fraction",
            "max_duplicate_fraction",
        },
        f"grpo.training.health_check in {path}",
    )
    if bool(health["enabled"]) and any(
        int(health[field]) <= 0
        for field in ("interval_steps", "examples", "batch_size", "max_new_tokens")
    ):
        raise ConfigError(f"grpo.training.health_check in {path} requires positive sizes")
    for field in (
        "min_mean_chrfpp",
        "max_repeated_token_ratio",
        "max_repeated_rows_fraction",
        "max_unterminated_fraction",
        "max_empty_fraction",
        "max_duplicate_fraction",
    ):
        if not 0.0 <= float(health[field]) <= 1.0:
            raise ConfigError(f"grpo.training.health_check.{field} in {path} must be in [0, 1]")
    lora = _ensure_mapping(grpo["lora"], f"grpo.lora in {path}")
    _require_fields(lora, {"r", "lora_alpha"}, f"grpo.lora in {path}")
    if int(lora["r"]) <= 0 or int(lora["lora_alpha"]) <= 0:
        raise ConfigError(f"grpo.lora in {path} requires positive r and lora_alpha")
    return config


def _ensure_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{context} must be a mapping, got {type(value).__name__}")
    return value


def _require_fields(entry: Mapping[str, Any], required_fields: set[str], context: str) -> None:
    missing = required_fields - set(entry.keys())
    if missing:
        raise ConfigError(f"{context} is missing required fields: {sorted(missing)}")


def load_models_config(path: str | Path = "configs/models.yaml") -> dict[str, ModelEntry]:
    """Load and validate the model registry, merging in top-level defaults."""
    raw = _read_yaml(path)
    defaults = _ensure_mapping(raw.get("defaults", {}), f"'defaults' section in {path}")
    models_raw = raw.get("models")
    if not models_raw:
        raise ConfigError(f"No 'models' section found in {path}")
    models_raw = _ensure_mapping(models_raw, f"'models' section in {path}")

    required_fields = {
        "hf_id",
        "family",
        "parameter_scale",
        "instruction_tuned",
        "comparison_group",
    }

    models: dict[str, ModelEntry] = {}
    for key, entry_raw in models_raw.items():
        entry_raw = _ensure_mapping(entry_raw, f"Model '{key}' in {path}")
        merged = copy.deepcopy(defaults)
        merged.update(entry_raw)

        _require_fields(merged, required_fields, f"Model '{key}' in {path}")

        backend = merged.get("backend", "transformers")
        dtype = merged.get("dtype", "bf16")
        quantization = merged.get("quantization")
        max_context_length = int(merged.get("max_context_length", 8192))
        tensor_parallel_size = int(merged.get("tensor_parallel_size", 1))
        chat_template_options = merged.get("chat_template_options", {})
        prompt_adapter = str(merged.get("prompt_adapter", "chat"))

        if backend not in _ALLOWED_MODEL_BACKENDS:
            raise ConfigError(
                f"Model '{key}' in {path} has unsupported backend '{backend}'. "
                f"Allowed values: {sorted(_ALLOWED_MODEL_BACKENDS)}"
            )
        if dtype not in _ALLOWED_MODEL_DTYPES:
            raise ConfigError(
                f"Model '{key}' in {path} has unsupported dtype '{dtype}'. "
                f"Allowed values: {sorted(_ALLOWED_MODEL_DTYPES)}"
            )
        if quantization not in _ALLOWED_MODEL_QUANTIZATION:
            raise ConfigError(
                f"Model '{key}' in {path} has unsupported quantization '{quantization}'. "
                f"Allowed values: {sorted(value for value in _ALLOWED_MODEL_QUANTIZATION if value is not None)} + [None]"
            )
        if prompt_adapter not in _ALLOWED_PROMPT_ADAPTERS:
            raise ConfigError(
                f"Model '{key}' in {path} has unsupported prompt_adapter "
                f"'{prompt_adapter}'. Allowed values: {sorted(_ALLOWED_PROMPT_ADAPTERS)}"
            )
        if max_context_length <= 0:
            raise ConfigError(f"Model '{key}' in {path} must have max_context_length > 0")
        if tensor_parallel_size <= 0:
            raise ConfigError(f"Model '{key}' in {path} must have tensor_parallel_size > 0")
        if not isinstance(chat_template_options, Mapping):
            raise ConfigError(
                f"Model '{key}' in {path} must define chat_template_options as a mapping"
            )

        if merged["hf_id"] == "TO_BE_CONFIRMED":
            logger.warning(
                "Model '%s' has an unconfirmed hf_id (TO_BE_CONFIRMED); "
                "it cannot be loaded until this is resolved.",
                key,
            )

        models[key] = ModelEntry(
            key=key,
            hf_id=merged["hf_id"],
            family=merged["family"],
            parameter_scale=merged["parameter_scale"],
            instruction_tuned=bool(merged["instruction_tuned"]),
            comparison_group=merged["comparison_group"],
            backend=backend,
            dtype=dtype,
            quantization=quantization,
            chat_template_options=chat_template_options,
            prompt_adapter=prompt_adapter,
            enable_thinking=bool(merged.get("enable_thinking", False)),
            max_context_length=max_context_length,
            tensor_parallel_size=tensor_parallel_size,
            gated=bool(merged.get("gated", False)),
        )

    logger.info("Loaded %d model entries from %s", len(models), path)
    return models


def get_model_entry(key: str, path: str | Path = "configs/models.yaml") -> ModelEntry:
    """Convenience accessor for a single model by its registry key."""
    models = load_models_config(path)
    if key not in models:
        raise ConfigError(f"Unknown model key '{key}'. Available keys: {sorted(models.keys())}")
    return models[key]


def load_datasets_config(path: str | Path = "configs/datasets.yaml") -> dict[str, Any]:
    raw = _read_yaml(path)
    datasets = _ensure_mapping(raw.get("datasets"), f"'datasets' section in {path}")
    for key, entry in datasets.items():
        entry = _ensure_mapping(entry, f"Dataset '{key}' in {path}")
        _require_fields(
            entry,
            {"name", "source_lang", "target_lang", "corpus_id", "loader"},
            f"Dataset '{key}' in {path}",
        )
        if entry["loader"] not in _ALLOWED_DATASET_LOADERS:
            raise ConfigError(
                f"Dataset '{key}' in {path} has unsupported loader '{entry['loader']}'"
            )
        if entry["loader"] == "ehu_hac":
            _require_fields(
                _ensure_mapping(entry.get("paths"), f"Dataset '{key}' paths in {path}"),
                {"root_dir", "source_file", "target_file"},
                f"Dataset '{key}' paths in {path}",
            )
        elif entry["loader"] == "local_jsonl":
            _require_fields(
                _ensure_mapping(entry.get("paths"), f"Dataset '{key}' paths in {path}"),
                {"file"},
                f"Dataset '{key}' paths in {path}",
            )
        else:
            _require_fields(
                entry, {"hf_repo_id", "split", "column_mapping"}, f"Dataset '{key}' in {path}"
            )
            _require_fields(
                _ensure_mapping(
                    entry["column_mapping"], f"Dataset '{key}' column_mapping in {path}"
                ),
                {"source", "target"},
                f"Dataset '{key}' column_mapping in {path}",
            )
    return dict(datasets)


def get_dataset_entry(key: str, path: str | Path = "configs/datasets.yaml") -> dict[str, Any]:
    datasets = load_datasets_config(path)
    if key not in datasets:
        raise ConfigError(f"Unknown dataset key '{key}'. Available keys: {sorted(datasets.keys())}")
    return datasets[key]


def load_preprocessing_config(path: str | Path = "configs/preprocessing.yaml") -> dict[str, Any]:
    """Load the minimal ordered-preprocessing configuration."""
    config = load_yaml_config(path)
    _require_fields(
        config,
        {"normalization", "text_filtering", "language_id", "length_filtering", "splits", "output"},
        f"Preprocessing config in {path}",
    )
    language_id = _ensure_mapping(config["language_id"], f"language_id in {path}")
    threshold = float(language_id.get("confidence_threshold", -1))
    if not 0.0 <= threshold <= 1.0 or not language_id.get("model_path"):
        raise ConfigError(
            f"Preprocessing config in {path} needs a fastText model path and confidence in [0, 1]"
        )
    length = _ensure_mapping(config["length_filtering"], f"length_filtering in {path}")
    min_tokens = int(length.get("min_tokens", 1))
    max_tokens = int(length.get("max_tokens", 10**9))
    if min_tokens <= 0 or min_tokens > max_tokens:
        raise ConfigError(
            f"Preprocessing config in {path} must satisfy 0 < min_tokens <= max_tokens"
        )
    for minimum, maximum in (
        ("min_token_ratio", "max_token_ratio"),
        ("min_char_ratio", "max_char_ratio"),
    ):
        if float(length[minimum]) <= 0 or float(length[minimum]) > float(length[maximum]):
            raise ConfigError(
                f"Preprocessing config in {path} must satisfy 0 < {minimum} <= {maximum}"
            )
    splits = _ensure_mapping(config["splits"], f"splits in {path}")
    for key in ("train_size", "dev_size", "test_size"):
        if int(splits.get(key, 0)) <= 0:
            raise ConfigError(f"Preprocessing config in {path} needs a positive splits.{key}")
    return config


def load_generation_config(path: str | Path = "configs/generation.yaml") -> dict[str, Any]:
    """Load and validate decoding defaults for baseline and adaptation runs."""
    config = load_yaml_config(path)

    _require_fields(
        config,
        {"baseline"},
        f"Generation config in {path}",
    )

    baseline = _ensure_mapping(config["baseline"], f"baseline section in {path}")
    if baseline.get("do_sample") is not False:
        raise ConfigError(f"Generation config in {path} must set baseline.do_sample to false")
    if int(baseline.get("num_beams", 1)) != 1:
        raise ConfigError(f"Generation config in {path} must set baseline.num_beams to 1")
    if baseline.get("temperature") is not None:
        raise ConfigError(f"Generation config in {path} must leave baseline.temperature as null")
    if baseline.get("top_p") is not None:
        raise ConfigError(f"Generation config in {path} must leave baseline.top_p as null")
    if baseline.get("qwen3_enable_thinking") is not False:
        raise ConfigError(
            f"Generation config in {path} must keep baseline.qwen3_enable_thinking disabled"
        )

    return config


def load_experiments_config(path: str | Path = "configs/experiments.yaml") -> dict[str, Any]:
    """Load and validate the experiment registry used by prompt-based runners."""
    raw = _read_yaml(path)
    experiments = raw.get("experiments")
    if not experiments:
        raise ConfigError(f"No 'experiments' section found in {path}")
    experiments = _ensure_mapping(experiments, f"'experiments' section in {path}")

    def _validate_prompt_mapping(prompt: Any, context: str) -> None:
        prompt = _ensure_mapping(prompt, context)
        if prompt.get("single_user_prompt_template") is not None:
            return
        _require_fields(prompt, {"system_prompt", "user_prompt_template"}, context)

    for key, entry in experiments.items():
        entry = _ensure_mapping(entry, f"Experiment '{key}' in {path}")
        _require_fields(
            entry,
            {"family", "mode", "description"},
            f"Experiment '{key}' in {path}",
        )
        if entry["family"] not in _ALLOWED_EXPERIMENT_FAMILIES:
            raise ConfigError(
                f"Experiment '{key}' in {path} has unsupported family '{entry['family']}'. "
                f"Allowed values: {sorted(_ALLOWED_EXPERIMENT_FAMILIES)}"
            )
        if entry["mode"] not in _ALLOWED_EXPERIMENT_MODES:
            raise ConfigError(
                f"Experiment '{key}' in {path} has unsupported mode '{entry['mode']}'. "
                f"Allowed values: {sorted(_ALLOWED_EXPERIMENT_MODES)}"
            )

        if entry["mode"] in {"single_pass", "grpo_sampling"}:
            _require_fields(
                entry,
                {"generation_profile", "prompt"},
                f"Experiment '{key}' in {path}",
            )
            _validate_prompt_mapping(entry["prompt"], f"Experiment '{key}' prompt in {path}")

        if entry["mode"] == "self_polish":
            _require_fields(
                entry,
                {
                    "initial_generation_profile",
                    "refinement_generation_profile",
                    "initial_prompt",
                    "refinement_prompt",
                },
                f"Experiment '{key}' in {path}",
            )
            _validate_prompt_mapping(
                entry["initial_prompt"], f"Experiment '{key}' initial_prompt in {path}"
            )
            _validate_prompt_mapping(
                entry["refinement_prompt"], f"Experiment '{key}' refinement_prompt in {path}"
            )

        if entry["mode"] in {"sft_training", "grpo_training"}:
            _require_fields(entry, {"config"}, f"Experiment '{key}'")

        if entry["mode"] == "step_by_step_pipeline":
            _require_fields(
                entry,
                {"generation_profile", "stages"},
                f"Experiment '{key}' in {path}",
            )
            stages = entry["stages"]
            if not isinstance(stages, list) or not stages:
                raise ConfigError(f"Experiment '{key}' stages in {path} must be a non-empty list")
            for index, stage in enumerate(stages):
                stage = _ensure_mapping(stage, f"Experiment '{key}' stage {index} in {path}")
                _require_fields(
                    stage,
                    {"key", "prompt"},
                    f"Experiment '{key}' stage {index} in {path}",
                )
                _validate_prompt_mapping(
                    stage["prompt"],
                    f"Experiment '{key}' stage {index} prompt in {path}",
                )

    return dict(experiments)


def get_experiment_entry(key: str, path: str | Path = "configs/experiments.yaml") -> dict[str, Any]:
    """Convenience accessor for a single experiment entry."""
    experiments = load_experiments_config(path)
    if key not in experiments:
        raise ConfigError(
            f"Unknown experiment key '{key}'. Available keys: {sorted(experiments.keys())}"
        )
    return experiments[key]


def load_sft_config(path: str | Path = "configs/sft.yaml") -> dict[str, Any]:
    """Load configuration for registry-driven LoRA supervised fine-tuning."""
    config = _read_yaml(path)
    sft = _ensure_mapping(config.get("sft"), f"'sft' section in {path}")
    required = {
        "data_dir",
        "adapter_root",
        "results_root",
        "target_languages",
        "chat_format",
        "lora",
        "quantization",
        "training",
        "evaluation",
    }
    _require_fields(sft, required, f"'sft' section in {path}")
    target_languages = _ensure_mapping(sft["target_languages"], f"sft.target_languages in {path}")
    for dataset_key in ("en_eu", "en_ca"):
        if dataset_key not in target_languages:
            raise ConfigError(f"sft.target_languages in {path} is missing {dataset_key}")
    if int(_ensure_mapping(sft["chat_format"], "sft.chat_format").get("max_seq_length", 0)) <= 0:
        raise ConfigError(f"sft.chat_format.max_seq_length in {path} must be positive")
    training = _ensure_mapping(sft["training"], "sft.training")
    _require_fields(
        training,
        {
            "num_train_epochs",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "warmup_ratio",
            "lr_scheduler_type",
            "optim",
            "packing",
        },
        f"sft.training in {path}",
    )
    if int(training["per_device_train_batch_size"]) <= 0:
        raise ConfigError(f"sft.training.per_device_train_batch_size in {path} must be positive")
    lora = _ensure_mapping(sft["lora"], "sft.lora")
    _require_fields(lora, {"r", "lora_alpha"}, f"sft.lora in {path}")
    if int(lora["r"]) <= 0 or int(lora["lora_alpha"]) <= 0:
        raise ConfigError(f"sft.lora in {path} requires positive r and lora_alpha")
    evaluation = _ensure_mapping(sft["evaluation"], "sft.evaluation")
    _require_fields(
        evaluation,
        {"do_sample", "num_beams", "temperature", "top_p"},
        f"sft.evaluation in {path}",
    )
    if (
        evaluation["do_sample"] is not False
        or int(evaluation["num_beams"]) != 1
        or evaluation["temperature"] is not None
        or evaluation["top_p"] is not None
    ):
        raise ConfigError(
            f"sft.evaluation in {path} must use deterministic greedy decoding for final P4 evaluation"
        )
    return config
