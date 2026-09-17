"""Load experiment settings from YAML files."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.utils.errors import PipelineError


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
        raise PipelineError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        try:
            data = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise PipelineError(f"Failed to parse YAML in {path}: {exc}") from exc
    if data is None:
        raise PipelineError(f"Config file is empty: {path}")
    return data


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    return _read_yaml(path)


def _load_section(path: str | Path, section: str) -> dict[str, Any]:
    config = _read_yaml(path)
    if not isinstance(config.get(section), Mapping):
        raise PipelineError(f"Config {path} must contain a mapping named {section!r}")
    return config


def load_classifier_config(path: str | Path = "configs/classifier.yaml") -> dict[str, Any]:
    return _load_section(path, "classifier")


def load_contrastive_lm_config(
    path: str | Path = "configs/contrastive_lm.yaml",
) -> dict[str, Any]:
    return _load_section(path, "contrastive_lm")


def load_grpo_config(path: str | Path = "configs/grpo.yaml") -> dict[str, Any]:
    """Load GRPO settings and verify each reward is a convex combination."""
    config = _load_section(path, "grpo")
    ablations = config["grpo"]["reward"]["ablations"]
    for name, entry in ablations.items():
        weights = [float(value) for value in entry["weights"].values()]
        if any(value < 0.0 for value in weights) or abs(sum(weights) - 1.0) > 1e-8:
            raise PipelineError(f"GRPO reward weights for {name} must be non-negative and sum to 1")
    return config


def load_models_config(path: str | Path = "configs/models.yaml") -> dict[str, ModelEntry]:
    config = _load_section(path, "models")
    defaults = dict(config.get("defaults", {}))
    models: dict[str, ModelEntry] = {}
    for key, values in config["models"].items():
        entry = {**defaults, **values}
        models[key] = ModelEntry(
            key=key,
            hf_id=str(entry["hf_id"]),
            family=str(entry["family"]),
            parameter_scale=str(entry["parameter_scale"]),
            instruction_tuned=bool(entry["instruction_tuned"]),
            comparison_group=str(entry["comparison_group"]),
            backend=str(entry.get("backend", "transformers")),
            dtype=str(entry.get("dtype", "bf16")),
            quantization=entry.get("quantization"),
            chat_template_options=dict(entry.get("chat_template_options", {})),
            prompt_adapter=str(entry.get("prompt_adapter", "chat")),
            enable_thinking=bool(entry.get("enable_thinking", False)),
            max_context_length=int(entry.get("max_context_length", 8192)),
            tensor_parallel_size=int(entry.get("tensor_parallel_size", 1)),
            gated=bool(entry.get("gated", False)),
        )
    return models


def get_model_entry(key: str, path: str | Path = "configs/models.yaml") -> ModelEntry:
    models = load_models_config(path)
    if key not in models:
        raise PipelineError(f"Unknown model key '{key}'. Available keys: {sorted(models.keys())}")
    return models[key]


def load_datasets_config(path: str | Path = "configs/datasets.yaml") -> dict[str, Any]:
    config = _load_section(path, "datasets")
    return dict(config["datasets"])


def get_dataset_entry(key: str, path: str | Path = "configs/datasets.yaml") -> dict[str, Any]:
    datasets = load_datasets_config(path)
    if key not in datasets:
        raise PipelineError(
            f"Unknown dataset key '{key}'. Available keys: {sorted(datasets.keys())}"
        )
    return datasets[key]


def load_preprocessing_config(path: str | Path = "configs/preprocessing.yaml") -> dict[str, Any]:
    return load_yaml_config(path)


def merge_model_overrides(
    config: Mapping[str, Any], section: str, model_key: str
) -> dict[str, Any]:
    """Apply the overrides for one model without changing the base configuration."""
    override = config.get("model_overrides", {}).get(model_key, {})
    if not isinstance(override, Mapping):
        raise PipelineError(f"{section}.model_overrides.{model_key} must be a mapping")

    def merge(base: Mapping[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(dict(base))
        for key, value in extra.items():
            if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
                result[key] = merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result

    return merge(config, override)


def load_generation_config(path: str | Path = "configs/generation.yaml") -> dict[str, Any]:
    return _load_section(path, "baseline")


def load_experiments_config(path: str | Path = "configs/experiments.yaml") -> dict[str, Any]:
    config = _load_section(path, "experiments")
    return dict(config["experiments"])


def get_experiment_entry(key: str, path: str | Path = "configs/experiments.yaml") -> dict[str, Any]:
    experiments = load_experiments_config(path)
    if key not in experiments:
        raise PipelineError(
            f"Unknown experiment key '{key}'. Available keys: {sorted(experiments.keys())}"
        )
    return experiments[key]


def load_sft_config(path: str | Path = "configs/sft.yaml") -> dict[str, Any]:
    return _load_section(path, "sft")
