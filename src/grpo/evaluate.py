"""Evaluate one P5 GRPO ablation with the shared deterministic MT protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.sft.evaluate import VALID_EVAL_SPLITS, evaluate_sft_split
from src.utils.config import load_grpo_config


class GRPOEvaluationError(RuntimeError):
    """Raised when a P5 ablation cannot be evaluated safely."""


def evaluate_grpo_split(
    dataset_key: str,
    model_key: str,
    ablation: str,
    config_path: str | Path = "configs/grpo.yaml",
    sft_config_path: str | Path = "configs/sft.yaml",
    models_config_path: str | Path = "configs/models.yaml",
    split: str = "global_dev",
    limit: int | None = None,
    evaluation_file: str | Path | None = None,
    output_dataset_key: str | None = None,
    max_input_length: int | None = None,
    max_new_tokens: int | None = None,
    sentence_boundary_count: int | None = None,
) -> dict[str, str]:
    """Decode a P5 adapter greedily on a canonical or immutable evaluation split."""
    if evaluation_file is None and split not in VALID_EVAL_SPLITS:
        raise GRPOEvaluationError(f"Unsupported evaluation split: {split}")
    config = load_grpo_config(config_path)["grpo"]
    if dataset_key not in config["target_languages"]:
        raise GRPOEvaluationError(f"Unsupported P5 dataset: {dataset_key}")
    if ablation not in config["reward"]["ablations"]:
        raise GRPOEvaluationError(f"Unknown P5 ablation: {ablation}")
    return evaluate_sft_split(
        dataset_key,
        model_key,
        config_path=sft_config_path,
        models_config_path=models_config_path,
        adapter_root=Path(config["adapter_root"]) / ablation,
        results_root=Path(config["results_root"]) / ablation,
        split=split,
        limit=limit,
        experiment_key=f"p5_grpo_{ablation}",
        evaluation_file=evaluation_file,
        output_dataset_key=output_dataset_key,
        max_input_length=max_input_length,
        max_new_tokens=max_new_tokens,
        sentence_boundary_count=sentence_boundary_count,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one P5 GRPO ablation.")
    parser.add_argument("--dataset", required=True, choices=("en_ca", "en_eu"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--ablation", required=True, choices=("a0", "a1", "a2", "a3", "a4", "a3v2", "a3v3", "a3v4", "a3v5", "a5"))
    parser.add_argument("--config", default="configs/grpo.yaml")
    parser.add_argument("--sft-config", default="configs/sft.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--split", default="global_dev", help="Canonical split label, or a label for --evaluation-file.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--evaluation-file", default=None, help="Immutable JSONL with id/source/target fields.")
    parser.add_argument("--output-dataset-key", default=None, help="Dataset label used only for result paths.")
    parser.add_argument("--max-input-length", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--sentence-boundary-count",
        type=int,
        default=None,
        help="Require this many tagged translation units in each completion.",
    )
    args = parser.parse_args()
    print(evaluate_grpo_split(
        args.dataset, args.model, args.ablation, args.config, args.sft_config,
        args.models_config, args.split, args.limit, args.evaluation_file, args.output_dataset_key,
        args.max_input_length, args.max_new_tokens,
        args.sentence_boundary_count,
    ))


if __name__ == "__main__":
    main()
