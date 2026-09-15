"""Persist held-out HT-versus-MT calibration statistics for one language."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.contrastive_lm.reward import ContrastiveHTMTNaturalnessScorer
from src.utils.config import load_contrastive_lm_config
from src.utils.io import save_json


def audit_calibration(
    dataset_key: str,
    config_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Recompute the development calibration and save its complete audit record."""
    config_path = Path(config_path)
    config = load_contrastive_lm_config(config_path)["contrastive_lm"]
    language = config["languages"][dataset_key]
    scorer = ContrastiveHTMTNaturalnessScorer(dataset_key, config_path)
    calibration = dict(scorer.calibration)
    calibration["class_mean_gap"] = (
        calibration["ht_mean_margin"] - calibration["mt_mean_margin"]
    )
    calibration["gate_passed"] = calibration["class_mean_gap"] > 0.0
    adapter_root = Path(config["adapter_root"]) / dataset_key
    destination = (
        Path(output_path)
        if output_path is not None
        else Path(config["results_root"]) / dataset_key / "calibration_metadata.json"
    )
    payload = {
        "method": "contrastive_ht_mt_development_calibration",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_key": dataset_key,
        "target_language": language["target_language"],
        "base_model": language["base_model"],
        "config_path": str(config_path),
        "data_root": config["data_root"],
        "adapters": {
            "ht": str(adapter_root / "ht"),
            "mt": str(adapter_root / "mt"),
        },
        "formula": "mean(avg_logprob_ht - avg_logprob_mt)",
        "calibration": calibration,
    }
    saved = save_json(payload, destination)
    print(json.dumps(payload, indent=2, sort_keys=True))
    del scorer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return saved


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("en_eu", "en_ca"))
    parser.add_argument("--config", default="configs/contrastive_lm_qwen.yaml")
    parser.add_argument("--output", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    saved = audit_calibration(args.dataset, args.config, args.output)
    print(f"Saved calibration audit: {saved}")


if __name__ == "__main__":
    main()
