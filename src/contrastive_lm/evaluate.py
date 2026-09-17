"""Evaluate the contrastive HT/MT scorer on development and test chunks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.contrastive_lm.data import read_jsonl, split_path
from src.contrastive_lm.reward import ContrastiveHTMTNaturalnessScorer
from src.utils.config import load_contrastive_lm_config
from src.utils.errors import PipelineError
from src.utils.io import save_json


def _split_records(data_root: str | Path, dataset_key: str, split: str) -> list[dict[str, Any]]:
    records = read_jsonl(split_path(data_root, dataset_key, split))
    if any("text" not in record or "label" not in record for record in records):
        raise PipelineError(f"{dataset_key}/{split} must contain non-empty text and binary labels.")
    labels = [int(record["label"]) for record in records]
    if set(labels) != {0, 1}:
        raise PipelineError(f"{dataset_key}/{split} must contain both HT=1 and MT=0 labels.")
    if any(not str(record["text"]).strip() for record in records):
        raise PipelineError(f"{dataset_key}/{split} contains empty target text.")
    return records


def _metrics(labels: list[int], scores: list[float]) -> dict[str, float | int]:
    predictions = [int(score >= 0.5) for score in scores]
    return {
        "examples": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
        "precision": float(precision_score(labels, predictions, pos_label=1, zero_division=0)),
        "recall": float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }


def evaluate_contrastive_lm(
    dataset_key: str,
    config_path: str | Path = "configs/contrastive_lm_qwen.yaml",
    output_path: str | Path | None = None,
) -> Path:
    """Calibrate on development data, then score the development and test splits."""
    config_path = Path(config_path)
    config = load_contrastive_lm_config(config_path)["contrastive_lm"]
    if dataset_key not in config["languages"]:
        raise PipelineError(f"No configuration for {dataset_key}.")
    scorer = ContrastiveHTMTNaturalnessScorer(dataset_key, config_path)
    split_results: dict[str, dict[str, float | int]] = {}
    for split in ("dev", "test"):
        records = _split_records(config["data_root"], dataset_key, split)
        labels = [int(record["label"]) for record in records]
        scores = scorer.score([str(record["text"]) for record in records])
        split_results[split] = _metrics(labels, scores)

    destination = (
        Path(output_path)
        if output_path is not None
        else Path(config["results_root"]) / dataset_key / "heldout_metrics.json"
    )
    payload = {
        "method": "contrastive_ht_mt_causal_lm_heldout_discrimination",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_key": dataset_key,
        "target_language": config["languages"][dataset_key]["target_language"],
        "base_model": config["languages"][dataset_key]["base_model"],
        "config_path": str(config_path),
        "data_root": config["data_root"],
        "unit": "five_sentence_chunk",
        "positive_class": "human reference text",
        "negative_class": "P4 SFT MT output",
        "score": "avg_logprob_ht - avg_logprob_mt",
        "decision_rule": "ht_like if calibrated_probability >= 0.5",
        "threshold_raw_margin": scorer.center,
        "threshold_calibrated_probability": 0.5,
        "calibration": dict(scorer.calibration),
        "metrics": split_results,
    }
    del scorer
    return save_json(payload, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("en_eu", "en_ca"))
    parser.add_argument("--config", default="configs/contrastive_lm_qwen.yaml")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    saved = evaluate_contrastive_lm(args.dataset, args.config, args.output)
    print(f"Saved held-out contrastive-LM metrics: {saved}")


if __name__ == "__main__":
    main()
