"""Train and evaluate target-side HT-versus-MT classifiers."""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.utils.config import load_classifier_config
from src.utils.errors import PipelineError
from src.utils.hf_auth import get_hf_token
from src.utils.io import save_dataframe_csv, save_json

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: list[list[int]]
    examples: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "confusion_matrix": self.confusion_matrix,
            "examples": self.examples,
        }


class TargetTextDataset(Dataset[dict[str, Any]]):
    """Target texts with binary labels: 1 for human and 0 for machine translation."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        if not records:
            raise PipelineError("Classifier split is empty.")
        required = {"text", "label"}
        missing = required - set(records[0])
        if missing:
            raise PipelineError(f"Classifier examples are missing fields: {sorted(missing)}")
        labels = [int(record["label"]) for record in records]
        if set(labels) != {0, 1}:
            raise PipelineError("Each classifier split must contain both label 0 and label 1.")
        self._records = [
            {"text": str(record["text"]), "label": int(record["label"])} for record in records
        ]

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._records[index]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _metrics(labels: list[int], probabilities: list[float]) -> SplitMetrics:
    predictions = [int(value >= 0.5) for value in probabilities]
    return SplitMetrics(
        accuracy=float(accuracy_score(labels, predictions)),
        precision=float(precision_score(labels, predictions, zero_division=0)),
        recall=float(recall_score(labels, predictions, zero_division=0)),
        f1=float(f1_score(labels, predictions, zero_division=0)),
        roc_auc=float(roc_auc_score(labels, probabilities)),
        confusion_matrix=confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        examples=len(labels),
    )


def _make_loader(
    dataset: TargetTextDataset,
    tokenizer: Any,
    batch_size: int,
    max_length: int,
    shuffle: bool,
) -> DataLoader[dict[str, torch.Tensor]]:
    def collate(records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = tokenizer(
            [record["text"] for record in records],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([record["label"] for record in records], dtype=torch.long)
        return encoded

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def evaluate_classifier(model: Any, loader: DataLoader[Any], device: torch.device) -> SplitMetrics:
    model.eval()
    labels: list[int] = []
    probabilities: list[float] = []
    for batch in loader:
        batch = {name: value.to(device) for name, value in batch.items()}
        output = model(**batch)
        probabilities.extend(torch.softmax(output.logits, dim=-1)[:, 1].detach().cpu().tolist())
        labels.extend(batch["labels"].detach().cpu().tolist())
    return _metrics(labels, probabilities)


def _write_split_metrics(output_dir: Path, split: str, metrics: SplitMetrics) -> None:
    save_json(metrics.as_dict(), output_dir / f"{split}_metrics.json")
    save_dataframe_csv(
        pd.DataFrame(
            metrics.confusion_matrix,
            index=["actual_mt", "actual_reference"],
            columns=["predicted_mt", "predicted_reference"],
        ).rename_axis("actual"),
        output_dir / f"{split}_confusion_matrix.csv",
    )


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise PipelineError(f"Expected {path} to contain a JSON object.")
    return value


def train_classifier(
    dataset_key: str,
    classifier_config_path: str | Path = "configs/classifier.yaml",
    run_name: str = "default",
    data_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
    device_name: str | None = None,
) -> dict[str, str]:
    """Train one target-side HT-versus-MT classifier."""
    config = load_classifier_config(classifier_config_path)["classifier"]
    if dataset_key not in config["backbones"]:
        raise PipelineError(f"No classifier backbone configured for {dataset_key}.")
    train_config = config["training"]
    seed = int(train_config["seed"])
    _set_seed(seed)

    data_root = Path(data_dir or config["output_dir"]) / dataset_key
    output_root = Path(results_dir or config["results_dir"]) / dataset_key / run_name
    output_root.mkdir(parents=True, exist_ok=True)
    source_pairs_root = Path(config["source_pairs_dir"]) / dataset_key
    records = {
        split: _load_jsonl(data_root / f"{split}.jsonl") for split in ("train", "dev", "test")
    }
    datasets = {split: TargetTextDataset(value) for split, value in records.items()}
    backbone = str(config["backbones"][dataset_key])
    token = get_hf_token()
    tokenizer = AutoTokenizer.from_pretrained(backbone, token=token)
    model = AutoModelForSequenceClassification.from_pretrained(backbone, num_labels=2, token=token)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    train_loader = _make_loader(
        datasets["train"],
        tokenizer,
        int(train_config["batch_size"]),
        int(train_config["max_length"]),
        True,
    )
    dev_loader = _make_loader(
        datasets["dev"],
        tokenizer,
        int(train_config["eval_batch_size"]),
        int(train_config["max_length"]),
        False,
    )
    test_loader = _make_loader(
        datasets["test"],
        tokenizer,
        int(train_config["eval_batch_size"]),
        int(train_config["max_length"]),
        False,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    total_steps = len(train_loader) * int(train_config["epochs"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * float(train_config["warmup_ratio"])),
        num_training_steps=total_steps,
    )

    best_dev_f1 = -1.0
    epoch_metrics: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, int(train_config["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {name: value.to(device) for name, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(output.loss.detach().cpu())
        dev_metrics = evaluate_classifier(model, dev_loader, device)
        epoch_metrics.append(
            {
                "epoch": epoch,
                "mean_train_loss": total_loss / max(len(train_loader), 1),
                "dev": dev_metrics.as_dict(),
            }
        )
        logger.info("%s epoch %d dev F1 %.4f", dataset_key, epoch, dev_metrics.f1)
        if dev_metrics.f1 > best_dev_f1:
            best_dev_f1 = dev_metrics.f1
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }

    if best_state is None:  # pragma: no cover - train loader cannot be empty after validation
        raise PipelineError("No classifier checkpoint was produced.")
    model.load_state_dict(best_state)
    model.to(device)
    dev_metrics = evaluate_classifier(model, dev_loader, device)
    test_metrics = evaluate_classifier(model, test_loader, device)
    model.save_pretrained(output_root / "best_model")
    tokenizer.save_pretrained(output_root / "best_model")
    _write_split_metrics(output_root, "dev", dev_metrics)
    _write_split_metrics(output_root, "test", test_metrics)
    metadata = {
        "task": "target-side reference-likeness / translationese classification",
        "dataset_key": dataset_key,
        "backbone": backbone,
        "input": {"fields": ["text"], "english_source_included": False},
        "labels": {"1": "human/reference-like target text", "0": "MT-like target text"},
        "split_sizes": {split: len(dataset) for split, dataset in datasets.items()},
        "source_pair_selection": _load_optional_json(source_pairs_root / "overlap_audit.json"),
        "negative_generation": _load_optional_json(data_root / "negative_generation_report.json"),
        "training": dict(train_config),
        "device": str(device),
        "epochs": epoch_metrics,
        "best_dev": dev_metrics.as_dict(),
        "test": test_metrics.as_dict(),
    }
    metadata_path = output_root / "run_metadata.json"
    save_json(metadata, metadata_path)
    return {
        "model_dir": str(output_root / "best_model"),
        "dev_metrics": str(output_root / "dev_metrics.json"),
        "test_metrics": str(output_root / "test_metrics.json"),
        "metadata": str(metadata_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a target-side translationese classifier.")
    parser.add_argument("--dataset", required=True, choices=("en_eu", "en_ca"))
    parser.add_argument("--classifier-config", default="configs/classifier.yaml")
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--device", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    paths = train_classifier(
        dataset_key=args.dataset,
        classifier_config_path=args.classifier_config,
        run_name=args.run_name,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        device_name=args.device,
    )
    logger.info("Classifier training complete: %s", paths)


if __name__ == "__main__":
    main()
