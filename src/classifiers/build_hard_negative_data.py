"""Build classifier datasets from deterministic P4 SFT hard negatives.

This module extracts the portable data logic that was originally embedded in
cluster job files. It never loads a model. Run ``prepare``, generate P4 outputs
with :mod:`src.sft.evaluate`, and then run ``combine``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

DATASETS = ("en_eu", "en_ca")
SPLITS = ("train", "dev", "test")
DEFAULT_MODELS = ("latxa_8b_instruct", "salamandrata_7b_instruct")


class HardNegativeDataError(RuntimeError):
    """Raised when a hard-negative split is incomplete or inconsistent."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise HardNegativeDataError(f"Missing JSONL file: {path}")
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise HardNegativeDataError(f"Empty JSONL file: {path}")
    return rows


def _write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _require_pair(row: dict[str, Any], *, split: str, index: int) -> None:
    missing = [field for field in ("source_id", "source", "target") if not str(row.get(field, "")).strip()]
    if missing:
        raise HardNegativeDataError(
            f"{split} row {index} is missing required values: {', '.join(missing)}"
        )


def protocol_names(dataset: str, unit: str) -> tuple[str, str]:
    if dataset not in DATASETS:
        raise HardNegativeDataError(f"Unsupported dataset: {dataset}")
    if unit == "chunk":
        return f"{dataset}_sft5", f"{dataset}_sft5_chunks"
    if unit == "sentence":
        return f"{dataset}_sft_hard_sentence_v3", f"{dataset}_sft_hard_sentence_v3"
    raise HardNegativeDataError(f"Unsupported unit: {unit}")


def prepare_inputs(
    dataset: str,
    unit: str,
    *,
    pairs_root: str | Path = "data/classifier_source_pairs",
    output_root: str | Path | None = None,
    models: tuple[str, ...] = DEFAULT_MODELS,
    chunk_size: int = 5,
) -> dict[str, Any]:
    """Prepare deterministic model-assigned P4 inputs from held-out pairs."""
    if not models:
        raise HardNegativeDataError("At least one negative model is required.")
    if chunk_size < 1:
        raise HardNegativeDataError("chunk_size must be positive.")
    if unit == "chunk" and chunk_size != 5:
        raise HardNegativeDataError("The thesis chunk protocol uses exactly five sentences.")

    id_prefix, output_dataset_key = protocol_names(dataset, unit)
    pairs_dir = Path(pairs_root) / dataset
    root = Path(output_root or ("data/classifier_chunks/sft5" if unit == "chunk" else "data/classifier_sentence/sft_hard_v3"))
    report: dict[str, Any] = {
        "dataset_key": dataset,
        "unit": "five_sentence_chunk" if unit == "chunk" else "sentence",
        "negative_models": list(models),
        "negative_stage": "P4 SFT deterministic greedy translation",
        "negative_assignment": "round-robin in stable selected-pair order",
        "output_dataset_key": output_dataset_key,
        "splits": {},
    }

    for split in SPLITS:
        rows = _read_jsonl(pairs_dir / f"{split}.jsonl")
        for index, row in enumerate(rows, start=1):
            _require_pair(row, split=split, index=index)

        records: list[dict[str, Any]] = []
        discarded = 0
        if unit == "chunk":
            discarded = len(rows) % chunk_size
            usable = len(rows) - discarded
            for start in range(0, usable, chunk_size):
                group = rows[start : start + chunk_size]
                record_index = start // chunk_size + 1
                records.append(
                    {
                        "id": f"{id_prefix}_{split}_{record_index:06d}",
                        "source": "\n\n".join(str(row["source"]).strip() for row in group),
                        "target": "\n\n".join(str(row["target"]).strip() for row in group),
                        "source_ids": [str(row["source_id"]) for row in group],
                        "original_indices": [int(row.get("original_index", start + offset)) for offset, row in enumerate(group)],
                        "split": split,
                        "chunk_size_sentences": chunk_size,
                    }
                )
        else:
            for index, row in enumerate(rows, start=1):
                records.append(
                    {
                        "id": f"{id_prefix}_{split}_{index:06d}",
                        "source": str(row["source"]).strip(),
                        "target": str(row["target"]).strip(),
                        "source_id": str(row["source_id"]),
                        "original_index": int(row.get("original_index", index - 1)),
                        "domain": row.get("domain"),
                        "positive_cometkiwi_score": row.get("positive_cometkiwi_score"),
                    }
                )

        assignments = Counter()
        inputs_by_model = {model: [] for model in models}
        for index, record in enumerate(records):
            model = models[index % len(models)]
            record["negative_model_key"] = model
            inputs_by_model[model].append(
                {key: record[key] for key in ("id", "source", "target")}
            )
            assignments[model] += 1
        for model, model_rows in inputs_by_model.items():
            _write_jsonl(model_rows, root / "p4_inputs" / model / f"{split}.jsonl")

        manifest_dir = "chunks" if unit == "chunk" else "sentences"
        _write_jsonl(records, root / dataset / manifest_dir / f"{split}.jsonl")

        report["splits"][split] = {
            "input_sentence_pairs": len(rows),
            "prepared_units": len(records),
            "discarded_remainder_pairs": discarded,
            "negative_model_counts": dict(assignments),
        }

    _write_json(report, root / dataset / f"{unit}_protocol_metadata.json")
    return report


def _prepared_records(root: Path, dataset: str, unit: str, split: str) -> list[dict[str, Any]]:
    if unit == "chunk":
        return _read_jsonl(root / dataset / "chunks" / f"{split}.jsonl")
    return _read_jsonl(root / dataset / "sentences" / f"{split}.jsonl")


def combine_outputs(
    dataset: str,
    unit: str,
    *,
    output_root: str | Path | None = None,
    models: tuple[str, ...] = DEFAULT_MODELS,
) -> dict[str, Any]:
    """Combine P4 predictions with references into balanced target-only data."""
    id_prefix, output_dataset_key = protocol_names(dataset, unit)
    root = Path(output_root or ("data/classifier_chunks/sft5" if unit == "chunk" else "data/classifier_sentence/sft_hard_v3"))
    target_language = dataset.rsplit("_", 1)[1]
    report: dict[str, Any] = {
        "dataset_key": dataset,
        "unit": "five_sentence_chunk" if unit == "chunk" else "sentence",
        "negative_stage": "P4 SFT deterministic greedy translation",
        "negative_models": list(models),
        "classifier_input": "target-side text only",
        "splits": {},
    }

    for split in SPLITS:
        prepared = _prepared_records(root, dataset, unit, split)
        expected = {str(row["id"]): row for row in prepared}
        predictions: dict[str, str] = {}
        prediction_models: dict[str, str] = {}
        for model in models:
            path = root / "p4_predictions" / output_dataset_key / model / split / "predictions.jsonl"
            for row in _read_jsonl(path):
                record_id = str(row.get("id") or row.get("sentence_id") or "")
                if not record_id or record_id in predictions:
                    raise HardNegativeDataError(f"Missing or duplicate prediction ID in {path}: {record_id!r}")
                predictions[record_id] = str(row.get("prediction", "")).strip()
                prediction_models[record_id] = model

        if set(predictions) != set(expected):
            missing = sorted(set(expected) - set(predictions))[:5]
            unexpected = sorted(set(predictions) - set(expected))[:5]
            raise HardNegativeDataError(
                f"Prediction IDs do not match {split}: missing={missing}, unexpected={unexpected}"
            )

        combined: list[dict[str, Any]] = []
        model_counts = Counter()
        for record_id, row in expected.items():
            prediction = predictions[record_id]
            lowered = prediction.lower()
            if not prediction or "<translation>" in lowered or "</translation>" in lowered:
                raise HardNegativeDataError(f"Invalid P4 negative for {record_id}.")
            model = prediction_models[record_id]
            provenance: dict[str, Any] = {
                "split": split,
                "language": target_language,
                "chunk_size_sentences": 5 if unit == "chunk" else 1,
            }
            if unit == "chunk":
                provenance["source_id"] = "|".join(str(value) for value in row["source_ids"])
                provenance["source_ids"] = row["source_ids"]
            else:
                provenance["source_id"] = str(row.get("source_id", record_id))
                for field in ("original_index", "domain", "positive_cometkiwi_score"):
                    if field in row and row[field] is not None:
                        provenance[field] = row[field]
            combined.extend(
                (
                    {
                        "id": f"{record_id}_ref",
                        "text": str(row["target"]).strip(),
                        "label": 1,
                        "origin": "reference_chunk" if unit == "chunk" else "reference_sentence",
                        **provenance,
                    },
                    {
                        "id": f"{record_id}_mt",
                        "text": prediction,
                        "label": 0,
                        "origin": "p4_sft_chunk" if unit == "chunk" else "p4_sft_sentence",
                        "mt_model_key": model,
                        **provenance,
                    },
                )
            )
            model_counts[model] += 1

        _write_jsonl(combined, root / dataset / f"{split}.jsonl")

        report["splits"][split] = {
            "reference_examples": len(expected),
            "mt_examples": len(expected),
            "balanced_examples": len(combined),
            "negative_model_counts": dict(model_counts),
        }

    _write_json(report, root / dataset / "negative_generation_report.json")
    return report


def _models(value: str) -> tuple[str, ...]:
    models = tuple(item.strip() for item in value.split(",") if item.strip())
    if not models:
        raise argparse.ArgumentTypeError("Provide at least one comma-separated model key.")
    return models


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "combine"))
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--unit", required=True, choices=("sentence", "chunk"))
    parser.add_argument("--pairs-root", default="data/classifier_source_pairs")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--models", type=_models, default=DEFAULT_MODELS)
    parser.add_argument("--chunk-size", type=int, default=5)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.stage == "prepare":
        result = prepare_inputs(
            args.dataset,
            args.unit,
            pairs_root=args.pairs_root,
            output_root=args.output_root,
            models=args.models,
            chunk_size=args.chunk_size,
        )
    else:
        result = combine_outputs(
            args.dataset,
            args.unit,
            output_root=args.output_root,
            models=args.models,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
