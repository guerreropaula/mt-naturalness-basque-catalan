"""Repair historical P2 files that mixed old and forced rerun predictions."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.io import save_json, save_jsonl


class P2RepairError(RuntimeError):
    """Raised when a duplicate P2 file cannot be repaired without ambiguity."""


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _uses_tagged_refinement_prompt(record: dict[str, Any]) -> bool:
    prompt = str(record.get("refinement_prompt", "")).lower()
    return "<translation>" in prompt and "</translation>" in prompt


def repair_p2_predictions(path: str | Path) -> dict[str, Any]:
    """Keep the one tagged-prompt record per sentence and retain a backup."""
    predictions_path = Path(path)
    records = _load_jsonl(predictions_path)
    if not records:
        raise P2RepairError(f"Prediction file is empty: {predictions_path}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        sentence_id = record.get("sentence_id")
        if sentence_id is None:
            raise P2RepairError(f"Record without sentence_id in {predictions_path}")
        grouped[str(sentence_id)].append(record)

    repaired: list[dict[str, Any]] = []
    ambiguous: dict[str, int] = {}
    for sentence_id, candidates in grouped.items():
        tagged = [record for record in candidates if _uses_tagged_refinement_prompt(record)]
        if len(candidates) == 1:
            repaired.append(candidates[0])
        elif len(tagged) == 1:
            repaired.append(tagged[0])
        else:
            ambiguous[sentence_id] = len(candidates)
    if ambiguous:
        raise P2RepairError(
            f"Cannot select a single tagged P2 record for {len(ambiguous)} sentence IDs. "
            "No files were changed."
        )

    if len(repaired) == len(records):
        raise P2RepairError("No duplicate P2 records were found; no repair was applied.")
    repaired.sort(key=lambda record: str(record["sentence_id"]))
    backup_path = predictions_path.with_name("predictions.before_p2_dedup.jsonl")
    if backup_path.exists():
        raise P2RepairError(f"Backup already exists: {backup_path}")
    save_jsonl(records, backup_path)
    save_jsonl(repaired, predictions_path)

    output_dir = predictions_path.parent
    metadata_path = output_dir / "run_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        validated = [record for record in repaired if "refinement_accepted" in record]
        accepted = sum(bool(record["refinement_accepted"]) for record in validated)
        rejected = Counter(
            str(record["refinement_rejection_reason"])
            for record in validated
            if not record["refinement_accepted"]
        )
        metadata["generated_rows"] = len(repaired)
        metadata["refinement_validation"] = {
            "accepted": accepted,
            "accepted_display": f"P2 refinement accepted: {accepted} / {len(validated)}",
            "fallback_to_draft": len(validated) - accepted,
            "fallback_display": f"P2 fallback to draft: {len(validated) - accepted} / {len(validated)}",
            "rejection_reasons": dict(rejected.most_common()),
        }
        metadata["repair"] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "reason": "Removed stale pre-tag P2 outputs duplicated by force plus resume.",
            "backup_predictions": str(backup_path),
            "input_rows": len(records),
            "retained_rows": len(repaired),
            "selection": "One record per sentence_id; prefer the tagged refinement prompt.",
            "analysis_rebuild_required": True,
        }
        save_json(metadata, metadata_path)

    marker_path = output_dir / "analysis_rebuild_required.json"
    save_json(
        {
            "reason": "Prediction rows were deduplicated after the prior analysis.",
            "predictions_path": str(predictions_path),
            "rows_before": len(records),
            "rows_after": len(repaired),
            "rebuild_analysis_only": True,
        },
        marker_path,
    )
    return {
        "predictions": str(predictions_path),
        "backup": str(backup_path),
        "analysis_marker": str(marker_path),
        "rows_before": len(records),
        "rows_after": len(repaired),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair duplicate P2 prediction JSONL files.")
    parser.add_argument("--predictions", required=True, action="append")
    args = parser.parse_args()
    for path in args.predictions:
        print(repair_p2_predictions(path))


if __name__ == "__main__":
    main()
