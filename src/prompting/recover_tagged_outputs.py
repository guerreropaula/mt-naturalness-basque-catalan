"""Recover leading tagged translations from P2/P3 outputs with trailing commentary."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.prompting.p2.run import _load_p2_language_model, _validate_refinement_prediction
from src.utils.io import save_json, save_jsonl


class TaggedRecoveryError(RuntimeError):
    """Raised when a tagged-output recovery cannot be applied safely."""


_FIELD_SETS = {
    "p2": {
        "accepted": "refinement_accepted",
        "reason": "refinement_rejection_reason",
        "raw": "raw_refinement_prediction",
        "fallback": "draft_prediction",
        "metadata_key": "refinement_validation",
        "fallback_label": "draft",
    },
    "p3": {
        "accepted": "proofreading_accepted",
        "reason": "proofreading_rejection_reason",
        "raw": "raw_proofreading_prediction",
        "fallback": "refined_translation",
        "metadata_key": "proofreading_validation",
        "fallback_label": "refined translation",
    },
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def recover_tagged_outputs(
    predictions_path: str | Path,
    *,
    experiment: str,
    target_lang: str,
    preprocessing_config_path: str | Path = "configs/preprocessing.yaml",
    apply: bool = False,
) -> dict[str, Any]:
    """Recover rejected rows whose first tagged translation passes validation.

    Raw outputs remain unchanged. With ``apply=True``, the original prediction
    file is backed up before recovered final predictions replace fallback values.
    """
    if experiment not in _FIELD_SETS:
        raise TaggedRecoveryError(f"experiment must be one of {sorted(_FIELD_SETS)}")
    fields = _FIELD_SETS[experiment]
    path = Path(predictions_path)
    records = _load_jsonl(path)
    if not records:
        raise TaggedRecoveryError(f"Prediction file is empty: {path}")
    required = {
        "sentence_id",
        "prediction",
        fields["accepted"],
        fields["reason"],
        fields["raw"],
        fields["fallback"],
    }
    missing = sorted(column for column in required if any(column not in record for record in records))
    if missing:
        raise TaggedRecoveryError(f"Prediction file is missing required fields: {', '.join(missing)}")

    language_model, threshold = _load_p2_language_model(preprocessing_config_path)
    recovered_records: list[dict[str, Any]] = []
    recovery_rejections: Counter[str] = Counter()
    for index, record in enumerate(records):
        updated = dict(record)
        if not bool(record[fields["accepted"]]):
            recovered, accepted, reason = _validate_refinement_prediction(
                str(record[fields["raw"]]),
                str(record[fields["fallback"]]),
                target_lang=target_lang,
                language_model=language_model,
                language_confidence_threshold=threshold,
            )
            if accepted:
                updated["prediction"] = recovered
                updated[fields["accepted"]] = True
                updated[fields["reason"]] = None
                updated["recovered_leading_translation_block"] = True
                recovered_records.append(updated)
            else:
                recovery_rejections[str(reason)] += 1
                updated.setdefault("recovered_leading_translation_block", False)
        else:
            updated.setdefault("recovered_leading_translation_block", False)
        records[index] = updated

    recovered_count = len(recovered_records)
    result: dict[str, Any] = {
        "predictions": str(path),
        "experiment": experiment,
        "target_lang": target_lang,
        "rows": len(records),
        "recovered_rows": recovered_count,
        "still_rejected_after_recovery": dict(recovery_rejections.most_common()),
        "applied": apply,
    }
    if not apply or not recovered_count:
        return result

    backup_path = path.with_name("predictions.before_tag_recovery.jsonl")
    if backup_path.exists():
        raise TaggedRecoveryError(f"Recovery backup already exists: {backup_path}")
    save_jsonl(_load_jsonl(path), backup_path)
    records.sort(key=lambda record: str(record["sentence_id"]))
    save_jsonl(records, path)

    metadata_path = path.parent / "run_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        accepted = sum(bool(record[fields["accepted"]]) for record in records)
        rejected = Counter(
            str(record[fields["reason"]])
            for record in records
            if not bool(record[fields["accepted"]])
        )
        total = len(records)
        metadata[fields["metadata_key"]] = {
            "accepted": accepted,
            "accepted_display": f"{experiment.upper()} accepted: {accepted} / {total}",
            f"fallback_to_{fields['fallback_label'].replace(' ', '_')}": total - accepted,
            "fallback_display": f"Fallback to {fields['fallback_label']}: {total - accepted} / {total}",
            "recovered_leading_translation_block": recovered_count,
            "rejection_reasons": dict(rejected.most_common()),
        }
        metadata["tag_recovery"] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "backup_predictions": str(backup_path),
            "recovered_rows": recovered_count,
            "policy": "Use the first leading <translation> block; ignore trailing content.",
            "analysis_rebuild_required": True,
        }
        save_json(metadata, metadata_path)

    marker_path = path.parent / "analysis_rebuild_required.json"
    save_json(
        {
            "reason": "Recovered leading tagged translations after trailing commentary.",
            "predictions_path": str(path),
            "backup_predictions": str(backup_path),
            "recovered_rows": recovered_count,
            "rebuild_full_analysis": True,
        },
        marker_path,
    )
    result.update({"backup": str(backup_path), "analysis_marker": str(marker_path)})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, action="append")
    parser.add_argument("--experiment", required=True, choices=sorted(_FIELD_SETS))
    parser.add_argument("--target-lang", required=True)
    parser.add_argument("--preprocessing-config", default="configs/preprocessing.yaml")
    parser.add_argument("--apply", action="store_true", help="Write recovered predictions and a backup.")
    args = parser.parse_args()
    for predictions_path in args.predictions:
        print(
            recover_tagged_outputs(
                predictions_path,
                experiment=args.experiment,
                target_lang=args.target_lang,
                preprocessing_config_path=args.preprocessing_config,
                apply=args.apply,
            )
        )


if __name__ == "__main__":
    main()
