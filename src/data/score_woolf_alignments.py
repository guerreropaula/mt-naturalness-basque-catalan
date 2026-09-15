"""Score full-book Woolf Vecalign candidates and export a high-confidence test."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.data.preprocessing import _score_alignment_batch


class WoolfAlignmentScoringError(RuntimeError):
    """Raised when full-book candidates cannot form a reliable test."""


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _character_ratio(source: str, target: str) -> float:
    source_length = len("".join(source.split()))
    target_length = len("".join(target.split()))
    if source_length == 0 or target_length == 0:
        return 0.0
    return target_length / source_length


def score_and_select(
    output_root: Path,
    target_language: str,
    threshold: float,
    min_character_ratio: float,
    max_character_ratio: float,
    model_name: str,
    batch_size: int,
) -> dict[str, Any]:
    candidate_path = output_root / "candidates" / f"en_{target_language}_vecalign.tsv"
    if not candidate_path.exists():
        raise WoolfAlignmentScoringError(f"Missing Vecalign candidates: {candidate_path}")
    with candidate_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    eligible_indexes = [index for index, row in enumerate(rows) if row["alignment_type"] == "1-1"]
    if not eligible_indexes:
        raise WoolfAlignmentScoringError(f"No 1-1 candidates in {candidate_path}")

    scores = _score_alignment_batch(
        [rows[index]["source"] for index in eligible_indexes],
        [rows[index]["target"] for index in eligible_indexes],
        {"model": model_name, "batch_size": batch_size, "gpus": 1, "progress_bar": True},
    )
    for index, score in zip(eligible_indexes, scores, strict=True):
        rows[index]["cometkiwi_score"] = f"{float(score):.8f}"

    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen_pairs: set[tuple[str, str]] = set()
    for row in rows:
        reason = "accepted"
        source = row["source"].strip()
        target = row["target"].strip()
        ratio = _character_ratio(source, target)
        row["character_ratio"] = f"{ratio:.6f}"
        if row["alignment_type"] != "1-1":
            reason = "non_1_to_1"
        elif not source or not target or not any(char.isalpha() for char in source + target):
            reason = "empty_or_nonlinguistic"
        elif ratio < min_character_ratio or ratio > max_character_ratio:
            reason = "character_ratio_out_of_range"
        elif float(row["cometkiwi_score"]) < threshold:
            reason = "cometkiwi_below_threshold"
        elif (source, target) in seen_pairs:
            reason = "duplicate_pair"
        row["automatic_status"] = reason
        counts[reason] += 1
        if reason != "accepted":
            continue
        seen_pairs.add((source, target))
        selected.append(
            {
                "id": f"woolf_full_en_{target_language}_{len(selected) + 1:04d}",
                "sentence_id": f"woolf_full_en_{target_language}_{len(selected) + 1:04d}",
                "source": source,
                "target": target,
                "part": int(row["part"]),
                "source_chapters": row["source_chapters"],
                "target_chapters": row["target_chapters"],
                "alignment_type": row["alignment_type"],
                "vecalign_cost": float(row["vecalign_cost"]),
                "cometkiwi_score": float(row["cometkiwi_score"]),
                "character_ratio": ratio,
                "corpus": f"woolf_to_the_lighthouse_full_en_{target_language}",
                "alignment_selection": "vecalign_1_to_1_cometkiwi_high_confidence",
            }
        )
    if not selected:
        raise WoolfAlignmentScoringError("No candidates passed the high-confidence filter.")

    scored_path = output_root / "scored" / f"en_{target_language}_vecalign_scored.tsv"
    scored_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    for field in ("cometkiwi_score", "character_ratio", "automatic_status"):
        if field not in fields:
            fields.append(field)
    with scored_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    test_path = output_root / "automatic_high_confidence" / f"en_{target_language}.jsonl"
    _write_jsonl(test_path, selected)
    accepted_scores = [row["cometkiwi_score"] for row in selected]
    metadata = {
        "source_candidates": str(candidate_path),
        "scored_candidates": str(scored_path),
        "test_file": str(test_path),
        "target_language": target_language,
        "selection_status": "automatic_high_confidence_not_manual_review",
        "sentence_level_only": True,
        "cometkiwi_model": model_name,
        "cometkiwi_threshold": threshold,
        "character_ratio_range": [min_character_ratio, max_character_ratio],
        "candidate_count": len(rows),
        "selection_counts": dict(counts),
        "retained_pair_count": len(selected),
        "retained_score_summary": {
            "min": min(accepted_scores),
            "mean": sum(accepted_scores) / len(accepted_scores),
            "max": max(accepted_scores),
        },
    }
    metadata_path = output_root / "automatic_high_confidence" / "selection_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path = output_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        alignment_status="automatic_high_confidence_ready",
        automatic_high_confidence_pair_count=len(selected),
        automatic_high_confidence_file=str(test_path),
        manual_review_status="pending",
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-language", choices=("ca", "eu"), required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--min-character-ratio", type=float, default=0.25)
    parser.add_argument("--max-character-ratio", type=float, default=4.0)
    parser.add_argument("--model", default="Unbabel/wmt23-cometkiwi-da-xl")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    output_root = args.output_root or Path(f"data/woolf_test/full_en_{args.target_language}")
    result = score_and_select(
        output_root, args.target_language, args.threshold, args.min_character_ratio,
        args.max_character_ratio, args.model, args.batch_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
