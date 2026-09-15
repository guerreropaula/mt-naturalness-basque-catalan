"""Select independent sentence-level Woolf tests from audited Vecalign output."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


class WoolfSentenceTestError(RuntimeError):
    """Raised when an audited bilingual sentence test cannot be built."""


def _character_ratio(source: str, target: str) -> float:
    source_length = len("".join(source.split()))
    target_length = len("".join(target.split()))
    if source_length == 0 or target_length == 0:
        return 0.0
    return target_length / source_length


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _first_in_book_order(
    records: list[dict[str, Any]], target_size: int
) -> tuple[list[dict[str, Any]], dict[int, int], dict[int, int]]:
    """Take the first eligible pairs in narrative order."""
    if target_size < 1:
        raise WoolfSentenceTestError("sampling.target_size must be positive.")
    if len(records) < target_size:
        raise WoolfSentenceTestError(
            f"Only {len(records)} eligible pairs are available; cannot select {target_size}."
        )
    ordered = sorted(records, key=lambda record: int(record["candidate_order"]))
    selected = ordered[:target_size]
    eligible_counts = Counter(int(record["part"]) for record in ordered)
    selected_counts = Counter(int(record["part"]) for record in selected)
    return selected, dict(sorted(eligible_counts.items())), dict(sorted(selected_counts.items()))


def select_sentence_test(
    output_root: Path,
    target_language: str,
    review_config_path: Path,
) -> dict[str, Any]:
    config = yaml.safe_load(review_config_path.read_text(encoding="utf-8"))
    selection = config["selection"]
    rejected = {
        str(item["source"]): str(item["reason"])
        for item in config.get("explicit_rejections", {}).get(target_language, [])
    }
    candidate_path = output_root / "candidates" / f"en_{target_language}_vecalign.tsv"
    with candidate_path.open(encoding="utf-8", newline="") as handle:
        candidates = list(csv.DictReader(handle, delimiter="\t"))

    eligible_records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    encountered_rejections: set[str] = set()
    for candidate_order, row in enumerate(candidates):
        source = row["source"].strip()
        target = row["target"].strip()
        ratio = _character_ratio(source, target)
        reason = "accepted"
        if row["alignment_type"] != str(selection["alignment_type"]):
            reason = "non_1_to_1"
        elif source in rejected:
            reason = rejected[source]
            encountered_rejections.add(source)
        elif not source or not target or not any(character.isalpha() for character in source + target):
            reason = "empty_or_nonlinguistic"
        elif ratio < float(selection["min_character_ratio"]) or ratio > float(selection["max_character_ratio"]):
            reason = "character_ratio_out_of_range"
        counts[reason] += 1
        if reason != "accepted":
            continue
        eligible_records.append(
            {
                "candidate_order": candidate_order,
                "source": source,
                "target": target,
                "part": int(row["part"]),
                "source_indexes": row["source_indexes"],
                "target_indexes": row["target_indexes"],
                "source_chapters": row["source_chapters"],
                "target_chapters": row["target_chapters"],
                "alignment_type": row["alignment_type"],
                "vecalign_cost": float(row["vecalign_cost"]),
                "character_ratio": ratio,
                "corpus": f"woolf_to_the_lighthouse_full_en_{target_language}",
                "selection_version": str(config["version"]),
            }
        )
    missing_rejections = set(rejected) - encountered_rejections
    if missing_rejections:
        raise WoolfSentenceTestError(
            "Configured rejected sources were not found: " + "; ".join(sorted(missing_rejections))
        )
    if not eligible_records:
        raise WoolfSentenceTestError("No sentence pairs passed the audited selection.")

    sampling = config["sampling"]
    records, eligible_part_counts, selected_part_counts = _first_in_book_order(
        eligible_records, int(sampling["target_size"])
    )
    for index, record in enumerate(records, start=1):
        record_id = f"woolf_full_en_{target_language}_{index:04d}"
        record["id"] = record_id
        record["sentence_id"] = record_id
        del record["candidate_order"]

    output_path = output_root / "sentence_test" / f"en_{target_language}.jsonl"
    _write_jsonl(output_path, records)
    metadata = {
        "selection_version": config["version"],
        "target_language": target_language,
        "source_candidates": str(candidate_path),
        "test_file": str(output_path),
        "alignment_scope": "part_bounded_independent_bilingual",
        "sentence_level_only": True,
        "selection_status": "automatic_vecalign_with_explicit_alignment_rejections",
        "manual_bilingual_review_status": "pending",
        "cometkiwi_role": "diagnostic_only_not_selection",
        "candidate_count": len(candidates),
        "selection_counts": dict(counts),
        "eligible_pair_count": len(eligible_records),
        "retained_pair_count": len(records),
        "sampling": {
            "target_size": int(sampling["target_size"]),
            "strategy": sampling["strategy"],
            "eligible_part_counts": eligible_part_counts,
            "selected_part_counts": selected_part_counts,
        },
        "review_config": str(review_config_path),
    }
    metadata_path = output_root / "sentence_test" / "selection_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path = output_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        sentence_test_status=metadata["selection_status"],
        sentence_test_pair_count=len(records),
        sentence_test_file=str(output_path),
        manual_bilingual_review_status="pending",
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-language", choices=("ca", "eu"), required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--review-config", type=Path, default=Path("configs/woolf_alignment_review.yaml"))
    args = parser.parse_args()
    output_root = args.output_root or Path(f"data/woolf_test/full_en_{args.target_language}")
    result = select_sentence_test(output_root, args.target_language, args.review_config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
