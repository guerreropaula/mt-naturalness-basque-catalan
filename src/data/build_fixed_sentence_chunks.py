"""Build fixed-size, non-overlapping chunks from an ordered parallel JSONL file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class FixedSentenceChunkError(RuntimeError):
    """Raised when fixed-size chunks cannot be constructed safely."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise FixedSentenceChunkError(f"Input file is empty: {path}")
    for index, row in enumerate(rows, start=1):
        missing = {field for field in ("source", "target") if not str(row.get(field, "")).strip()}
        if missing:
            raise FixedSentenceChunkError(
                f"Row {index} in {path} is missing non-empty fields: {sorted(missing)}"
            )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_fixed_sentence_chunks(
    input_path: str | Path,
    output_path: str | Path,
    *,
    id_prefix: str,
    chunk_size: int = 5,
    separator: str = "\n\n",
    require_complete: bool = True,
) -> dict[str, Any]:
    """Join consecutive parallel rows using the classifier's chunking protocol."""
    if chunk_size < 1:
        raise FixedSentenceChunkError("chunk_size must be positive.")
    source_path = Path(input_path)
    destination = Path(output_path)
    rows = _read_jsonl(source_path)
    remainder = len(rows) % chunk_size
    if remainder and require_complete:
        raise FixedSentenceChunkError(
            f"{len(rows)} rows are not divisible by chunk_size={chunk_size}; "
            "pass require_complete=False only when dropping the final remainder is intended."
        )

    usable_rows = len(rows) - remainder
    chunks: list[dict[str, Any]] = []
    cross_part_chunks = 0
    cross_source_chapter_chunks = 0
    cross_target_chapter_chunks = 0
    for start in range(0, usable_rows, chunk_size):
        group = rows[start : start + chunk_size]
        source_ids = [str(row.get("id") or row.get("sentence_id") or start + offset + 1) for offset, row in enumerate(group)]
        parts = [row.get("part") for row in group]
        source_chapters = [str(row.get("source_chapters", "")) for row in group]
        target_chapters = [str(row.get("target_chapters", "")) for row in group]
        crosses_part = len(set(parts)) > 1
        crosses_source_chapter = len(set(source_chapters)) > 1
        crosses_target_chapter = len(set(target_chapters)) > 1
        cross_part_chunks += int(crosses_part)
        cross_source_chapter_chunks += int(crosses_source_chapter)
        cross_target_chapter_chunks += int(crosses_target_chapter)
        chunk_id = f"{id_prefix}_{len(chunks) + 1:04d}"
        chunks.append(
            {
                "id": chunk_id,
                "sentence_id": chunk_id,
                "source": separator.join(str(row["source"]).strip() for row in group),
                "target": separator.join(str(row["target"]).strip() for row in group),
                "source_ids": source_ids,
                "chunk_size_sentences": chunk_size,
                "first_source_id": source_ids[0],
                "last_source_id": source_ids[-1],
                "parts": parts,
                "source_chapters": source_chapters,
                "target_chapters": target_chapters,
                "crosses_part_boundary": crosses_part,
                "crosses_source_chapter_boundary": crosses_source_chapter,
                "crosses_target_chapter_boundary": crosses_target_chapter,
                "chunking_protocol": "consecutive_nonoverlapping_rows_double_newline_separator",
            }
        )

    _write_jsonl(destination, chunks)
    metadata = {
        "input_file": str(source_path),
        "output_file": str(destination),
        "input_sentence_pairs": len(rows),
        "chunk_size_sentences": chunk_size,
        "complete_chunks": len(chunks),
        "discarded_remainder_pairs": remainder,
        "separator": "double_newline" if separator == "\n\n" else separator,
        "preserves_input_order": True,
        "overlapping_chunks": False,
        "cross_part_chunks": cross_part_chunks,
        "cross_source_chapter_chunks": cross_source_chapter_chunks,
        "cross_target_chapter_chunks": cross_target_chapter_chunks,
        "classifier_protocol_match": "same fixed size, order, and double-newline concatenation as classifier v2",
    }
    metadata_path = destination.with_name(f"{destination.stem}_metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--id-prefix", required=True)
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--allow-remainder", action="store_true")
    args = parser.parse_args()
    metadata = build_fixed_sentence_chunks(
        args.input,
        args.output,
        id_prefix=args.id_prefix,
        chunk_size=args.chunk_size,
        require_complete=not args.allow_remainder,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
