"""Create reviewable Vecalign candidates from selected parallel EPUB sections."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from zipfile import ZipFile

from src.data.prepare_woolf_final_test import (
    _blocks,
    _embed,
    _run,
    _sentence_records,
    _write_candidates,
    _write_sentence_records,
)


class EpubAlignmentError(RuntimeError):
    """Raised when selected EPUB members cannot be aligned."""


def _resolve_members(epub_path: Path, explicit: list[str], pattern: str | None) -> list[str]:
    if explicit:
        return explicit
    if pattern is None:
        raise EpubAlignmentError("Provide one or more --*-member values or a --*-member-pattern.")
    with ZipFile(epub_path) as archive:
        members = sorted(name for name in archive.namelist() if Path(name).match(pattern))
    if not members:
        raise EpubAlignmentError(f"No EPUB members in {epub_path} match {pattern!r}.")
    return members


def _extract_paragraphs(epub_path: Path, members: list[str], start_after: str | None) -> list[dict[str, object]]:
    paragraphs: list[dict[str, object]] = []
    started = start_after is None
    marker = start_after.casefold() if start_after else None
    for chapter, member in enumerate(members, start=1):
        paragraph_index = 0
        for tag, text in _blocks(epub_path, member):
            if not started and text.casefold() == marker:
                started = True
                continue
            if started and tag == "p":
                paragraph_index += 1
                paragraphs.append({"chapter": chapter, "paragraph": paragraph_index, "text": text})
    if not paragraphs:
        raise EpubAlignmentError(f"No paragraphs extracted from {epub_path}.")
    return paragraphs


def _write_records(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def _write_manifest(output_root: Path, **payload: object) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def align_epub_sections(
    source_epub: Path,
    target_epub: Path,
    source_language: str,
    target_language: str,
    source_members: list[str],
    target_members: list[str],
    output_root: Path,
    embedding_model: str,
    source_start_after: str | None = None,
    target_start_after: str | None = None,
) -> Path:
    source_paragraphs = _extract_paragraphs(source_epub, source_members, source_start_after)
    target_paragraphs = _extract_paragraphs(target_epub, target_members, target_start_after)
    _write_records(output_root / "paragraphs" / f"{source_language}.jsonl", source_paragraphs)
    _write_records(output_root / "paragraphs" / f"{target_language}.jsonl", target_paragraphs)

    source_sentences = _sentence_records(source_paragraphs, source_language)
    target_sentences = _sentence_records(target_paragraphs, target_language)
    sentence_dir = output_root / "sentences"
    _write_sentence_records(sentence_dir / f"{source_language}.jsonl", source_sentences)
    _write_sentence_records(sentence_dir / f"{target_language}.jsonl", target_sentences)
    source_text = sentence_dir / f"{source_language}.txt"
    target_text = sentence_dir / f"{target_language}.txt"
    source_text.write_text("\n".join(str(row["text"]) for row in source_sentences) + "\n", encoding="utf-8")
    target_text.write_text("\n".join(str(row["text"]) for row in target_sentences) + "\n", encoding="utf-8")

    work_dir = output_root / "vecalign"
    work_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, tuple[Path, Path]] = {}
    for language, sentence_path in ((source_language, source_text), (target_language, target_text)):
        overlap = work_dir / f"{language}.overlap.txt"
        embedding = work_dir / f"{language}.overlap.emb"
        _run([sys.executable, "-m", "vecalign.overlap", "-i", str(sentence_path), "-o", str(overlap), "-n", "4"])
        _embed(overlap, embedding, embedding_model)
        paths[language] = (overlap, embedding)

    alignment_path = work_dir / f"{source_language}_{target_language}.alignments.txt"
    _run(
        [
            sys.executable, "-m", "vecalign.vecalign", "--alignment_max_size", "4",
            "--src", str(source_text.resolve()), "--tgt", str(target_text.resolve()),
            "--src_embed", str(paths[source_language][0].resolve()), str(paths[source_language][1].resolve()),
            "--tgt_embed", str(paths[target_language][0].resolve()), str(paths[target_language][1].resolve()),
        ],
        stdout=alignment_path,
    )
    candidate_path = output_root / "candidates" / f"{source_language}_{target_language}_vecalign.tsv"
    _write_candidates(candidate_path, source_sentences, target_sentences, alignment_path)
    _write_manifest(
        output_root,
        source_epub=str(source_epub), target_epub=str(target_epub),
        source_members=source_members, target_members=target_members,
        source_language=source_language, target_language=target_language,
        embedding_model=embedding_model, candidate_path=str(candidate_path), review_status="pending",
    )
    return candidate_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Align selected EPUB sections with Vecalign.")
    parser.add_argument("--source-epub", type=Path, required=True)
    parser.add_argument("--target-epub", type=Path, required=True)
    parser.add_argument("--source-lang", required=True)
    parser.add_argument("--target-lang", required=True)
    parser.add_argument("--source-member", action="append", default=[])
    parser.add_argument("--target-member", action="append", default=[])
    parser.add_argument("--source-member-pattern", default=None)
    parser.add_argument("--target-member-pattern", default=None)
    parser.add_argument("--source-start-after", default=None)
    parser.add_argument("--target-start-after", default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--embedding-model", default="sentence-transformers/LaBSE")
    args = parser.parse_args()
    candidate_path = align_epub_sections(
        args.source_epub, args.target_epub, args.source_lang, args.target_lang,
        _resolve_members(args.source_epub, args.source_member, args.source_member_pattern),
        _resolve_members(args.target_epub, args.target_member, args.target_member_pattern),
        args.output_root, args.embedding_model, args.source_start_after, args.target_start_after,
    )
    print(f"Wrote reviewable Vecalign candidates to {candidate_path}")


if __name__ == "__main__":
    main()
