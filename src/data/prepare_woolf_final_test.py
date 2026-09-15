"""Prepare the shared To the Lighthouse Chapter 1-3 external test excerpt."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile


class WoolfPreparationError(RuntimeError):
    """Raised when the local editions cannot form the declared shared excerpt."""


class _BlockParser(HTMLParser):
    _BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._stack: list[tuple[str, list[str]]] = []
        self.blocks: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self._BLOCK_TAGS:
            self._stack.append((tag.lower(), []))

    def handle_endtag(self, tag: str) -> None:
        if self._stack and self._stack[-1][0] == tag.lower():
            block_tag, parts = self._stack.pop()
            text = " ".join("".join(parts).split())
            if text:
                self.blocks.append((block_tag, text))

    def handle_data(self, data: str) -> None:
        if self._stack:
            self._stack[-1][1].append(data)


def _blocks(epub_path: Path, member: str) -> list[tuple[str, str]]:
    with ZipFile(epub_path) as archive:
        raw = archive.read(member).decode("utf-8", errors="replace")
    parser = _BlockParser()
    parser.feed(raw)
    return parser.blocks


def _english_chapters(epub_path: Path) -> dict[int, list[str]]:
    blocks = _blocks(epub_path, "OEBPS/Text/OML004.html")
    in_window = False
    active_chapter: int | None = None
    chapters = {1: [], 2: [], 3: []}
    for tag, text in blocks:
        if tag == "h1" and text.casefold() == "part 1. the window":
            in_window = True
            continue
        if not in_window:
            continue
        if tag == "h2":
            match = re.fullmatch(r"chapter\s+(\d+)", text, flags=re.IGNORECASE)
            if match is not None:
                chapter = int(match.group(1))
                active_chapter = chapter if chapter in chapters else None
                if chapter > 3:
                    break
                continue
        if tag == "p" and active_chapter is not None:
            chapters[active_chapter].append(text)
    return chapters


def _catalan_chapters(epub_path: Path) -> dict[int, list[str]]:
    chapters: dict[int, list[str]] = {}
    for chapter in (1, 2, 3):
        member = f"OEBPS/Text/cap-{chapter:02d}.xhtml"
        chapters[chapter] = [text for tag, text in _blocks(epub_path, member) if tag == "p"]
    return chapters


def _basque_chapters(epub_path: Path) -> dict[int, list[str]]:
    blocks = _blocks(epub_path, "Ops/1.html")
    in_window = False
    active_chapter: int | None = None
    chapters = {1: [], 2: [], 3: []}
    for tag, text in blocks:
        if tag != "p":
            continue
        if text.casefold() == "leihoa":
            in_window = True
            continue
        if not in_window:
            continue
        if text.isdigit():
            chapter = int(text)
            if chapter > 3:
                break
            active_chapter = chapter
            continue
        if active_chapter in chapters:
            chapters[active_chapter].append(text)
    return chapters


def _validate_chapters(chapters: dict[int, list[str]], language: str) -> None:
    missing = [str(chapter) for chapter, paragraphs in chapters.items() if not paragraphs]
    if missing:
        raise WoolfPreparationError(f"{language} has no paragraphs for Chapter(s): {', '.join(missing)}")


def _write_paragraphs(path: Path, chapters: dict[int, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chapter in (1, 2, 3):
            for paragraph_index, text in enumerate(chapters[chapter], start=1):
                handle.write(json.dumps({"chapter": chapter, "paragraph": paragraph_index, "text": text}, ensure_ascii=False))
                handle.write("\n")


def extract_shared_excerpt(input_root: Path, output_root: Path) -> dict[str, dict[int, int]]:
    editions = {
        "en": _english_chapters(input_root / "en" / "To the Lighthouse - Virginia Woolf - EPUB.epub"),
        "ca": _catalan_chapters(input_root / "ca" / "cap_al_far_mostra.epub"),
        "eu": _basque_chapters(input_root / "eu" / "Virginia Woolf, Farorantz.epub"),
    }
    for language, chapters in editions.items():
        _validate_chapters(chapters, language)
        _write_paragraphs(output_root / "paragraphs" / f"{language}.jsonl", chapters)

    counts = {
        language: {chapter: len(paragraphs) for chapter, paragraphs in chapters.items()}
        for language, chapters in editions.items()
    }
    manifest = {
        "name": "woolf_shared_chapters_1_3",
        "purpose": "external_final_literary_test",
        "included": "Part I, The Window, Chapters 1-3 only",
        "excluded": "front matter, notes, and all text beyond the Catalan sample",
        "languages": ["en", "ca", "eu"],
        "paragraph_counts": counts,
        "alignment_status": "not_aligned",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return counts


def _load_paragraphs(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sentence_records(paragraphs: Iterable[dict[str, object]], language: str) -> list[dict[str, object]]:
    import stanza

    pipeline = stanza.Pipeline(
        lang=language,
        processors="tokenize",
        tokenize_no_ssplit=False,
        use_gpu=False,
        verbose=False,
    )
    records: list[dict[str, object]] = []
    for paragraph in paragraphs:
        document = pipeline(str(paragraph["text"]))
        for sentence_index, sentence in enumerate(document.sentences, start=1):
            text = sentence.text.strip()
            if text:
                records.append(
                    {
                        "chapter": int(paragraph["chapter"]),
                        "paragraph": int(paragraph["paragraph"]),
                        "sentence": sentence_index,
                        "text": text,
                    }
                )
    return records


def _write_sentence_records(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def _embed(overlap_path: Path, embedding_path: Path, model_name: str) -> None:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    sentences = overlap_path.read_text(encoding="utf-8").splitlines()
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        sentences,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    np.asarray(embeddings, dtype=np.float32).tofile(embedding_path)


def _run(command: list[str], cwd: Path | None = None, stdout: Path | None = None) -> None:
    if stdout is None:
        subprocess.run(command, cwd=cwd, check=True)
        return
    with stdout.open("w", encoding="utf-8") as handle:
        subprocess.run(command, cwd=cwd, check=True, stdout=handle)


def _parse_alignment(line: str) -> tuple[list[int], list[int], float] | None:
    match = re.fullmatch(r"\[([^]]*)\]:\[([^]]*)\]:([-+0-9.eE]+)", line.strip())
    if match is None:
        return None
    parse_indexes = lambda value: [] if not value.strip() else [int(index.strip()) for index in value.split(",")]
    return parse_indexes(match.group(1)), parse_indexes(match.group(2)), float(match.group(3))


def _write_candidates(
    output_path: Path,
    source: list[dict[str, object]],
    target: list[dict[str, object]],
    alignment_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_indexes", "target_indexes", "chapter", "alignment_type", "vecalign_cost",
        "source", "target", "review_status",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for line in alignment_path.read_text(encoding="utf-8").splitlines():
            parsed = _parse_alignment(line)
            if parsed is None:
                continue
            source_indexes, target_indexes, cost = parsed
            source_rows = [source[index] for index in source_indexes]
            target_rows = [target[index] for index in target_indexes]
            chapters = {int(record["chapter"]) for record in source_rows + target_rows}
            writer.writerow(
                {
                    "source_indexes": ",".join(map(str, source_indexes)),
                    "target_indexes": ",".join(map(str, target_indexes)),
                    "chapter": ",".join(map(str, sorted(chapters))),
                    "alignment_type": f"{len(source_indexes)}-{len(target_indexes)}",
                    "vecalign_cost": f"{cost:.6f}",
                    "source": " ".join(str(record["text"]) for record in source_rows),
                    "target": " ".join(str(record["text"]) for record in target_rows),
                    "review_status": "pending",
                }
            )


def align_with_vecalign(output_root: Path, vecalign_root: Path, embedding_model: str) -> None:
    try:
        import vecalign  # noqa: F401
    except ImportError as exc:
        raise WoolfPreparationError("Install Vecalign before running --align.") from exc

    sentences_dir = output_root / "sentences"
    sentence_records: dict[str, list[dict[str, object]]] = {}
    for language in ("en", "ca", "eu"):
        records = _sentence_records(_load_paragraphs(output_root / "paragraphs" / f"{language}.jsonl"), language)
        sentence_records[language] = records
        _write_sentence_records(sentences_dir / f"{language}.jsonl", records)
        (sentences_dir / f"{language}.txt").write_text(
            "\n".join(str(record["text"]) for record in records) + "\n", encoding="utf-8"
        )

    work_dir = output_root / "vecalign"
    work_dir.mkdir(parents=True, exist_ok=True)
    for language in ("en", "ca", "eu"):
        sentence_path = sentences_dir / f"{language}.txt"
        overlap_path = work_dir / f"{language}.overlap.txt"
        embedding_path = work_dir / f"{language}.overlap.emb"
        _run([sys.executable, "-m", "vecalign.overlap", "-i", str(sentence_path), "-o", str(overlap_path), "-n", "4"])
        _embed(overlap_path, embedding_path, embedding_model)

    for target_language in ("ca", "eu"):
        alignment_path = work_dir / f"en_{target_language}.alignments.txt"
        _run(
            [
                sys.executable, "-m", "vecalign.vecalign", "--alignment_max_size", "4",
                "--src", str((sentences_dir / "en.txt").resolve()),
                "--tgt", str((sentences_dir / f"{target_language}.txt").resolve()),
                "--src_embed", str((work_dir / "en.overlap.txt").resolve()), str((work_dir / "en.overlap.emb").resolve()),
                "--tgt_embed", str((work_dir / f"{target_language}.overlap.txt").resolve()), str((work_dir / f"{target_language}.overlap.emb").resolve()),
            ],
            cwd=vecalign_root,
            stdout=alignment_path,
        )
        _write_candidates(
            output_root / "candidates" / f"en_{target_language}_vecalign.tsv",
            sentence_records["en"],
            sentence_records[target_language],
            alignment_path,
        )
    _write_review_instructions(output_root)
    _update_manifest(output_root, alignment_status="candidates_ready")


def _update_manifest(output_root: Path, **updates: object) -> None:
    path = output_root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(updates)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_review_instructions(output_root: Path) -> None:
    text = (
        "# Woolf Alignment Review\n\n"
        "Review `candidates/en_ca_vecalign.tsv` and `candidates/en_eu_vecalign.tsv`. "
        "Set `review_status` to `accepted` only when the aligned source and target preserve the same content. "
        "Use `rejected` for incorrect, incomplete, front-matter, or note alignments. Keep every other row as `pending`.\n\n"
        "Only accepted rows are exported by `--build-final`; Vecalign output alone is never treated as final test data.\n"
    )
    (output_root / "REVIEW.md").write_text(text, encoding="utf-8")


def build_reviewed_final(output_root: Path) -> dict[str, int]:
    final_dir = output_root / "final"
    counts: dict[str, int] = {}
    for target_language in ("ca", "eu"):
        candidate_path = output_root / "candidates" / f"en_{target_language}_vecalign.tsv"
        if not candidate_path.exists():
            raise WoolfPreparationError(f"Candidate file not found: {candidate_path}")
        with candidate_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        accepted = [row for row in rows if row.get("review_status") == "accepted"]
        if not accepted:
            raise WoolfPreparationError(
                f"No accepted EN-{target_language} candidates. Review {candidate_path} before building final data."
            )
        records = [
            {
                "sentence_id": f"woolf_{target_language}_{index:04d}",
                "source": row["source"],
                "target": row["target"],
                "chapter": row["chapter"],
                "vecalign_cost": float(row["vecalign_cost"]),
                "alignment_type": row["alignment_type"],
                "corpus": "woolf_shared_chapters_1_3",
            }
            for index, row in enumerate(accepted, start=1)
        ]
        final_dir.mkdir(parents=True, exist_ok=True)
        output_path = final_dir / f"en_{target_language}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")
        counts[target_language] = len(records)
    _update_manifest(output_root, alignment_status="reviewed_final", final_pair_counts=counts)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the shared Woolf external final-test excerpt.")
    parser.add_argument("--input-root", type=Path, default=Path("data/woolf_test"))
    parser.add_argument("--output-root", type=Path, default=Path("data/woolf_test/prepared"))
    parser.add_argument("--align", action="store_true")
    parser.add_argument("--build-final", action="store_true")
    parser.add_argument("--vecalign-root", type=Path, default=Path(".local-tools/vecalign"))
    parser.add_argument("--embedding-model", default="sentence-transformers/LaBSE")
    args = parser.parse_args()
    if args.align or not args.build_final:
        counts = extract_shared_excerpt(args.input_root, args.output_root)
        print(f"Extracted shared Chapter 1-3 excerpt: {counts}")
    if args.align:
        align_with_vecalign(args.output_root, args.vecalign_root, args.embedding_model)
        print(f"Wrote Vecalign candidate alignments under {args.output_root / 'candidates'}")
    if args.build_final:
        counts = build_reviewed_final(args.output_root)
        print(f"Wrote reviewed final pairs: {counts}")


if __name__ == "__main__":
    main()
