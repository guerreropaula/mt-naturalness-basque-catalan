"""Prepare a reviewable full-book Woolf literary test for EN-to-EU or EN-to-CA.

English and Basque are split by the novel's three semantic part headings, not
by EPUB member boundaries. Vecalign is run independently inside each part so
that an alignment can never cross from one part of the novel into another.
Only rows explicitly marked ``accepted`` are exported as final-test data.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from xml.etree import ElementTree
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from src.data.prepare_woolf_final_test import _blocks, _parse_alignment


class WoolfFullBookError(RuntimeError):
    """Raised when the full-book editions cannot form a valid test set."""


_EN_PART_MARKERS = {
    "part 1. the window": 1,
    "part 2. time passes": 2,
    "part 3. the lighthouse": 3,
}
_EU_PART_MARKERS = {"leihoa": 1, "denborak aurrera": 2, "faroa": 3}
_PARTS = (1, 2, 3)
# The editions divide the final part differently. The Basque chapter 12 starts
# at English chapter 13, and its chapter 13 contains English chapter 14.
_EXPECTED_CHAPTERS = {
    "en": {1: 19, 2: 10, 3: 14},
    "eu": {1: 19, 2: 10, 3: 13},
    "ca": {1: 19, 2: 10, 3: 13},
}
_CA_CHAPTER_RANGES = {1: range(1, 20), 2: range(20, 30), 3: range(30, 43)}
_CA_EXCLUDED_PARAGRAPH_CLASSES = {"footnote", "ftline", "end", "image"}


def _english_parts(epub: Path) -> dict[int, list[dict[str, Any]]]:
    parts = {part: [] for part in _PARTS}
    active_part: int | None = None
    active_chapter: int | None = None
    paragraph = 0
    for tag, text in _blocks(epub, "OEBPS/Text/OML004.html"):
        if tag == "h1":
            active_part = _EN_PART_MARKERS.get(text.casefold())
            active_chapter = None
            paragraph = 0
            continue
        if active_part is None:
            continue
        if tag == "h2":
            match = re.fullmatch(r"chapter\s+(\d+)", text, flags=re.IGNORECASE)
            active_chapter = int(match.group(1)) if match else None
            paragraph = 0
            continue
        if tag == "p" and active_chapter is not None:
            paragraph += 1
            parts[active_part].append(
                {"part": active_part, "chapter": active_chapter, "paragraph": paragraph, "text": text}
            )
    return parts


def _basque_parts(epub: Path) -> dict[int, list[dict[str, Any]]]:
    parts = {part: [] for part in _PARTS}
    active_part: int | None = None
    active_chapter: int | None = None
    paragraph = 0
    for member in ("Ops/1.html", "Ops/2.html", "Ops/3.html"):
        for tag, text in _blocks(epub, member):
            if tag != "p":
                continue
            marker_part = _EU_PART_MARKERS.get(text.casefold())
            if marker_part is not None:
                active_part = marker_part
                active_chapter = None
                paragraph = 0
                continue
            if active_part is None:
                continue
            if text in {"I", "II", "III"}:
                continue
            if text.isdigit():
                active_chapter = int(text)
                paragraph = 0
                continue
            if active_chapter is not None:
                paragraph += 1
                parts[active_part].append(
                    {"part": active_part, "chapter": active_chapter, "paragraph": paragraph, "text": text}
                )
    return parts


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _element_text_without_superscripts(element: ElementTree.Element) -> str:
    parts: list[str] = []
    if element.text:
        parts.append(element.text)
    for child in element:
        if _local_name(child.tag) != "sup":
            parts.append(_element_text_without_superscripts(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def _catalan_chapter_paragraphs(epub: Path, chapter: int) -> list[str]:
    member = f"OEBPS/Text/cap-{chapter:02d}.xhtml"
    with ZipFile(epub) as archive:
        root = ElementTree.fromstring(archive.read(member))
    paragraphs: list[str] = []
    for element in root.iter():
        if _local_name(element.tag) != "p":
            continue
        classes = set(element.attrib.get("class", "").split())
        if classes & _CA_EXCLUDED_PARAGRAPH_CLASSES:
            continue
        text = " ".join(_element_text_without_superscripts(element).split())
        if text:
            paragraphs.append(text)
    return paragraphs


def _catalan_parts(epub: Path) -> dict[int, list[dict[str, Any]]]:
    parts = {part: [] for part in _PARTS}
    for part, chapter_range in _CA_CHAPTER_RANGES.items():
        for local_chapter, epub_chapter in enumerate(chapter_range, start=1):
            for paragraph, text in enumerate(_catalan_chapter_paragraphs(epub, epub_chapter), start=1):
                parts[part].append(
                    {
                        "part": part,
                        "chapter": local_chapter,
                        "edition_chapter": epub_chapter,
                        "paragraph": paragraph,
                        "text": text,
                    }
                )
    return parts


def _validate_parts(parts: dict[int, list[dict[str, Any]]], language: str) -> None:
    for part, expected in _EXPECTED_CHAPTERS[language].items():
        chapters = {int(row["chapter"]) for row in parts[part]}
        expected_set = set(range(1, expected + 1))
        if chapters != expected_set:
            raise WoolfFullBookError(
                f"{language} part {part} chapter mismatch: expected {sorted(expected_set)}, got {sorted(chapters)}"
            )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_full_book(
    input_root: Path,
    output_root: Path,
    target_language: str = "eu",
) -> dict[str, dict[int, int]]:
    if target_language not in {"eu", "ca"}:
        raise WoolfFullBookError(f"Unsupported target language: {target_language}")
    target_epub = {
        "eu": input_root / "eu" / "Virginia Woolf, Farorantz.epub",
        "ca": input_root / "ca" / "Cap al far - Virginia Woolf.epub",
    }[target_language]
    target_parts = _basque_parts(target_epub) if target_language == "eu" else _catalan_parts(target_epub)
    editions = {
        "en": _english_parts(input_root / "en" / "To the Lighthouse - Virginia Woolf - EPUB.epub"),
        target_language: target_parts,
    }
    counts: dict[str, dict[int, int]] = {}
    for language, parts in editions.items():
        _validate_parts(parts, language)
        counts[language] = {}
        for part, rows in parts.items():
            _write_jsonl(output_root / "paragraphs" / f"{language}_part{part}.jsonl", rows)
            counts[language][part] = len(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": f"woolf_to_the_lighthouse_full_en_{target_language}",
        "purpose": "external_final_literary_test",
        "languages": ["en", target_language],
        "alignment_scope": "part-bounded",
        "paragraph_counts": counts,
        "alignment_status": "not_aligned",
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return counts


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sentence_records(rows: list[dict[str, Any]], language: str) -> list[dict[str, Any]]:
    import stanza

    pipeline = stanza.Pipeline(
        lang=language,
        processors="tokenize",
        download_method=None,
        use_gpu=False,
        verbose=False,
    )
    sentences: list[dict[str, Any]] = []
    for row in rows:
        document = pipeline(str(row["text"]))
        for sentence_index, sentence in enumerate(document.sentences, start=1):
            text = sentence.text.strip()
            if text:
                sentences.append({**row, "sentence": sentence_index, "text": text})
    return sentences


def _run(command: list[str], *, stdout: Path | None = None) -> None:
    if stdout is None:
        subprocess.run(command, check=True)
        return
    with stdout.open("w", encoding="utf-8") as handle:
        subprocess.run(command, check=True, stdout=handle)


def _embed(overlap: Path, output: Path, model_name: str, device: str) -> None:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    values = model.encode(
        overlap.read_text(encoding="utf-8").splitlines(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    np.asarray(values, dtype=np.float32).tofile(output)


def _candidate_rows(
    part: int,
    source: list[dict[str, Any]],
    target: list[dict[str, Any]],
    alignment_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in alignment_path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_alignment(line)
        if parsed is None:
            continue
        source_indexes, target_indexes, cost = parsed
        source_rows = [source[index] for index in source_indexes]
        target_rows = [target[index] for index in target_indexes]
        rows.append(
            {
                "part": part,
                "source_indexes": ",".join(map(str, source_indexes)),
                "target_indexes": ",".join(map(str, target_indexes)),
                "source_chapters": ",".join(map(str, sorted({int(row["chapter"]) for row in source_rows}))),
                "target_chapters": ",".join(map(str, sorted({int(row["chapter"]) for row in target_rows}))),
                "alignment_type": f"{len(source_indexes)}-{len(target_indexes)}",
                "vecalign_cost": f"{cost:.6f}",
                "source": " ".join(str(row["text"]) for row in source_rows),
                "target": " ".join(str(row["text"]) for row in target_rows),
                "sentence_level_eligible": "yes" if len(source_indexes) == 1 and len(target_indexes) == 1 else "no",
                "review_status": "pending" if len(source_indexes) == 1 and len(target_indexes) == 1 else "excluded_non_1_to_1",
            }
        )
    return rows


def align_full_book(
    output_root: Path,
    target_language: str,
    embedding_model: str,
    embedding_device: str,
    force: bool,
) -> Path:
    candidate_path = output_root / "candidates" / f"en_{target_language}_vecalign.tsv"
    if candidate_path.exists() and not force:
        with candidate_path.open(encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle, delimiter="\t"))
        if any(row.get("review_status") in {"accepted", "rejected"} for row in existing):
            raise WoolfFullBookError(f"Refusing to overwrite reviewed candidates: {candidate_path}")

    all_candidates: list[dict[str, Any]] = []
    sentence_dir = output_root / "sentences"
    work_dir = output_root / "vecalign"
    sentence_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    for part in _PARTS:
        per_language: dict[str, list[dict[str, Any]]] = {}
        paths: dict[str, tuple[Path, Path]] = {}
        for language in ("en", target_language):
            paragraphs = _load_jsonl(output_root / "paragraphs" / f"{language}_part{part}.jsonl")
            sentences = _sentence_records(paragraphs, language)
            per_language[language] = sentences
            _write_jsonl(sentence_dir / f"{language}_part{part}.jsonl", sentences)
            text_path = sentence_dir / f"{language}_part{part}.txt"
            text_path.write_text("\n".join(str(row["text"]) for row in sentences) + "\n", encoding="utf-8")
            overlap = work_dir / f"{language}_part{part}.overlap.txt"
            embedding = work_dir / f"{language}_part{part}.overlap.emb"
            _run([sys.executable, "-m", "vecalign.overlap", "-i", str(text_path), "-o", str(overlap), "-n", "4"])
            _embed(overlap, embedding, embedding_model, embedding_device)
            paths[language] = (overlap, embedding)

        alignment = work_dir / f"en_{target_language}_part{part}.alignments.txt"
        _run(
            [
                sys.executable,
                "-m",
                "vecalign.vecalign",
                "--alignment_max_size",
                "4",
                "--src",
                str((sentence_dir / f"en_part{part}.txt").resolve()),
                "--tgt",
                str((sentence_dir / f"{target_language}_part{part}.txt").resolve()),
                "--src_embed",
                str(paths["en"][0].resolve()),
                str(paths["en"][1].resolve()),
                "--tgt_embed",
                str(paths[target_language][0].resolve()),
                str(paths[target_language][1].resolve()),
            ],
            stdout=alignment,
        )
        all_candidates.extend(
            _candidate_rows(part, per_language["en"], per_language[target_language], alignment)
        )

    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(all_candidates[0])
    with candidate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(all_candidates)
    (output_root / "REVIEW.md").write_text(
        f"# Full-book EN-{target_language.upper()} Woolf alignment review\n\n"
        f"Review `candidates/en_{target_language}_vecalign.tsv`. Only rows labelled `sentence_level_eligible=yes` may be reviewed. Set `review_status` to `accepted` only for a valid 1-1 sentence pair. "
        "Use `rejected` for wrong, incomplete, note, or empty alignments. Only accepted rows are exported.\n",
        encoding="utf-8",
    )
    manifest_path = output_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        alignment_status="candidates_ready",
        candidate_count=len(all_candidates),
        sentence_level_candidate_count=sum(row["alignment_type"] == "1-1" for row in all_candidates),
        excluded_grouped_alignment_count=sum(row["alignment_type"] != "1-1" for row in all_candidates),
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return candidate_path


def build_reviewed_final(output_root: Path, target_language: str = "eu") -> int:
    candidate_path = output_root / "candidates" / f"en_{target_language}_vecalign.tsv"
    with candidate_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    accepted = [row for row in rows if row.get("review_status") == "accepted"]
    non_sentence_level = [row for row in accepted if row.get("alignment_type") != "1-1"]
    if non_sentence_level:
        raise WoolfFullBookError(
            f"Final test requires 1-1 sentence pairs; found {len(non_sentence_level)} accepted grouped alignments."
        )
    if not accepted:
        raise WoolfFullBookError(f"No accepted candidates in {candidate_path}")
    if any(not row.get("source", "").strip() or not row.get("target", "").strip() for row in accepted):
        raise WoolfFullBookError("Accepted candidates must contain both source and target text.")
    records = [
        {
            "id": f"woolf_full_en_{target_language}_{index:04d}",
            "sentence_id": f"woolf_full_en_{target_language}_{index:04d}",
            "source": row["source"],
            "target": row["target"],
            "part": int(row["part"]),
            "source_chapters": row["source_chapters"],
            "target_chapters": row["target_chapters"],
            "alignment_type": row["alignment_type"],
            "vecalign_cost": float(row["vecalign_cost"]),
            "corpus": f"woolf_to_the_lighthouse_full_en_{target_language}",
        }
        for index, row in enumerate(accepted, start=1)
    ]
    output_path = output_root / "final" / f"en_{target_language}.jsonl"
    _write_jsonl(output_path, records)
    manifest_path = output_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(alignment_status="reviewed_final", final_pair_count=len(records))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("data/woolf_test"))
    parser.add_argument("--target-language", choices=("eu", "ca"), default="eu")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--align", action="store_true")
    parser.add_argument("--build-final", action="store_true")
    parser.add_argument("--embedding-model", default="sentence-transformers/LaBSE")
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root or Path(f"data/woolf_test/full_en_{args.target_language}")
    if args.align or not args.build_final:
        print(
            f"Extracted full EN-{args.target_language.upper()} book: "
            f"{extract_full_book(args.input_root, output_root, args.target_language)}"
        )
    if args.align:
        candidate = align_full_book(
            output_root, args.target_language, args.embedding_model, args.embedding_device, args.force
        )
        print(f"Wrote reviewable candidates to {candidate}")
    if args.build_final:
        print(f"Wrote {build_reviewed_final(output_root, args.target_language)} reviewed final pairs")


if __name__ == "__main__":
    main()
