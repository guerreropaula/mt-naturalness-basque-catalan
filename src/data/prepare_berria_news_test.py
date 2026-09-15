"""Segment and document-align the Berria EN-EU news test with Vecalign."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np


class BerriaPreparationError(RuntimeError):
    """Raised when the local Berria files cannot form an auditable test set."""


_SHEET_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _column_index(cell_reference: str) -> int:
    letters = re.match(r"[A-Z]+", cell_reference)
    if letters is None:
        raise BerriaPreparationError(f"Invalid XLSX cell reference: {cell_reference}")
    value = 0
    for character in letters.group(0):
        value = value * 26 + ord(character) - ord("A") + 1
    return value - 1


def _read_xlsx_table(path: Path) -> list[dict[str, str]]:
    """Read the first XLSX sheet with the standard library."""
    with ZipFile(path) as archive:
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = [
            "".join(node.text or "" for node in item.iterfind(".//m:t", _SHEET_NS))
            for item in shared_root.findall("m:si", _SHEET_NS)
        ]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    rows: list[list[str]] = []
    for row in sheet.findall(".//m:sheetData/m:row", _SHEET_NS):
        values: dict[int, str] = {}
        for cell in row.findall("m:c", _SHEET_NS):
            index = _column_index(cell.attrib["r"])
            value_node = cell.find("m:v", _SHEET_NS)
            value = "" if value_node is None else value_node.text or ""
            if cell.attrib.get("t") == "s" and value:
                value = shared[int(value)]
            values[index] = value
        if values:
            rows.append([values.get(index, "") for index in range(max(values) + 1)])

    header_index = next(
        (
            index
            for index, row in enumerate(rows)
            if len(row) >= 2 and row[0].strip() == "document_id" and row[1].strip() == "text"
        ),
        None,
    )
    if header_index is None:
        raise BerriaPreparationError(f"Could not find document_id/text headers in {path}")
    records = [
        {"document_id": row[0].strip(), "text": row[1].strip()}
        for row in rows[header_index + 1 :]
        if len(row) >= 2 and row[0].strip() and row[1].strip()
    ]
    if not records:
        raise BerriaPreparationError(f"No source articles found in {path}")
    return records


def _read_csv_table(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        records = [
            {
                "document_id": str(row.get("document_id", "")).strip(),
                "text": str(row.get("text", "")).strip(),
            }
            for row in csv.DictReader(handle)
        ]
    records = [row for row in records if row["document_id"] and row["text"]]
    if not records:
        raise BerriaPreparationError(f"No target articles found in {path}")
    return records


def _validate_articles(
    source_articles: list[dict[str, str]],
    target_articles: list[dict[str, str]],
) -> None:
    source_ids = [row["document_id"] for row in source_articles]
    target_ids = [row["document_id"] for row in target_articles]
    if len(source_ids) != len(set(source_ids)) or len(target_ids) != len(set(target_ids)):
        raise BerriaPreparationError("Berria document identifiers must be unique.")
    if source_ids != target_ids:
        raise BerriaPreparationError("English and Basque document IDs differ or are ordered differently.")


def _segment_articles(
    articles: list[dict[str, str]],
    language: str,
) -> dict[str, list[str]]:
    import stanza

    pipeline = stanza.Pipeline(
        lang=language,
        processors="tokenize",
        tokenize_no_ssplit=False,
        use_gpu=False,
        verbose=False,
    )
    segmented: dict[str, list[str]] = {}
    for article in articles:
        document = pipeline(article["text"])
        sentences = [sentence.text.strip() for sentence in document.sentences if sentence.text.strip()]
        if not sentences:
            raise BerriaPreparationError(
                f"Stanza produced no {language} sentences for {article['document_id']}."
            )
        segmented[article["document_id"]] = sentences
    return segmented


def _embed_overlaps(
    documents: list[list[str]],
    *,
    alignment_max_size: int,
    model_name: str,
    batch_size: int,
    device: str,
) -> tuple[dict[str, int], np.ndarray]:
    from sentence_transformers import SentenceTransformer
    from vecalign.overlap import yield_overlaps

    overlaps = sorted(
        {
            overlap
            for sentences in documents
            for overlap in yield_overlaps(sentences, alignment_max_size)
        }
    )
    if not overlaps:
        raise BerriaPreparationError("No sentence overlaps were produced for Vecalign.")
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        overlaps,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return {text: index for index, text in enumerate(overlaps)}, np.asarray(embeddings, dtype=np.float32)


def _align_documents(
    document_ids: list[str],
    source_sentences: dict[str, list[str]],
    target_sentences: dict[str, list[str]],
    *,
    embedding_model: str,
    alignment_max_size: int,
    embedding_batch_size: int,
    embedding_device: str,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    from vecalign.vecalign import make_alignment_types, make_doc_embedding, vecalign

    source_mapping, source_embeddings = _embed_overlaps(
        [source_sentences[document_id] for document_id in document_ids],
        alignment_max_size=alignment_max_size,
        model_name=embedding_model,
        batch_size=embedding_batch_size,
        device=embedding_device,
    )
    target_mapping, target_embeddings = _embed_overlaps(
        [target_sentences[document_id] for document_id in document_ids],
        alignment_max_size=alignment_max_size,
        model_name=embedding_model,
        batch_size=embedding_batch_size,
        device=embedding_device,
    )

    random.seed(seed)
    np.random.seed(seed)
    rows: list[dict[str, Any]] = []
    counts = {"alignment_steps": 0, "source_deletions": 0, "target_insertions": 0}
    alignment_types = make_alignment_types(alignment_max_size)
    width_over2 = math.ceil(alignment_max_size / 2.0) + 5
    for document_index, document_id in enumerate(document_ids, start=1):
        source = source_sentences[document_id]
        target = target_sentences[document_id]
        source_vectors = make_doc_embedding(
            source_mapping, source_embeddings, source, alignment_max_size
        )
        target_vectors = make_doc_embedding(
            target_mapping, target_embeddings, target, alignment_max_size
        )
        stack = vecalign(
            vecs0=source_vectors,
            vecs1=target_vectors,
            final_alignment_types=alignment_types,
            del_percentile_frac=0.2,
            width_over2=width_over2,
            max_size_full_dp=300,
            costs_sample_size=20_000,
            num_samps_for_norm=100,
        )
        alignments = stack[0]["final_alignments"]
        scores = stack[0]["alignment_scores"]
        for step, ((source_indices, target_indices), score) in enumerate(
            zip(alignments, scores), start=1
        ):
            counts["alignment_steps"] += 1
            if not source_indices:
                counts["target_insertions"] += 1
                continue
            if not target_indices:
                counts["source_deletions"] += 1
                continue
            rows.append(
                {
                    "document_id": document_id,
                    "document_index": document_index,
                    "alignment_step": step,
                    "source_sentence_indices": source_indices,
                    "target_sentence_indices": target_indices,
                    "alignment_type": f"{len(source_indices)}-{len(target_indices)}",
                    "vecalign_cost": float(score),
                    "source": " ".join(source[index] for index in source_indices),
                    "target": " ".join(target[index] for index in target_indices),
                }
            )
    return rows, counts


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            serialized = {
                key: ",".join(map(str, value)) if isinstance(value, list) else value
                for key, value in row.items()
            }
            writer.writerow(serialized)


def prepare_berria_news_test(
    source_xlsx: Path,
    target_csv: Path,
    output_root: Path,
    processed_output: Path,
    *,
    embedding_model: str = "sentence-transformers/LaBSE",
    alignment_max_size: int = 4,
    embedding_batch_size: int = 32,
    embedding_device: str = "cpu",
    seed: int = 42,
) -> dict[str, Any]:
    """Build sentence-segmented, document-bounded EN-EU Vecalign rows."""
    source_articles = _read_xlsx_table(source_xlsx)
    target_articles = _read_csv_table(target_csv)
    _validate_articles(source_articles, target_articles)
    document_ids = [row["document_id"] for row in source_articles]
    source_sentences = _segment_articles(source_articles, "en")
    target_sentences = _segment_articles(target_articles, "eu")

    sentence_rows: dict[str, list[dict[str, Any]]] = {"en": [], "eu": []}
    for document_index, document_id in enumerate(document_ids, start=1):
        for language, sentence_map in (("en", source_sentences), ("eu", target_sentences)):
            for sentence_index, text in enumerate(sentence_map[document_id], start=1):
                sentence_rows[language].append(
                    {
                        "document_id": document_id,
                        "document_index": document_index,
                        "sentence_index": sentence_index,
                        "text": text,
                    }
                )
    _write_jsonl(output_root / "sentences" / "en.jsonl", sentence_rows["en"])
    _write_jsonl(output_root / "sentences" / "eu.jsonl", sentence_rows["eu"])

    aligned, alignment_counts = _align_documents(
        document_ids,
        source_sentences,
        target_sentences,
        embedding_model=embedding_model,
        alignment_max_size=alignment_max_size,
        embedding_batch_size=embedding_batch_size,
        embedding_device=embedding_device,
        seed=seed,
    )
    final_rows = [
        {
            "id": f"berria_news_en_eu_{index:04d}",
            "sentence_id": f"berria_news_en_eu_{index:04d}",
            **row,
            "corpus": "berria_news_en_eu",
            "domain": "news",
            "alignment_method": "document-bounded Vecalign",
        }
        for index, row in enumerate(aligned, start=1)
    ]
    if not final_rows:
        raise BerriaPreparationError("Vecalign produced no non-empty bilingual alignments.")
    _write_jsonl(output_root / "candidates" / "en_eu_vecalign.jsonl", final_rows)
    _write_tsv(output_root / "candidates" / "en_eu_vecalign.tsv", final_rows)
    _write_jsonl(processed_output, final_rows)

    metadata = {
        "name": "Berria EN-EU sentence-segmented news test",
        "source_files": {"en": str(source_xlsx), "eu": str(target_csv)},
        "documents": len(document_ids),
        "sentence_counts_before_alignment": {
            "en": len(sentence_rows["en"]),
            "eu": len(sentence_rows["eu"]),
        },
        "aligned_rows": len(final_rows),
        "alignment_type_counts": {
            alignment_type: sum(row["alignment_type"] == alignment_type for row in final_rows)
            for alignment_type in sorted({row["alignment_type"] for row in final_rows})
        },
        "alignment_diagnostics": alignment_counts,
        "target_whitespace_tokens": sum(len(row["target"].split()) for row in final_rows),
        "source_whitespace_tokens": sum(len(row["source"].split()) for row in final_rows),
        "segmentation": "Stanza tokenize processor, language-specific EN and EU models",
        "alignment": {
            "tool": "Vecalign",
            "scope": "each corresponding document independently",
            "embedding_model": embedding_model,
            "embedding_device": embedding_device,
            "alignment_max_size": alignment_max_size,
            "null_alignments_excluded": True,
        },
        "seed": seed,
        "processed_output": str(processed_output),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-xlsx", type=Path, default=Path("data/news_test/eu/BERRIA-EN.xlsx")
    )
    parser.add_argument(
        "--target-csv", type=Path, default=Path("data/news_test/eu/BERRIA-EU.csv")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/news_test/eu/segmented")
    )
    parser.add_argument(
        "--processed-output",
        type=Path,
        default=Path("data/processed/berria_news_en_eu/test.jsonl"),
    )
    parser.add_argument("--embedding-model", default="sentence-transformers/LaBSE")
    parser.add_argument("--alignment-max-size", type=int, default=4)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    metadata = prepare_berria_news_test(
        args.source_xlsx,
        args.target_csv,
        args.output_root,
        args.processed_output,
        embedding_model=args.embedding_model,
        alignment_max_size=args.alignment_max_size,
        embedding_batch_size=args.embedding_batch_size,
        embedding_device=args.embedding_device,
        seed=args.seed,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
