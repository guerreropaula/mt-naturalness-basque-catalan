#!/usr/bin/env python3
"""Prepare a Catalan news-style MT test set from MaCoCu ca-en.

The script streams the MaCoCu sentence-level file, keeps high-quality rows whose
URL/title/domain look news-like, and writes EN->CA test files under data/news_test/ca.
It is intentionally stdlib-only so it can run on the cluster login node.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import heapq
import json
import random
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

MACOCU_CA_EN_SENT_URL = (
    "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1857/"
    "MaCoCu-ca-en.sent.txt.gz?sequence=2&isAllowed=y"
)

FALLBACK_COLUMNS = [
    "src_url",
    "trg_url",
    "src_text",
    "trg_text",
    "bleualign_score",
    "src_deferred_hash",
    "trg_deferred_hash",
    "src_paragraph_id",
    "trg_paragraph_id",
    "src_doc_title",
    "trg_doc_title",
    "src_crawl_date",
    "trg_crawl_date",
    "src_file_type",
    "trg_file_type",
    "src_boilerplate",
    "trg_boilerplate",
    "src_heading_html_tag",
    "trg_heading_html_tag",
    "bifixer_hash",
    "bifixer_score",
    "bicleaner_ai_score",
    "biroamer_entities_detected",
    "dsi",
    "translation_direction",
    "en_document_level_variant",
    "domain_en",
    "en_domain_level_variant",
]

HEADER_MARKERS = {
    "src_url",
    "trg_url",
    "src_text",
    "trg_text",
    "source",
    "target",
    "bicleaner_ai_score",
    "bleualign_score",
}

NEWS_DOMAINS = {
    "acn.cat",
    "aldia.cat",
    "ara.cat",
    "beteve.cat",
    "bondia.ad",
    "catalannews.com",
    "catalannewsagency.com",
    "ccma.cat",
    "diaridegirona.cat",
    "diarimes.com",
    "elnacional.cat",
    "elperiodico.cat",
    "elpuntavui.cat",
    "eltemps.cat",
    "emporda.info",
    "europapress.es",
    "lavanguardia.com",
    "naciodigital.cat",
    "regio7.cat",
    "segre.com",
    "vilaweb.cat",
    "3cat.cat",
    "324.cat",
}

NEWS_KEYWORDS = {
    "actualitat",
    "article",
    "articles",
    "catalan-news",
    "catalannews",
    "diari",
    "diario",
    "economia",
    "esports",
    "internacional",
    "journal",
    "magazine",
    "news",
    "noticia",
    "noticias",
    "noticies",
    "opinio",
    "opinion",
    "periodic",
    "periodico",
    "politica",
    "premsa",
    "press",
    "societat",
}

TEXT_TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


def clean_text(text: str | None) -> str:
    text = text or ""
    text = TEXT_TAG_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def normalize_host(url: str | None) -> str:
    if not url:
        return ""
    parsed = urlparse(url if "://" in url else "https://" + url)
    host = parsed.netloc.lower().split("@").pop()
    host = host.split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    return host


def host_matches(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain)


def looks_like_news(
    row: dict[str, str], extra_domains: set[str], *, allow_keyword_fallback: bool
) -> bool:
    """Keep known news publishers; keyword matching is an explicit fallback only."""
    hosts = {
        normalize_host(row.get("src_url")),
        normalize_host(row.get("trg_url")),
        normalize_host(row.get("domain_en")),
    }
    domains = NEWS_DOMAINS | extra_domains
    for host in hosts:
        if host and any(host_matches(host, domain) for domain in domains):
            return True

    if not allow_keyword_fallback:
        return False
    haystack = " ".join(
        [
            row.get("src_url", ""),
            row.get("trg_url", ""),
            row.get("src_doc_title", ""),
            row.get("trg_doc_title", ""),
            row.get("domain_en", ""),
        ]
    ).lower()
    haystack = haystack.replace("%c3%ad", "i").replace("%c3%a0", "a").replace("%c3%a8", "e")
    return any(keyword in haystack for keyword in NEWS_KEYWORDS)


def get_first(row: dict[str, str], names: Iterable[str]) -> str:
    for name in names:
        value = row.get(name)
        if value:
            return value
    return ""


def count_csv_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return sum(1 for _ in reader)


def count_csv_tokens(path: Path, column: str) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames and column not in reader.fieldnames:
            raise ValueError(f"Column {column!r} is not in {path}; columns={reader.fieldnames}")
        return sum(word_count(row.get(column, "")) for row in reader)


def resolve_target_count(value: str, berria_file: Path) -> int:
    if value != "auto":
        return int(value)
    count = count_csv_rows(berria_file)
    if count is None:
        return 77
    return count


def resolve_target_tokens(value: str, reference_file: Path, reference_column: str) -> int:
    if value != "auto":
        return int(value)
    count = count_csv_tokens(reference_file, reference_column)
    if count is None:
        return 10129
    return count


def download_if_needed(url: str, output_path: Path) -> None:
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Using existing MaCoCu file: {output_path}")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    print(f"Downloading MaCoCu sentence file to {output_path}")
    print("This is about 934 MB compressed; it can take a while.")
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as out:
        total = response.headers.get("Content-Length")
        total_int = int(total) if total and total.isdigit() else None
        read = 0
        next_report = 100 * 1024 * 1024
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            read += len(chunk)
            if read >= next_report:
                if total_int:
                    pct = read / total_int * 100
                    print(f"  downloaded {read / 1024 / 1024:.0f} MB ({pct:.1f}%)")
                else:
                    print(f"  downloaded {read / 1024 / 1024:.0f} MB")
                next_report += 100 * 1024 * 1024
    tmp_path.replace(output_path)


def iter_rows(path: Path):
    # Some crawled document titles/URLs exceed Python csv's 128 KiB default.
    csv.field_size_limit(sys.maxsize)
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        first = next(reader, None)
        if first is None:
            return
        first_norm = [item.strip() for item in first]
        has_header = bool(set(first_norm) & HEADER_MARKERS)
        header = first_norm if has_header else FALLBACK_COLUMNS
        if not has_header:
            yield dict(zip(header, first))
        for fields in reader:
            if not fields or len(fields) < 4:
                continue
            if len(fields) < len(header):
                fields = fields + [""] * (len(header) - len(fields))
            yield dict(zip(header, fields))


def candidate_from_row(row: dict[str, str], english_side: str) -> dict[str, object] | None:
    src_text = clean_text(get_first(row, ["src_text", "source", "src_segment"]))
    trg_text = clean_text(get_first(row, ["trg_text", "target", "trg_segment"]))
    if english_side == "src":
        source = src_text
        target = trg_text
        source_url = row.get("src_url", "")
        target_url = row.get("trg_url", "")
        source_title = row.get("src_doc_title", "")
        target_title = row.get("trg_doc_title", "")
        source_paragraph_id = row.get("src_paragraph_id", "")
        target_paragraph_id = row.get("trg_paragraph_id", "")
    else:
        source = trg_text
        target = src_text
        source_url = row.get("trg_url", "")
        target_url = row.get("src_url", "")
        source_title = row.get("trg_doc_title", "")
        target_title = row.get("src_doc_title", "")
        source_paragraph_id = row.get("trg_paragraph_id", "")
        target_paragraph_id = row.get("src_paragraph_id", "")
    if not source or not target:
        return None
    return {
        "source": source,
        "target": target,
        "source_url": source_url,
        "target_url": target_url,
        "source_title": source_title,
        "target_title": target_title,
        "domain": normalize_host(target_url) or normalize_host(source_url) or row.get("domain_en", ""),
        "bicleaner_ai_score": safe_float(row.get("bicleaner_ai_score")),
        "bleualign_score": safe_float(row.get("bleualign_score")),
        "translation_direction": row.get("translation_direction", ""),
        "dsi": row.get("dsi", ""),
        "source_paragraph_id": source_paragraph_id,
        "target_paragraph_id": target_paragraph_id,
    }


def passes_filters(row: dict[str, str], args, extra_domains: set[str]) -> bool:
    if not looks_like_news(
        row, extra_domains, allow_keyword_fallback=args.allow_news_keyword_fallback
    ):
        return False

    bicleaner = safe_float(row.get("bicleaner_ai_score"))
    if bicleaner is None or bicleaner < args.min_bicleaner:
        return False

    bleualign = safe_float(row.get("bleualign_score"))
    if args.min_bleualign is not None and (bleualign is None or bleualign < args.min_bleualign):
        return False

    if args.exclude_machine_translated:
        direction = row.get("translation_direction", "").lower()
        if "mt" in direction:
            return False

    entities = row.get("biroamer_entities_detected", "").strip().lower()
    if args.exclude_personal_info and entities.startswith(("y", "true", "1")):
        return False

    boilerplate_values = {
        row.get("src_boilerplate", "").strip().lower(),
        row.get("trg_boilerplate", "").strip().lower(),
    }
    bad_boilerplate = {"bad", "short", "unknown"}
    if boilerplate_values & bad_boilerplate:
        return False

    return True


def word_count(text: str) -> int:
    return len(text.split())


def load_extra_domains(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    domains = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().lower()
        if line and not line.startswith("#"):
            domains.add(line[4:] if line.startswith("www.") else line)
    return domains


def select_candidates(args) -> tuple[list[dict[str, object]], dict[str, object]]:
    extra_domains = load_extra_domains(args.news_domains_file)
    heap: list[tuple[float, int, dict[str, object]]] = []
    scanned = 0
    news_like = 0
    accepted = 0
    rejects = Counter()

    for row in iter_rows(args.input_gz):
        scanned += 1
        if scanned % args.progress_every == 0:
            print(f"Scanned {scanned:,} rows; accepted candidates so far: {accepted:,}")

        if not looks_like_news(
            row, extra_domains, allow_keyword_fallback=args.allow_news_keyword_fallback
        ):
            rejects["not_news_like"] += 1
            continue
        news_like += 1
        if not passes_filters(row, args, extra_domains):
            rejects["quality_or_metadata_filter"] += 1
            continue

        candidate = candidate_from_row(row, args.english_side)
        if candidate is None:
            rejects["missing_text"] += 1
            continue
        src_len = word_count(str(candidate["source"]))
        tgt_len = word_count(str(candidate["target"]))
        if src_len < args.min_words or tgt_len < args.min_words:
            rejects["too_short"] += 1
            continue
        if src_len > args.max_words or tgt_len > args.max_words:
            rejects["too_long"] += 1
            continue

        score = float(candidate["bicleaner_ai_score"] or 0.0)
        item = (score, -scanned, candidate)
        if len(heap) < args.candidate_limit:
            heapq.heappush(heap, item)
        else:
            heapq.heappushpop(heap, item)
        accepted += 1

    candidates = [item[2] for item in sorted(heap, key=lambda item: (item[0], item[1]), reverse=True)]
    stats = {
        "scanned_rows": scanned,
        "news_like_rows": news_like,
        "accepted_candidate_rows": accepted,
        "kept_candidate_limit": len(candidates),
        "rejects": dict(rejects),
    }
    return candidates, stats


def publisher_group(domain: object) -> str:
    """Collapse publisher subdomains so selection caps apply to one outlet."""
    labels = str(domain or "unknown").lower().split(".")
    return ".".join(labels[-2:]) if len(labels) >= 2 else str(domain or "unknown").lower()


def diversify_and_sample(
    candidates: list[dict[str, object]], args
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rng = random.Random(args.seed)
    seen_pairs: set[tuple[str, str]] = set()
    seen_sources: set[str] = set()
    seen_targets: set[str] = set()
    unique: list[dict[str, object]] = []
    dedup_drops = Counter()
    for candidate in candidates:
        source = str(candidate["source"]).casefold()
        target = str(candidate["target"]).casefold()
        pair = (source, target)
        if pair in seen_pairs:
            dedup_drops["pair"] += 1
            continue
        if source in seen_sources:
            dedup_drops["source"] += 1
            continue
        if target in seen_targets:
            dedup_drops["target"] += 1
            continue
        seen_pairs.add(pair)
        seen_sources.add(source)
        seen_targets.add(target)
        unique.append(candidate)

    if args.size_mode == "rows":
        approx_rows = args.target_count_resolved
    else:
        approx_rows = max(1, args.target_token_count_resolved // max(args.expected_words_per_segment, 1))

    top_pool_size = min(len(unique), max(approx_rows * args.top_pool_multiplier, approx_rows))
    top_pool = unique[:top_pool_size]
    rng.shuffle(top_pool)
    top_pool.sort(key=lambda row: float(row.get("bicleaner_ai_score") or 0.0), reverse=True)

    selected: list[dict[str, object]] = []
    per_publisher = Counter()
    selected_pairs: set[tuple[str, str]] = set()
    selected_sources: set[str] = set()
    selected_targets: set[str] = set()

    def selected_target_tokens() -> int:
        return sum(word_count(str(row.get("target") or "")) for row in selected)

    def enough() -> bool:
        if args.size_mode == "rows":
            return len(selected) >= args.target_count_resolved
        return selected_target_tokens() >= args.target_token_count_resolved

    def add_candidate(candidate: dict[str, object], use_publisher_cap: bool) -> bool:
        source = str(candidate["source"]).casefold()
        target = str(candidate["target"]).casefold()
        pair = (source, target)
        if pair in selected_pairs or source in selected_sources or target in selected_targets:
            return False
        publisher = publisher_group(candidate.get("domain"))
        if use_publisher_cap and per_publisher[publisher] >= args.max_per_domain:
            return False
        selected.append(candidate)
        selected_pairs.add(pair)
        selected_sources.add(source)
        selected_targets.add(target)
        per_publisher[publisher] += 1
        return True

    cap_relaxed = False
    for use_publisher_cap in [True, False]:
        for candidate in top_pool:
            add_candidate(candidate, use_publisher_cap=use_publisher_cap)
            if enough():
                break
        if enough():
            break
        cap_relaxed = True

    if args.size_mode == "target-tokens" and len(selected) > 1:
        budget = args.target_token_count_resolved
        total = selected_target_tokens()
        without_last = total - word_count(str(selected[-1].get("target") or ""))
        if without_last >= budget and abs(budget - without_last) < abs(total - budget):
            selected.pop()

    selected.sort(key=lambda row: (str(row.get("domain") or ""), str(row.get("source") or "")))
    for idx, row in enumerate(selected, start=1):
        row["id"] = f"macocu_news_ca_{idx:05d}"
        row["source_lang"] = "en"
        row["target_lang"] = "ca"
    selected_publishers = Counter(publisher_group(row.get("domain")) for row in selected)
    selection_stats = {
        "unique_candidates": len(unique),
        "deduplication_drops": dict(dedup_drops),
        "top_pool_size": top_pool_size,
        "publisher_cap_relaxed": cap_relaxed,
        "selected_publishers": dict(selected_publishers.most_common()),
    }
    return selected, selection_stats

def write_outputs(
    rows: list[dict[str, object]], args, stats: dict[str, object], selection_stats: dict[str, object]
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / args.output_stem
    jsonl_path = stem.with_suffix(".jsonl")
    csv_path = stem.with_suffix(".csv")
    tsv_path = stem.with_suffix(".tsv")
    metadata_path = args.output_dir / "metadata.json"

    fieldnames = [
        "id",
        "source_lang",
        "target_lang",
        "source",
        "target",
        "domain",
        "source_url",
        "target_url",
        "source_title",
        "target_title",
        "bicleaner_ai_score",
        "bleualign_score",
        "translation_direction",
        "dsi",
        "source_paragraph_id",
        "target_paragraph_id",
    ]

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    metadata = {
        "source_corpus": "MaCoCu-ca-en 1.0 sentence-level TXT",
        "download_url": args.download_url,
        "input_gz": str(args.input_gz),
        "output_rows": len(rows),
        "size_mode": args.size_mode,
        "target_count": args.target_count_resolved,
        "target_token_count": args.target_token_count_resolved,
        "selected_source_tokens": sum(word_count(str(row.get("source") or "")) for row in rows),
        "selected_target_tokens": sum(word_count(str(row.get("target") or "")) for row in rows),
        "filters": {
            "news_domains_file": str(args.news_domains_file) if args.news_domains_file else None,
            "min_bicleaner": args.min_bicleaner,
            "min_bleualign": args.min_bleualign,
            "exclude_machine_translated": args.exclude_machine_translated,
            "exclude_personal_info": args.exclude_personal_info,
            "min_words": args.min_words,
            "max_words": args.max_words,
            "max_per_domain": args.max_per_domain,
            "english_side": args.english_side,
            "allow_news_keyword_fallback": args.allow_news_keyword_fallback,
            "seed": args.seed,
        },
        "streaming_stats": stats,
        "selection": selection_stats,
        "selected_domains": Counter(str(row.get("domain") or "unknown") for row in rows),
    }
    metadata["selected_domains"] = dict(metadata["selected_domains"].most_common())
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.processed_output is not None:
        args.processed_output.parent.mkdir(parents=True, exist_ok=True)
        with args.processed_output.open("w", encoding="utf-8") as handle:
            for row in rows:
                record = {
                    "id": row["id"],
                    "source": row["source"],
                    "target": row["target"],
                    "corpus": "macocu_news_en_ca",
                    "domain": row["domain"],
                    "publisher": publisher_group(row["domain"]),
                    "bicleaner_ai_score": row["bicleaner_ai_score"],
                    "bleualign_score": row["bleualign_score"],
                    "source_url": row["source_url"],
                    "target_url": row["target_url"],
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  {args.processed_output}")

    print(f"Wrote {len(rows)} rows")
    print(f"  {jsonl_path}")
    print(f"  {csv_path}")
    print(f"  {tsv_path}")
    print(f"  {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/news_test/ca"))
    parser.add_argument("--output-stem", default="macocu_news_en_ca")
    parser.add_argument(
        "--processed-output",
        type=Path,
        default=None,
        help="Optional normalized test JSONL written for the P0-P5 experiment loaders.",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/news_test/ca/raw"))
    parser.add_argument("--input-gz", type=Path, default=None, help="Existing MaCoCu .sent.txt.gz file")
    parser.add_argument("--download-url", default=MACOCU_CA_EN_SENT_URL)
    parser.add_argument("--size-mode", choices=["target-tokens", "rows"], default="target-tokens")
    parser.add_argument("--target-count", default="auto", help="Integer, or auto to match BERRIA-EU.csv rows; used only with --size-mode rows")
    parser.add_argument("--target-tokens", default="auto", help="Integer, or auto to match the token count in --token-reference-file")
    parser.add_argument("--berria-file", type=Path, default=Path("data/news_test/eu/BERRIA-EU.csv"))
    parser.add_argument("--token-reference-file", type=Path, default=Path("data/news_test/eu/BERRIA-EU.csv"))
    parser.add_argument("--token-reference-column", default="text")
    parser.add_argument("--english-side", choices=["src", "trg"], default="trg", help="In MaCoCu ca-en, English is normally the trg side")
    parser.add_argument("--news-domains-file", type=Path, default=None)
    parser.add_argument(
        "--allow-news-keyword-fallback",
        action="store_true",
        help="Also accept URLs/titles with news keywords when the publisher is not in the news-domain allowlist.",
    )
    parser.add_argument("--min-bicleaner", type=float, default=0.80)
    parser.add_argument("--min-bleualign", type=float, default=None)
    parser.add_argument("--include-machine-translated", dest="exclude_machine_translated", action="store_false")
    parser.set_defaults(exclude_machine_translated=True)
    parser.add_argument("--include-personal-info", dest="exclude_personal_info", action="store_false")
    parser.set_defaults(exclude_personal_info=True)
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--max-words", type=int, default=120)
    parser.add_argument("--candidate-limit", type=int, default=20000)
    parser.add_argument("--top-pool-multiplier", type=int, default=20)
    parser.add_argument("--expected-words-per-segment", type=int, default=45)
    parser.add_argument("--max-per-domain", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=250000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.target_count_resolved = None
    args.target_token_count_resolved = None
    if args.size_mode == "rows":
        args.target_count_resolved = resolve_target_count(args.target_count, args.berria_file)
    else:
        args.target_token_count_resolved = resolve_target_tokens(
            args.target_tokens, args.token_reference_file, args.token_reference_column
        )
    if args.input_gz is None:
        args.input_gz = args.cache_dir / "MaCoCu-ca-en.sent.txt.gz"
        download_if_needed(args.download_url, args.input_gz)
    elif not args.input_gz.exists():
        print(f"Input file does not exist: {args.input_gz}", file=sys.stderr)
        return 2

    if args.size_mode == "rows":
        print(f"Target row count: {args.target_count_resolved}")
    else:
        print(f"Target Catalan token count: {args.target_token_count_resolved}")
    print(f"Quality filter: bicleaner_ai_score >= {args.min_bicleaner}")
    print("Selecting news-like MaCoCu candidates...")
    candidates, stats = select_candidates(args)
    if args.size_mode == "rows":
        enough_candidates = len(candidates) >= args.target_count_resolved
        need_label = f"{args.target_count_resolved} rows"
    else:
        available_tokens = sum(word_count(str(row.get("target") or "")) for row in candidates)
        enough_candidates = available_tokens >= args.target_token_count_resolved
        need_label = f"{args.target_token_count_resolved} target tokens"
    if not enough_candidates:
        print(
            f"Only found {len(candidates)} candidates; need {need_label}. "
            "Try lowering --min-bicleaner, increasing --max-words, or using --include-machine-translated.",
            file=sys.stderr,
        )
        return 1

    selected, selection_stats = diversify_and_sample(candidates, args)
    selected_tokens = sum(word_count(str(row.get("target") or "")) for row in selected)
    if args.size_mode == "rows":
        selection_is_sufficient = len(selected) >= args.target_count_resolved
        selected_need_label = f"{args.target_count_resolved} rows"
        selected_have_label = f"{len(selected)} rows"
    else:
        selection_is_sufficient = selected_tokens >= args.target_token_count_resolved
        selected_need_label = f"{args.target_token_count_resolved} target tokens"
        selected_have_label = f"{selected_tokens} target tokens"
    if not selection_is_sufficient:
        print(
            f"Strict deduplication and publisher balancing retained only {selected_have_label}; "
            f"need {selected_need_label}.",
            file=sys.stderr,
        )
        return 1
    write_outputs(selected, args, stats, selection_stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
