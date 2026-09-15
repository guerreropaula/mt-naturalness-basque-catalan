"""Synonym Frequency Analysis (SFA) metrics and Apertium dictionary helpers."""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_SOURCE_UPOS = frozenset({"NOUN", "VERB", "ADJ"})
_ROOT_DIR = Path(__file__).resolve().parents[2]
_WHITESPACE_RE = re.compile(r"\s+")

DEFAULT_APERTIUM_DICTIONARIES = {
    "ca": {
        "path_candidates": [
            _ROOT_DIR / "data" / "dictionaries" / "apertium-eng-cat.eng-cat.dix",
        ],
        "invert": False,
        "resource": "Apertium apertium-eng-cat",
    },
    "eu": {
        "path_candidates": [
            _ROOT_DIR / "data" / "dictionaries" / "apertium-eu-en.eu-en.dix",
        ],
        "invert": True,
        "resource": "Apertium apertium-eu-en (inverted)",
    },
}


class SFAAnalysisError(RuntimeError):
    """Raised when SFA resources or computations are unavailable."""


def empty_sfa_summary() -> dict[str, float | int | str | bool | None]:
    """Return an empty, traceable SFA summary."""
    return {
        "sfa_syn_ttr": None,
        "sfa_ptf": None,
        "sfa_cdu": None,
        "ptf": None,
        "cdu": None,
        "sfa_source_lemmas_considered": 0,
        "sfa_source_lemmas_with_dictionary_entries": 0,
        "sfa_valid_source_lemmas": 0,
        "source_lemmas_considered": 0,
        "source_lemmas_with_matches": 0,
        "sentences_with_candidates": 0,
        "dictionary_entries": 0,
        "dictionary_path": None,
        "dictionary_resource": None,
        "dictionary_inverted": None,
    }


def _normalize_lexical_item(value: Any) -> str | None:
    if value is None:
        return None
    normalized = _WHITESPACE_RE.sub(" ", str(value)).strip().lower()
    return normalized or None


def _extract_apertium_text(node: ET.Element) -> str | None:
    parts: list[str] = []
    if node.text:
        parts.append(node.text)
    for child in node:
        if child.tag in {"b", "j", "a"}:
            parts.append(" ")
        elif child.text:
            parts.append(child.text)
        if child.tail:
            parts.append(child.tail)
    return _normalize_lexical_item("".join(parts))


def resolve_default_apertium_dictionary(target_lang: str) -> dict[str, Any]:
    """Return the default Apertium dictionary spec for a supported target language."""
    language_key = str(target_lang).strip().lower()
    if language_key not in DEFAULT_APERTIUM_DICTIONARIES:
        raise SFAAnalysisError(
            f"No default Apertium dictionary is configured for target language '{target_lang}'."
        )
    spec = dict(DEFAULT_APERTIUM_DICTIONARIES[language_key])
    path_candidates = [Path(candidate) for candidate in spec.pop("path_candidates")]
    for candidate in path_candidates:
        if candidate.exists():
            spec["path"] = candidate
            return spec
    spec["path"] = path_candidates[0]
    return spec


def load_apertium_translation_options(
    path: str | Path,
    *,
    invert: bool = False,
) -> dict[str, set[str]]:
    """Load translation options from an Apertium XML dictionary."""
    dictionary_path = Path(path)
    if not dictionary_path.exists():
        raise SFAAnalysisError(f"Apertium dictionary not found: {dictionary_path}")

    try:
        root = ET.parse(dictionary_path).getroot()
    except ET.ParseError as exc:
        raise SFAAnalysisError(
            f"Could not parse Apertium dictionary XML: {dictionary_path}"
        ) from exc

    translation_options: dict[str, set[str]] = {}
    for entry in root.iter("e"):
        left_forms = {
            value for node in entry.iter("l") if (value := _extract_apertium_text(node)) is not None
        }
        right_forms = {
            value for node in entry.iter("r") if (value := _extract_apertium_text(node)) is not None
        }
        if not left_forms or not right_forms:
            continue

        source_forms = right_forms if invert else left_forms
        target_forms = left_forms if invert else right_forms
        for source_form in source_forms:
            translation_options.setdefault(source_form, set()).update(target_forms)

    if not translation_options:
        raise SFAAnalysisError(
            f"No usable <l>/<r> bilingual entries were found in {dictionary_path}"
        )
    return translation_options


def _eligible_source_lemmas(
    annotations: list[dict[str, Any]],
    translation_options: dict[str, set[str]],
    source_upos: set[str],
) -> list[str]:
    seen: set[str] = set()
    source_lemmas: list[str] = []
    for token in annotations:
        if token.get("upos") not in source_upos:
            continue
        lemma = _normalize_lexical_item(token.get("lemma") or token.get("text"))
        if lemma is None or lemma not in translation_options or lemma in seen:
            continue
        source_lemmas.append(lemma)
        seen.add(lemma)
    return source_lemmas


def _target_lemma_counter(annotations: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for token in annotations:
        lemma = _normalize_lexical_item(token.get("lemma") or token.get("text"))
        if lemma is not None:
            counts[lemma] += 1
    return counts


def collect_sfa_option_counts(
    source_annotations: list[list[dict[str, Any]]],
    target_annotations: list[list[dict[str, Any]]],
    translation_options: dict[str, set[str]],
    *,
    source_upos: set[str] | None = None,
) -> dict[str, Any]:
    """Collect sentence-level dictionary-option counts for SFA metrics."""
    if len(source_annotations) != len(target_annotations):
        raise SFAAnalysisError("Source and target annotation lists must have the same length.")

    filtered_upos = set(source_upos or DEFAULT_SOURCE_UPOS)
    counts_by_source_lemma: dict[str, Counter[str]] = {}
    sentences_with_candidates = 0

    for source_sentence, target_sentence in zip(source_annotations, target_annotations):
        source_lemmas = _eligible_source_lemmas(source_sentence, translation_options, filtered_upos)
        if not source_lemmas:
            continue
        sentences_with_candidates += 1
        target_counts = _target_lemma_counter(target_sentence)
        for source_lemma in source_lemmas:
            option_counts = counts_by_source_lemma.setdefault(
                source_lemma,
                Counter({option: 0 for option in sorted(translation_options[source_lemma])}),
            )
            for option in translation_options[source_lemma]:
                option_counts[option] += target_counts.get(option, 0)

    return {
        "counts_by_source_lemma": counts_by_source_lemma,
        "sentences_with_candidates": sentences_with_candidates,
        "source_upos": sorted(filtered_upos),
    }


def _cosine_distance_from_uniform(counts: Counter[str], support: list[str]) -> float | None:
    if len(support) <= 1:
        return 0.0
    observed = [counts.get(item, 0) for item in support]
    total = sum(observed)
    if total == 0:
        return None
    observed_probs = [value / total for value in observed]
    uniform_value = 1.0 / len(support)
    uniform_probs = [uniform_value] * len(support)
    dot_product = sum(left * right for left, right in zip(observed_probs, uniform_probs))
    observed_norm = math.sqrt(sum(value * value for value in observed_probs))
    uniform_norm = math.sqrt(sum(value * value for value in uniform_probs))
    if observed_norm == 0.0 or uniform_norm == 0.0:
        return None
    cosine_similarity = dot_product / (observed_norm * uniform_norm)
    return 1.0 - cosine_similarity


def summarize_sfa_metrics(
    counts_by_source_lemma: dict[str, Counter[str]],
    *,
    sentences_with_candidates: int,
    dictionary_path: str | Path,
    dictionary_resource: str | None,
    dictionary_inverted: bool,
) -> dict[str, float | int | str | bool | None]:
    """Aggregate paper-style SFA metrics over valid source lemma choices."""
    summary = empty_sfa_summary()
    summary["dictionary_path"] = str(Path(dictionary_path))
    summary["dictionary_resource"] = dictionary_resource
    summary["dictionary_inverted"] = dictionary_inverted
    summary["dictionary_entries"] = len(counts_by_source_lemma)
    summary["sentences_with_candidates"] = sentences_with_candidates
    summary["sfa_source_lemmas_considered"] = len(counts_by_source_lemma)
    summary["source_lemmas_considered"] = len(counts_by_source_lemma)

    valid_counts: list[Counter[str]] = []
    matched_lemma_count = 0
    for option_counts in counts_by_source_lemma.values():
        if len(option_counts) < 2:
            continue
        total_occurrences = sum(option_counts.values())
        if total_occurrences <= 0:
            continue
        matched_lemma_count += 1
        valid_counts.append(option_counts)

    summary["sfa_source_lemmas_with_dictionary_entries"] = len(counts_by_source_lemma)
    summary["source_lemmas_with_matches"] = matched_lemma_count
    summary["sfa_valid_source_lemmas"] = len(valid_counts)
    if not valid_counts:
        return summary

    ptf_scores = [max(counts.values()) / sum(counts.values()) for counts in valid_counts]
    cdu_scores = [
        score
        for counts in valid_counts
        if (score := _cosine_distance_from_uniform(counts, sorted(counts))) is not None
    ]
    observed_option_counts: Counter[str] = Counter()
    for counts in valid_counts:
        observed_option_counts.update(
            {option: count for option, count in counts.items() if count > 0}
        )
    total_option_occurrences = sum(observed_option_counts.values())

    summary["sfa_syn_ttr"] = (
        len(observed_option_counts) / total_option_occurrences if total_option_occurrences else None
    )
    summary["sfa_ptf"] = sum(ptf_scores) / len(ptf_scores)
    summary["sfa_cdu"] = sum(cdu_scores) / len(cdu_scores) if cdu_scores else None
    summary["ptf"] = summary["sfa_ptf"]
    summary["cdu"] = summary["sfa_cdu"]
    return summary
