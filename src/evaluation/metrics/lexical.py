"""Compute lexical and morphological diversity."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any


def compute_shannon_index(counts: Counter[str]) -> float | None:
    """Compute Shannon's H over a frequency dictionary."""
    if not counts:
        return None
    total = sum(counts.values())
    return -sum((count / total) * math.log(count / total) for count in counts.values() if count)


def compute_simpson_index(counts: Counter[str]) -> float | None:
    """Compute Simpson's concentration index as in evaluation.py: sum(p^2)."""
    if not counts:
        return None
    total = sum(counts.values())
    return sum((count / total) ** 2 for count in counts.values() if count)


def compute_inverse_simpson_index(counts: Counter[str]) -> float | None:
    """Compute the inverse Simpson index."""
    simpson = compute_simpson_index(counts)
    if simpson in (None, 0.0):
        return None
    return 1.0 / simpson


def _safe_diversity_from_counts(counts: Counter[str]) -> dict[str, float | None]:
    if not counts:
        return {
            "unique": 0,
            "shannon": None,
            "simpson": None,
            "inverse_simpson": None,
        }
    return {
        "unique": len(counts),
        "shannon": compute_shannon_index(counts),
        "simpson": compute_simpson_index(counts),
        "inverse_simpson": compute_inverse_simpson_index(counts),
    }


def compute_annotation_diversity_metrics(
    annotations: list[list[dict[str, Any]]],
) -> dict[str, float | int | None]:
    """Compute diversity metrics from token-level linguistic annotations."""
    word_forms = Counter()
    lemmas = Counter()
    morphological_features = Counter()

    for sentence in annotations:
        for token in sentence:
            text = token.get("text")
            lemma = token.get("lemma")
            feats = token.get("feats")
            if text:
                word_forms[str(text)] += 1
            if lemma:
                lemmas[str(lemma)] += 1
            if feats:
                morphological_features[str(feats)] += 1

    word_form_metrics = _safe_diversity_from_counts(word_forms)
    lemma_metrics = _safe_diversity_from_counts(lemmas)
    feature_metrics = _safe_diversity_from_counts(morphological_features)
    return {
        "unique_word_forms": word_form_metrics["unique"],
        "unique_lemmas": lemma_metrics["unique"],
        "word_form_shannon_diversity": word_form_metrics["shannon"],
        "word_form_simpson_diversity": word_form_metrics["simpson"],
        "word_form_inverse_simpson_diversity": word_form_metrics["inverse_simpson"],
        "lemma_shannon_diversity": lemma_metrics["shannon"],
        "lemma_simpson_diversity": lemma_metrics["simpson"],
        "lemma_inverse_simpson_diversity": lemma_metrics["inverse_simpson"],
        "morph_feature_inventory_size": feature_metrics["unique"],
        "morph_feature_shannon_diversity": feature_metrics["shannon"],
        "morph_feature_simpson_diversity": feature_metrics["simpson"],
        "morph_feature_inverse_simpson_diversity": feature_metrics["inverse_simpson"],
    }


EXCLUDED_MORPH_UPOS = frozenset({"PUNCT", "SYM", "NUM"})


def compute_lemma_wordform_diversity(
    annotations: list[list[dict[str, Any]]],
) -> dict[str, float | int | None]:
    """Compute paper-style morphological diversity over wordforms per lemma."""
    forms_by_lemma: dict[str, Counter[str]] = {}
    for sentence in annotations:
        for token in sentence:
            if token.get("upos") in EXCLUDED_MORPH_UPOS:
                continue
            lemma = token.get("lemma") or token.get("text")
            wordform = token.get("text")
            if not lemma or not wordform:
                continue
            normalized_lemma = str(lemma).strip().lower()
            normalized_wordform = str(wordform).strip().lower()
            if not normalized_lemma or not normalized_wordform:
                continue
            forms_by_lemma.setdefault(normalized_lemma, Counter())[normalized_wordform] += 1

    multi_wordform = [counts for counts in forms_by_lemma.values() if len(counts) >= 2]
    shannon_scores = [compute_shannon_index(counts) for counts in multi_wordform]
    simpson_scores = [compute_simpson_index(counts) for counts in multi_wordform]
    inverse_scores = [compute_inverse_simpson_index(counts) for counts in multi_wordform]
    return {
        "morph_shannon_entropy": (
            sum(score for score in shannon_scores if score is not None) / len(shannon_scores)
            if shannon_scores
            else None
        ),
        "morph_simpson_d": (
            sum(score for score in simpson_scores if score is not None) / len(simpson_scores)
            if simpson_scores
            else None
        ),
        "morph_inverse_simpson_optional": (
            sum(score for score in inverse_scores if score is not None) / len(inverse_scores)
            if inverse_scores
            else None
        ),
        "morph_lemmas_used_for_entropy": len(multi_wordform),
        "morph_single_wordform_lemmas": sum(
            1 for counts in forms_by_lemma.values() if len(counts) == 1
        ),
        "morph_multi_wordform_lemmas": len(multi_wordform),
    }
