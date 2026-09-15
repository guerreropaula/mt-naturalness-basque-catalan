"""Stanza syntactic and morphological analysis."""

from __future__ import annotations

import logging
from typing import Any

from src.evaluation.lexical_diversity import (
    compute_annotation_diversity_metrics,
    compute_lemma_wordform_diversity,
)
from src.evaluation.metrics import compute_ttr

logger = logging.getLogger(__name__)

CONTENT_UPOS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
FUNCTION_UPOS = {"DET", "PRON", "ADP", "AUX", "CCONJ", "SCONJ", "PART"}


class StanzaAnalysisError(RuntimeError):
    """Raised when Stanza-based annotation cannot be completed."""


def load_stanza_pipeline(lang: str, use_gpu: bool = False) -> Any:
    """Load a Stanza pipeline for tokenization, POS, lemma, morphology, and dependencies."""
    try:  # pragma: no cover - dependent on local Stanza resources
        import stanza
    except Exception as exc:  # pragma: no cover
        raise StanzaAnalysisError("stanza is required for syntactic analysis") from exc

    try:  # pragma: no cover
        return stanza.Pipeline(
            lang=lang,
            processors="tokenize,pos,lemma,depparse",
            tokenize_no_ssplit=False,
            use_gpu=use_gpu,
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover
        raise StanzaAnalysisError(
            f"Could not initialize a Stanza pipeline for language '{lang}'. "
            "Make sure the corresponding Stanza model has been downloaded."
        ) from exc


def annotate_texts(
    texts: list[str],
    lang: str,
    pipeline: Any | None = None,
) -> list[list[dict[str, Any]]]:
    """Annotate a list of texts with Stanza and return flattened token records per text."""
    pipeline = pipeline or load_stanza_pipeline(lang)
    annotated_texts: list[list[dict[str, Any]]] = []
    for text in texts:
        try:  # pragma: no cover - depends on runtime model state
            document = pipeline(text)
        except Exception as exc:  # pragma: no cover
            raise StanzaAnalysisError(f"Stanza annotation failed for text: {text!r}") from exc
        sentence_tokens: list[dict[str, Any]] = []
        for sentence in document.sentences:
            for word in sentence.words:
                sentence_tokens.append(
                    {
                        "text": word.text,
                        "lemma": getattr(word, "lemma", None),
                        "upos": getattr(word, "upos", None),
                        "feats": getattr(word, "feats", None),
                        "deprel": getattr(word, "deprel", None),
                    }
                )
        annotated_texts.append(sentence_tokens)
    return annotated_texts


def compute_lexical_density_metrics(
    annotations: list[list[dict[str, Any]]],
) -> dict[str, float | None]:
    """Compute lexical density and function/content ratios from UPOS tags."""
    content_count = 0
    function_count = 0
    valid_tokens = 0

    for sentence in annotations:
        for token in sentence:
            upos = token.get("upos")
            if not upos:
                continue
            valid_tokens += 1
            if upos in CONTENT_UPOS:
                content_count += 1
            elif upos in FUNCTION_UPOS:
                function_count += 1

    if valid_tokens == 0:
        return {
            "lexical_density": None,
            "function_to_content_ratio": None,
            "content_to_function_ratio": None,
            "function_word_proportion": None,
        }

    lexical_density = content_count / valid_tokens
    function_word_proportion = function_count / valid_tokens
    function_to_content_ratio = (function_count / content_count) if content_count else None
    content_to_function_ratio = (content_count / function_count) if function_count else None
    return {
        "lexical_density": lexical_density,
        "function_to_content_ratio": function_to_content_ratio,
        "content_to_function_ratio": content_to_function_ratio,
        "function_word_proportion": function_word_proportion,
    }


def compute_upos_deprel_ttr(
    annotations: list[list[dict[str, Any]]],
) -> float | None:
    """Compute a syntax-oriented TTR proxy over `UPOS:DEPREL` sequences."""
    syntactic_tokens: list[str] = []
    for sentence in annotations:
        for token in sentence:
            upos = token.get("upos")
            deprel = token.get("deprel")
            if upos and deprel:
                syntactic_tokens.append(f"{upos}:{deprel}")
    return compute_ttr(syntactic_tokens)


def compute_syn_ttr(
    annotations: list[list[dict[str, Any]]],
) -> float | None:
    """Backward-compatible alias for the older, ambiguous metric name."""
    return compute_upos_deprel_ttr(annotations)


def compute_stanza_summary_metrics(
    annotations: list[list[dict[str, Any]]],
) -> dict[str, float | int | None]:
    """Compute the Stanza-enabled metric block used by analysis."""
    metrics = {}
    metrics.update(compute_lexical_density_metrics(annotations))
    metrics["upos_deprel_ttr"] = compute_upos_deprel_ttr(annotations)
    metrics.update(compute_annotation_diversity_metrics(annotations))
    metrics.update(compute_lemma_wordform_diversity(annotations))
    return metrics
