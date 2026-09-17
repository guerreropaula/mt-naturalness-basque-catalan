"""ASTrED analysis."""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any

import pandas as pd

from src.utils.errors import PipelineError


def _install_nltk_draw_tree_stub_if_needed() -> None:
    """Allow ASTrED to import on headless Python builds without Tkinter."""
    try:
        import tkinter  # noqa: F401
    except Exception:
        if "nltk.draw.tree" not in sys.modules:
            stub = types.ModuleType("nltk.draw.tree")

            def draw_trees(*_args: Any, **_kwargs: Any) -> None:
                raise RuntimeError(
                    "nltk.draw.tree is unavailable because Tkinter is not installed."
                )

            stub.draw_trees = draw_trees
            sys.modules["nltk.draw.tree"] = stub


def _load_astred_backend() -> tuple[Any, Any, Any]:
    try:
        _install_nltk_draw_tree_stub_if_needed()
        astred_module = importlib.import_module("astred")
        astred_utils = importlib.import_module("astred.utils")
    except Exception as exc:
        raise PipelineError(
            "astred is required for ASTrED analysis. Install it with `pip install astred`."
        ) from exc

    try:
        sentence_cls = astred_module.Sentence
        aligned_sentences_cls = astred_module.AlignedSentences
        load_parser = astred_utils.load_parser
    except AttributeError as exc:
        raise PipelineError(
            "The installed astred package does not expose the expected Sentence/AlignedSentences API."
        ) from exc

    return sentence_cls, aligned_sentences_cls, load_parser


def _build_parsers(
    load_parser: Any,
    source_model_or_lang: str,
    target_model_or_lang: str,
    parser: str | None,
    use_gpu: bool,
) -> tuple[Any, Any]:
    parser_kwargs: dict[str, Any] = {}
    if parser == "stanza":
        # Reuse installed Stanza models so cluster jobs do not need network access.
        parser_kwargs["download_method"] = None

    try:
        source_parser = load_parser(
            source_model_or_lang,
            parser,
            is_tokenized=False,
            use_gpu=use_gpu,
            **parser_kwargs,
        )
        if target_model_or_lang == source_model_or_lang:
            target_parser = source_parser
        else:
            target_parser = load_parser(
                target_model_or_lang,
                parser,
                is_tokenized=False,
                use_gpu=use_gpu,
                **parser_kwargs,
            )
    except Exception as exc:
        raise PipelineError(
            "Could not initialize ASTrED parsing resources. "
            "For `parser='stanza'`, pass a language code such as `ca` or `eu`. "
            "For `parser='spacy'`, pass an installed model name such as `en_core_web_sm`."
        ) from exc

    return source_parser, target_parser


def _normalize_alignment_value(value: Any) -> Any | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, (list, tuple)):
        return list(value) if value else None
    if pd.isna(value):
        return None
    return value


def _serialize_alignment_value(value: Any) -> str | None:
    normalized = _normalize_alignment_value(value)
    if normalized is None:
        return None
    if isinstance(normalized, str):
        return normalized
    if isinstance(normalized, list):
        parts = []
        for item in normalized:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                parts.append(f"{item[0]}-{item[1]}")
            else:
                parts.append(str(item))
        return " ".join(parts)
    return str(normalized)


def _aligned_ratio(sentence: Any) -> float | None:
    words = list(getattr(sentence, "no_null_words", []))
    if not words:
        return None
    aligned_count = sum(1 for word in words if getattr(word, "is_aligned", False))
    return aligned_count / len(words)


def _non_null_pair_count(aligned_sentences: Any) -> int:
    if hasattr(aligned_sentences, "no_null_word_pairs"):
        return len(aligned_sentences.no_null_word_pairs)

    aligned_words = getattr(aligned_sentences, "aligned_words", [])
    return sum(
        1
        for pair in aligned_words
        if not getattr(pair.src, "is_null", False) and not getattr(pair.tgt, "is_null", False)
    )


def empty_astred_summary() -> dict[str, float | int | str | None]:
    """Return ASTrED fields with unavailable values."""
    return {
        "ted": None,
        "word_cross": None,
        "seq_cross": None,
        "sacr_cross": None,
        "aligned_word_pairs": None,
        "aligned_source_ratio": None,
        "aligned_target_ratio": None,
        "rows_total": 0,
        "rows_eligible": 0,
        "rows_excluded_sentence_count_mismatch": 0,
        "rows_scored": 0,
        "rows_scored_multisentence": 0,
        "sentence_pairs_scored": 0,
        "rows_failed": 0,
        "parser": None,
        "source_column": None,
        "target_column": None,
        "alignment_mode": None,
        "top_error": None,
    }


def summarize_astred_metrics(
    per_example: pd.DataFrame,
    *,
    parser: str | None,
    source_column: str,
    target_column: str,
    alignment_column: str | None,
) -> dict[str, float | int | str | None]:
    """Aggregate row-level ASTrED metrics into a corpus summary."""
    summary = empty_astred_summary()
    metric_columns = {
        "ted": "astred_ted",
        "word_cross": "astred_word_cross",
        "seq_cross": "astred_seq_cross",
        "sacr_cross": "astred_sacr_cross",
        "aligned_word_pairs": "astred_aligned_word_pairs",
        "aligned_source_ratio": "astred_aligned_source_ratio",
        "aligned_target_ratio": "astred_aligned_target_ratio",
    }
    for summary_key, column_name in metric_columns.items():
        values = per_example[column_name].dropna()
        summary[summary_key] = float(values.mean()) if not values.empty else None

    statuses = per_example["astred_status"]
    summary["rows_total"] = int(len(per_example))
    summary["rows_eligible"] = int((statuses != "excluded_sentence_count_mismatch").sum())
    summary["rows_excluded_sentence_count_mismatch"] = int(
        (statuses == "excluded_sentence_count_mismatch").sum()
    )
    summary["rows_scored"] = int(statuses.str.startswith("scored").sum())
    summary["rows_scored_multisentence"] = int((statuses == "scored_multisentence").sum())
    summary["sentence_pairs_scored"] = int(per_example["astred_sentence_pairs"].sum())
    errors = per_example.loc[statuses == "failed", "astred_error"].dropna()
    summary["rows_failed"] = int(errors.count())
    if not errors.empty:
        top_error, top_error_count = next(iter(errors.value_counts().items()))
        summary["top_error"] = f"{top_error} ({int(top_error_count)} rows)"
    summary["parser"] = parser
    summary["source_column"] = source_column
    summary["target_column"] = target_column
    summary["alignment_mode"] = "provided" if alignment_column else "automatic"
    return summary


def _parse_sentence_units(sentence_cls: Any, text: str, parser: Any) -> list[Any]:
    """Parse text once and convert each native parser sentence to ASTrED's type."""
    document = parser(text)
    if hasattr(document, "sentences"):
        parsed_sentences = list(document.sentences)
    else:
        parsed_sentences = list(document.sents)
    if not parsed_sentences:
        raise ValueError("The parser produced no sentences.")
    return [sentence_cls.from_parser(sentence) for sentence in parsed_sentences]


def _sentence_weight(sentence: Any) -> int:
    """Use parsed token count to weight sentence-level metrics within a row."""
    return max(len(getattr(sentence, "no_null_words", [])), 1)


def _weighted_mean(values: list[float | int | None], weights: list[int]) -> float | None:
    present = [
        (float(value), weight) for value, weight in zip(values, weights) if value is not None
    ]
    if not present:
        return None
    total_weight = sum(weight for _, weight in present)
    return sum(value * weight for value, weight in present) / total_weight


def compute_astred_per_example(
    df: pd.DataFrame,
    *,
    source_model_or_lang: str,
    target_model_or_lang: str | None = None,
    parser: str | None = "stanza",
    source_column: str = "prediction",
    target_column: str = "reference",
    alignment_column: str | None = None,
    use_gpu: bool = False,
    on_multiple: str = "raise",
) -> pd.DataFrame:
    """Compute ASTrED metrics per row, including matched multi-sentence rows.

    ASTrED itself compares one dependency tree pair at a time. When both sides
    contain the same number of parser-detected sentences, this wrapper scores
    pairs by position and aggregates their metrics with parsed-token weights.
    Rows with unequal sentence counts are retained as explicit exclusions.
    """
    del on_multiple  # Kept for backwards-compatible callers; all sentences are scored here.
    if source_column not in df.columns:
        raise ValueError(
            f"The DataFrame does not contain the ASTrED source column '{source_column}'"
        )
    if target_column not in df.columns:
        raise ValueError(
            f"The DataFrame does not contain the ASTrED target column '{target_column}'"
        )
    if alignment_column and alignment_column not in df.columns:
        raise ValueError(
            f"The DataFrame does not contain the ASTrED alignment column '{alignment_column}'"
        )

    sentence_cls, aligned_sentences_cls, load_parser = _load_astred_backend()
    target_model_or_lang = target_model_or_lang or source_model_or_lang
    source_parser, target_parser = _build_parsers(
        load_parser,
        source_model_or_lang=source_model_or_lang,
        target_model_or_lang=target_model_or_lang,
        parser=parser,
        use_gpu=use_gpu,
    )

    source_texts = df[source_column].fillna("").astype(str).tolist()
    target_texts = df[target_column].fillna("").astype(str).tolist()
    alignments = df[alignment_column].tolist() if alignment_column else [None] * len(df)

    rows: list[dict[str, Any]] = []
    for source_text, target_text, alignment_value in zip(source_texts, target_texts, alignments):
        row = {
            "astred_ted": None,
            "astred_word_cross": None,
            "astred_seq_cross": None,
            "astred_sacr_cross": None,
            "astred_aligned_word_pairs": None,
            "astred_aligned_source_ratio": None,
            "astred_aligned_target_ratio": None,
            "astred_word_aligns": None,
            "astred_sentence_pairs": 0,
            "astred_error": None,
            "astred_status": "scored",
        }
        try:
            source_sentences = _parse_sentence_units(sentence_cls, source_text, source_parser)
            target_sentences = _parse_sentence_units(sentence_cls, target_text, target_parser)
            if len(source_sentences) != len(target_sentences):
                row["astred_status"] = "excluded_sentence_count_mismatch"
                row["astred_error"] = (
                    "Sentence count mismatch: "
                    f"{source_column} has {len(source_sentences)}, "
                    f"{target_column} has {len(target_sentences)}."
                )
                rows.append(row)
                continue

            normalized_alignment = _normalize_alignment_value(alignment_value)
            if normalized_alignment is not None and len(source_sentences) > 1:
                raise ValueError(
                    "Provided word alignments are only supported for single-sentence ASTrED rows."
                )

            pair_metrics: list[dict[str, Any]] = []
            for source_sentence, target_sentence in zip(source_sentences, target_sentences):
                aligned_sentences = aligned_sentences_cls(
                    source_sentence,
                    target_sentence,
                    word_aligns=normalized_alignment,
                )
                pair_metrics.append(
                    {
                        "weight": max(
                            _sentence_weight(source_sentence), _sentence_weight(target_sentence)
                        ),
                        "ted": getattr(aligned_sentences, "ted", None),
                        "word_cross": getattr(aligned_sentences, "word_cross", None),
                        "seq_cross": getattr(aligned_sentences, "seq_cross", None),
                        "sacr_cross": getattr(aligned_sentences, "sacr_cross", None),
                        "aligned_word_pairs": _non_null_pair_count(aligned_sentences),
                        "aligned_source_ratio": _aligned_ratio(source_sentence),
                        "aligned_target_ratio": _aligned_ratio(target_sentence),
                        "word_aligns": getattr(
                            aligned_sentences,
                            "giza_word_aligns",
                            _serialize_alignment_value(normalized_alignment),
                        ),
                    }
                )
            weights = [item["weight"] for item in pair_metrics]
            row.update(
                {
                    "astred_ted": _weighted_mean([item["ted"] for item in pair_metrics], weights),
                    "astred_word_cross": _weighted_mean(
                        [item["word_cross"] for item in pair_metrics], weights
                    ),
                    "astred_seq_cross": _weighted_mean(
                        [item["seq_cross"] for item in pair_metrics], weights
                    ),
                    "astred_sacr_cross": _weighted_mean(
                        [item["sacr_cross"] for item in pair_metrics], weights
                    ),
                    "astred_aligned_word_pairs": _weighted_mean(
                        [item["aligned_word_pairs"] for item in pair_metrics], weights
                    ),
                    "astred_aligned_source_ratio": _weighted_mean(
                        [item["aligned_source_ratio"] for item in pair_metrics], weights
                    ),
                    "astred_aligned_target_ratio": _weighted_mean(
                        [item["aligned_target_ratio"] for item in pair_metrics], weights
                    ),
                    "astred_word_aligns": " || ".join(
                        str(item["word_aligns"])
                        for item in pair_metrics
                        if item["word_aligns"] is not None
                    )
                    or None,
                    "astred_sentence_pairs": len(pair_metrics),
                    "astred_status": (
                        "scored_multisentence" if len(pair_metrics) > 1 else "scored"
                    ),
                }
            )
        except Exception as exc:
            row["astred_error"] = str(exc)
            row["astred_status"] = "failed"
        rows.append(row)

    return pd.DataFrame(rows, index=df.index)


def compute_astred_metrics(
    df: pd.DataFrame,
    *,
    source_model_or_lang: str,
    target_model_or_lang: str | None = None,
    parser: str | None = "stanza",
    source_column: str = "prediction",
    target_column: str = "reference",
    alignment_column: str | None = None,
    use_gpu: bool = False,
    on_multiple: str = "raise",
) -> dict[str, Any]:
    """Compute ASTrED per-example outputs plus corpus-level summary metrics."""
    per_example = compute_astred_per_example(
        df,
        source_model_or_lang=source_model_or_lang,
        target_model_or_lang=target_model_or_lang,
        parser=parser,
        source_column=source_column,
        target_column=target_column,
        alignment_column=alignment_column,
        use_gpu=use_gpu,
        on_multiple=on_multiple,
    )
    summary = summarize_astred_metrics(
        per_example,
        parser=parser,
        source_column=source_column,
        target_column=target_column,
        alignment_column=alignment_column,
    )
    return {"per_example": per_example, "summary": summary}
