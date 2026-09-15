"""Baseline translationese and adequacy analysis."""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.astred_analysis import (
    ASTrEDAnalysisError,
    compute_astred_metrics,
    empty_astred_summary,
)
from src.evaluation.metrics import (
    compute_automatic_metrics,
    compute_optional_metricx,
    compute_per_example_metrics,
    compute_target_lexical_metrics,
)
from src.evaluation.metrics import (
    DEFAULT_COMETKIWI_MODEL,
    DEFAULT_METRICX_MODEL,
    DEFAULT_XCOMET_XXL_MODEL,
)
from src.evaluation.sfa import (
    SFAAnalysisError,
    collect_sfa_option_counts,
    empty_sfa_summary,
    load_apertium_translation_options,
    resolve_default_apertium_dictionary,
    summarize_sfa_metrics,
)
from src.evaluation.syntactic_analysis import (
    StanzaAnalysisError,
    annotate_texts,
    compute_stanza_summary_metrics,
)
from src.utils.io import save_dataframe_csv, save_dataframe_jsonl, save_dataframe_tsv, save_json

logger = logging.getLogger(__name__)

_ASTRED_COLUMNS = (
    "astred_ted",
    "astred_word_cross",
    "astred_seq_cross",
    "astred_sacr_cross",
    "astred_aligned_word_pairs",
    "astred_aligned_source_ratio",
    "astred_aligned_target_ratio",
    "astred_word_aligns",
    "astred_sentence_pairs",
    "astred_error",
    "astred_status",
)
_ASTRED_SOURCE_PREDICTION_COLUMNS = tuple(
    column.replace("astred_", "astred_source_prediction_", 1)
    for column in _ASTRED_COLUMNS
)
_ASTRED_SOURCE_REFERENCE_COLUMNS = tuple(
    column.replace("astred_", "astred_source_reference_", 1)
    for column in _ASTRED_COLUMNS
)


def _rename_astred_columns(
    per_example: pd.DataFrame, *, prefix: str
) -> pd.DataFrame:
    """Give a second ASTrED pass an unambiguous, non-overlapping schema."""
    return per_example.rename(
        columns={column: column.replace("astred_", prefix, 1) for column in _ASTRED_COLUMNS}
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _first_non_null_value(df: pd.DataFrame, column: str) -> Any:
    if column not in df.columns:
        return None
    values = df[column].dropna()
    if values.empty:
        return None
    return values.iloc[0]


def _unique_values(df: pd.DataFrame, column: str, limit: int = 20) -> list[Any]:
    if column not in df.columns:
        return []
    values = []
    for value in df[column].dropna().tolist():
        if value not in values:
            values.append(value)
        if len(values) >= limit:
            break
    return values


def _load_generation_metadata(predictions_path: Path) -> dict[str, Any] | None:
    metadata_path = predictions_path.resolve().parent / "run_metadata.json"
    if not metadata_path.exists():
        return None
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read generation metadata from %s: %s", metadata_path, exc)
        return {"metadata_path": str(metadata_path), "read_error": str(exc)}


def _prediction_traceability(
    predictions_df: pd.DataFrame, predictions_path: Path
) -> dict[str, Any]:
    prompt_columns = [
        column
        for column in ("prompt", "initial_prompt", "refinement_prompt", "prompts")
        if column in predictions_df.columns
    ]
    prompt_samples: dict[str, Any] = {}
    if not predictions_df.empty:
        first_row = predictions_df.iloc[0]
        for column in prompt_columns:
            value = first_row.get(column)
            prompt_samples[column] = value

    generation_config = _first_non_null_value(predictions_df, "generation_config")
    return {
        "predictions_path": str(predictions_path),
        "predictions_file_size_bytes": predictions_path.stat().st_size
        if predictions_path.exists()
        else None,
        "row_count": int(len(predictions_df)),
        "columns": list(predictions_df.columns),
        "model_key": _first_non_null_value(predictions_df, "model_key"),
        "model_id": _first_non_null_value(predictions_df, "model_id"),
        "experiment_keys": _unique_values(predictions_df, "experiment_key"),
        "experiment_families": _unique_values(predictions_df, "experiment_family"),
        "experiment_tracks": _unique_values(predictions_df, "experiment_track"),
        "prompt_variants": _unique_values(predictions_df, "prompt_variant"),
        "initial_prompt_variants": _unique_values(predictions_df, "initial_prompt_variant"),
        "refinement_prompt_variants": _unique_values(predictions_df, "refinement_prompt_variant"),
        "prompt_columns": prompt_columns,
        "prompt_samples_first_row": prompt_samples,
        "generation_config_first_row": generation_config,
    }


def _analysis_metadata(
    *,
    predictions_df: pd.DataFrame,
    predictions_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    started_at_utc: str,
    finished_at_utc: str,
    runtime_seconds: float,
) -> dict[str, Any]:
    return {
        "analysis_started_at_utc": started_at_utc,
        "analysis_finished_at_utc": finished_at_utc,
        "analysis_runtime_seconds": runtime_seconds,
        "analysis_module": "src.evaluation.analyze_translationese",
        "output_dir": str(output_dir),
        "input": _prediction_traceability(predictions_df, predictions_path),
        "generation_run_metadata": _load_generation_metadata(predictions_path),
        "metric_requests": {
            "comet_model": args.comet_model,
            "xcomet_model": args.xcomet_model,
            "comet_qe_model": args.comet_qe_model,
            "cometkiwi_model": args.cometkiwi_model,
            "comet_batch_size": args.comet_batch_size,
            "comet_gpus": args.comet_gpus,
            "background_file": args.background_file,
            "background_column": args.background_column,
            "stanza_lang": args.stanza_lang,
            "sfa_target_lang": args.sfa_target_lang,
            "sfa_source_lang": args.sfa_source_lang,
            "sfa_dictionary_path": args.sfa_dictionary_path,
            "sfa_invert_dictionary": bool(args.sfa_invert_dictionary),
            "astred_model_or_lang": args.astred_model_or_lang,
            "astred_parser": args.astred_parser,
            "astred_source_column": args.astred_source_column,
            "astred_target_column": args.astred_target_column,
            "astred_alignment_column": args.astred_alignment_column,
            "astred_source_prediction": bool(args.astred_source_prediction),
            "astred_source_prediction_source_model_or_lang": (
                args.astred_source_prediction_source_model_or_lang
            ),
            "astred_source_prediction_target_model_or_lang": (
                args.astred_source_prediction_target_model_or_lang
            ),
            "astred_source_prediction_source_column": (
                args.astred_source_prediction_source_column
            ),
            "astred_source_prediction_target_column": (
                args.astred_source_prediction_target_column
            ),
        },
    }


def _empty_stanza_summary() -> dict[str, float | int | None]:
    return {
        "lexical_density": None,
        "function_to_content_ratio": None,
        "content_to_function_ratio": None,
        "function_word_proportion": None,
        "upos_deprel_ttr": None,
        "unique_lemmas": None,
        "morph_feature_inventory_size": None,
    }


def load_predictions(path: str | Path) -> pd.DataFrame:
    """Load baseline predictions from JSONL or Parquet."""
    predictions_path = Path(path)
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
    if predictions_path.suffix == ".parquet":
        return pd.read_parquet(predictions_path)
    return pd.read_json(predictions_path, lines=True)


def load_background_texts(path: str | Path, text_column: str) -> list[str]:
    """Load a background text corpus for lexical frequency profiling."""
    background_path = Path(path)
    if not background_path.exists():
        raise FileNotFoundError(f"Background file not found: {background_path}")
    if background_path.suffix == ".parquet":
        df = pd.read_parquet(background_path)
    else:
        df = pd.read_json(background_path, lines=True)
    if text_column not in df.columns:
        raise ValueError(
            f"Background file {background_path} does not contain text column '{text_column}'"
        )
    return df[text_column].fillna("").astype(str).tolist()


def _flatten_summary(summary: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for section_name, section_value in summary.items():
        if section_name == "report_context":
            continue
        if isinstance(section_value, dict):
            for metric_name, metric_value in section_value.items():
                rows.append(
                    {
                        "section": section_name,
                        "metric": metric_name,
                        "value": metric_value,
                    }
                )
        else:
            rows.append({"section": "meta", "metric": section_name, "value": section_value})
    return pd.DataFrame(rows)


_REPORT_SECTIONS: list[tuple[str, str, list[tuple[str, str]]]] = [
    (
        "Adequacy",
        "adequacy",
        [
            ("bleu", "BLEU"),
            ("ter", "TER"),
            ("chrf_pp", "chrF++"),
            ("comet", "COMET"),
            ("xcomet", "xCOMET"),
            ("cometkiwi", "COMETKiwi"),
            ("metricx", "MetricX-24"),
        ],
    ),
    (
        "Reference Naturalness: Lexical Diversity",
        "reference_lexical_diversity",
        [
            ("token_count", "Tokens"),
            ("type_count", "Types"),
            ("ttr", "TTR"),
            ("mtld", "MTLD"),
            ("mattr_50", "MATTR (window 50)"),
            ("mtld_status", "MTLD status"),
            ("yules_k", "Yule's K"),
            ("yules_i", "Yule's I"),
        ],
    ),
    (
        "Prediction Naturalness: Lexical Diversity",
        "prediction_lexical_diversity",
        [
            ("token_count", "Tokens"),
            ("type_count", "Types"),
            ("ttr", "TTR"),
            ("mtld", "MTLD"),
            ("mattr_50", "MATTR (window 50)"),
            ("mtld_status", "MTLD status"),
            ("yules_k", "Yule's K"),
            ("yules_i", "Yule's I"),
        ],
    ),
    (
        "Reference Naturalness: Lexical Frequency Profile",
        "reference_lexical_frequency_profile",
        [
            ("b1_top_1000", "B1: top 1,000"),
            ("b2_1001_2000", "B2: ranks 1,001-2,000"),
            ("b3_beyond_2000", "B3: beyond 2,000 or unseen"),
            ("beyond_2000", "Beyond 2,000"),
        ],
    ),
    (
        "Prediction Naturalness: Lexical Frequency Profile",
        "prediction_lexical_frequency_profile",
        [
            ("b1_top_1000", "B1: top 1,000"),
            ("b2_1001_2000", "B2: ranks 1,001-2,000"),
            ("b3_beyond_2000", "B3: beyond 2,000 or unseen"),
            ("beyond_2000", "Beyond 2,000"),
        ],
    ),
    (
        "Reference Naturalness: Morphological Diversity",
        "reference_morphological_diversity",
        [
            ("morph_shannon_entropy", "Shannon entropy"),
            ("morph_simpson_d", "Simpson D"),
            ("morph_inverse_simpson_optional", "Inverse Simpson (optional)"),
            ("morph_lemmas_used_for_entropy", "Lemmas used for entropy"),
            ("morph_single_wordform_lemmas", "Single-wordform lemmas"),
            ("morph_multi_wordform_lemmas", "Multi-wordform lemmas"),
        ],
    ),
    (
        "Prediction Naturalness: Morphological Diversity",
        "prediction_morphological_diversity",
        [
            ("morph_shannon_entropy", "Shannon entropy"),
            ("morph_simpson_d", "Simpson D"),
            ("morph_inverse_simpson_optional", "Inverse Simpson (optional)"),
            ("morph_lemmas_used_for_entropy", "Lemmas used for entropy"),
            ("morph_single_wordform_lemmas", "Single-wordform lemmas"),
            ("morph_multi_wordform_lemmas", "Multi-wordform lemmas"),
        ],
    ),
    (
        "Reference Naturalness: Synonym Frequency Analysis",
        "reference_synonym_frequency_analysis",
        [
            ("sfa_syn_ttr", "SynTTR"),
            ("sfa_ptf", "PTF"),
            ("sfa_cdu", "CDU"),
            ("sfa_valid_source_lemmas", "Valid source lemmas"),
            ("sfa_source_lemmas_with_dictionary_entries", "Source lemmas with dictionary entries"),
        ],
    ),
    (
        "Prediction Naturalness: Synonym Frequency Analysis",
        "prediction_synonym_frequency_analysis",
        [
            ("sfa_syn_ttr", "SynTTR"),
            ("sfa_ptf", "PTF"),
            ("sfa_cdu", "CDU"),
            ("sfa_valid_source_lemmas", "Valid source lemmas"),
            ("sfa_source_lemmas_with_dictionary_entries", "Source lemmas with dictionary entries"),
        ],
    ),
    (
        "Lexical Diversity",
        "lexical_diversity",
        [
            ("ttr", "TTR"),
            ("mattr_50", "MATTR (window 50)"),
            ("mtld", "MTLD"),
            ("yules_k", "Yule's K"),
            ("yules_i", "Yule's I"),
        ],
    ),
    (
        "Morphological Diversity Proxy",
        "morphological_diversity_proxy",
        [
            ("unique_word_forms", "Unique word forms"),
            ("shannon_diversity", "Shannon diversity"),
            ("simpson_diversity", "Simpson diversity"),
            ("inverse_simpson_diversity", "Inverse Simpson diversity"),
        ],
    ),
    (
        "Repetition",
        "repetition",
        [
            ("repeated_token_rate", "Repeated-token rate"),
            ("content_word_repetition_rate", "Content-word repetition rate"),
            ("repeated_3gram_rate", "Repeated 3-gram rate"),
            ("repeated_4gram_rate", "Repeated 4-gram rate"),
            ("consecutive_repetition_rate", "Consecutive repetition rate"),
        ],
    ),
    (
        "Lexical Density",
        "lexical_density",
        [
            ("lexical_density_proxy", "Lexical density proxy"),
            ("function_to_content_ratio_proxy", "Function-to-content ratio proxy"),
            ("content_to_function_ratio_proxy", "Content-to-function ratio proxy"),
            ("function_word_proportion_proxy", "Function-word proportion proxy"),
        ],
    ),
    (
        "Lexical Frequency Profile",
        "lexical_frequency_profile",
        [
            ("top_10_percent", "Top 10% frequency proportion"),
            ("top_50_percent", "Top 50% frequency proportion"),
            ("bottom_50_percent", "Bottom 50% frequency proportion"),
            ("hapax", "Hapax proportion"),
        ],
    ),
    (
        "Repeated-Source Translation Choice Proxy",
        "repeated_source_translation_choice",
        [
            ("ptf", "PTF"),
            ("cdu", "CDU"),
            ("groups_analyzed", "Repeated-source groups analyzed"),
        ],
    ),
    (
        "Optional Stanza Metrics",
        "stanza_metrics",
        [
            ("lexical_density", "Lexical density"),
            ("function_to_content_ratio", "Function-to-content ratio"),
            ("content_to_function_ratio", "Content-to-function ratio"),
            ("function_word_proportion", "Function-word proportion"),
            ("upos_deprel_ttr", "UPOS:DEPREL TTR"),
            ("unique_word_forms", "Unique word forms"),
            ("unique_lemmas", "Unique lemmas"),
            ("word_form_shannon_diversity", "Word-form Shannon diversity"),
            ("word_form_simpson_diversity", "Word-form Simpson diversity"),
            ("word_form_inverse_simpson_diversity", "Word-form inverse Simpson diversity"),
            ("lemma_shannon_diversity", "Lemma Shannon diversity"),
            ("lemma_simpson_diversity", "Lemma Simpson diversity"),
            ("lemma_inverse_simpson_diversity", "Lemma inverse Simpson diversity"),
            ("morph_feature_inventory_size", "Morphological feature inventory size"),
            ("morph_feature_shannon_diversity", "Morph-feature Shannon diversity"),
            ("morph_feature_simpson_diversity", "Morph-feature Simpson diversity"),
            ("morph_feature_inverse_simpson_diversity", "Morph-feature inverse Simpson diversity"),
        ],
    ),
    (
        "Optional Synonym Frequency Analysis (SFA)",
        "synonym_frequency_analysis",
        [
            ("sfa_syn_ttr", "SynTTR"),
            ("ptf", "PTF"),
            ("cdu", "CDU"),
            ("source_lemmas_considered", "Source lemmas considered"),
            ("source_lemmas_with_matches", "Source lemmas with matches"),
            ("sentences_with_candidates", "Sentences with candidates"),
            ("dictionary_entries", "Dictionary entries"),
            ("dictionary_resource", "Dictionary resource"),
            ("dictionary_path", "Dictionary path"),
            ("dictionary_inverted", "Dictionary inverted"),
        ],
    ),
    (
        "Optional ASTrED Metrics",
        "astred",
        [
            ("ted", "ASTrED TED"),
            ("word_cross", "Word crossings"),
            ("seq_cross", "Sequence crossings"),
            ("sacr_cross", "SACR crossings"),
            ("aligned_word_pairs", "Aligned word pairs"),
            ("aligned_source_ratio", "Aligned source ratio"),
            ("aligned_target_ratio", "Aligned target ratio"),
            ("rows_total", "Rows total"),
            ("rows_eligible", "Rows eligible"),
            ("rows_excluded_sentence_count_mismatch", "Rows excluded: sentence-count mismatch"),
            ("rows_scored", "Rows scored"),
            ("rows_scored_multisentence", "Multi-sentence rows scored"),
            ("sentence_pairs_scored", "Sentence pairs scored"),
            ("rows_failed", "Rows failed"),
            ("parser", "Parser"),
            ("source_column", "Source column"),
            ("target_column", "Target column"),
            ("alignment_mode", "Alignment mode"),
            ("top_error", "Top error"),
        ],
    ),    (
        "Bilingual ASTrED: Source to Prediction",
        "astred_source_prediction",
        [
            ("ted", "ASTrED TED"),
            ("word_cross", "Word crossings"),
            ("seq_cross", "Sequence crossings"),
            ("sacr_cross", "SACR crossings"),
            ("aligned_word_pairs", "Aligned word pairs"),
            ("aligned_source_ratio", "Aligned source ratio"),
            ("aligned_target_ratio", "Aligned target ratio"),
            ("rows_total", "Rows total"),
            ("rows_eligible", "Rows eligible"),
            ("rows_excluded_sentence_count_mismatch", "Rows excluded: sentence-count mismatch"),
            ("rows_scored", "Rows scored"),
            ("rows_scored_multisentence", "Multi-sentence rows scored"),
            ("sentence_pairs_scored", "Sentence pairs scored"),
            ("rows_failed", "Rows failed"),
            ("parser", "Parser"),
            ("source_column", "Source column"),
            ("target_column", "Target column"),
            ("alignment_mode", "Alignment mode"),
            ("top_error", "Top error"),
        ],
    ),
    (
        "Bilingual ASTrED: Source to Reference",
        "astred_source_reference",
        [
            ("ted", "ASTrED TED"),
            ("word_cross", "Word crossings"),
            ("seq_cross", "Sequence crossings"),
            ("sacr_cross", "SACR crossings"),
            ("aligned_word_pairs", "Aligned word pairs"),
            ("aligned_source_ratio", "Aligned source ratio"),
            ("aligned_target_ratio", "Aligned target ratio"),
            ("rows_total", "Rows total"),
            ("rows_eligible", "Rows eligible"),
            ("rows_excluded_sentence_count_mismatch", "Rows excluded: sentence-count mismatch"),
            ("rows_scored", "Rows scored"),
            ("rows_scored_multisentence", "Multi-sentence rows scored"),
            ("sentence_pairs_scored", "Sentence pairs scored"),
            ("rows_failed", "Rows failed"),
            ("parser", "Parser"),
            ("source_column", "Source column"),
            ("target_column", "Target column"),
            ("alignment_mode", "Alignment mode"),
            ("top_error", "Top error"),
        ],
    ),
]

_OPTIONAL_ADEQUACY_KEYS = ("comet", "xcomet", "cometkiwi", "metricx")
_OPTIONAL_SECTION_KEYS = {
    "stanza_metrics": "stanza",
    "synonym_frequency_analysis": "sfa",
    "astred": "astred",
    "astred_source_prediction": "astred_source_prediction",
    "astred_source_reference": "astred_source_reference",
}


_PROMPT_COLUMNS = ("P0", "P1", "P2", "P3", "P4", "P5")
_COMPARISON_EXCLUDED_SECTIONS = {"analysis_metadata", "report_context"}


def _comparison_section_label(section: str) -> str:
    for title, section_key, _metrics in _REPORT_SECTIONS:
        if section_key == section:
            return title
    return "Run metadata" if section == "meta" else section.replace("_", " ").title()


def _comparison_metric_label(section: str, metric: str) -> str:
    for _title, section_key, metrics in _REPORT_SECTIONS:
        if section_key == section:
            for metric_key, label in metrics:
                if metric_key == metric:
                    return label
    return metric.replace("_", " ")


def _comparison_interpretation(section: str, metric: str) -> str:
    base_section = section.removeprefix("reference_").removeprefix("prediction_")
    if section == "adequacy":
        return "lower is better" if metric == "ter" else "higher is better"
    if base_section == "lexical_diversity":
        if metric == "yules_k":
            return "lower means more diversity"
        if metric in {"ttr", "mattr_50", "mtld", "yules_i"}:
            return "higher means more diversity"
        return "corpus size or reliability context"
    if base_section == "lexical_frequency_profile":
        return (
            "higher means rarer vocabulary"
            if metric.startswith(("b2_", "b3_", "beyond_"))
            else "higher means more frequent vocabulary"
        )
    if base_section == "morphological_diversity":
        if metric == "morph_simpson_d":
            return "lower means more diversity"
        if metric.startswith("morph_") and "single" not in metric:
            return "higher means more diversity"
        return "lemma-count context"
    if base_section == "synonym_frequency_analysis":
        if metric in {"sfa_ptf", "sfa_cdu", "ptf", "cdu"}:
            return "lower means more diverse choice"
        if metric == "sfa_syn_ttr":
            return "higher means more diverse choice"
        return "coverage count"
    if section == "repetition":
        return "lower means less repetition"
    if section in {"astred", "astred_source_prediction"}:
        if metric in {"ted", "word_cross", "seq_cross", "sacr_cross"}:
            return (
                "lower means closer to source structure"
                if section == "astred_source_prediction"
                else "lower means closer to reference structure"
            )
        if metric in {"aligned_source_ratio", "aligned_target_ratio"}:
            return "higher alignment coverage"
        return "coverage count"
    return "descriptive; compare across prompts"


def _prompt_column(experiment_key: Any) -> str | None:
    if not isinstance(experiment_key, str):
        return None
    prefix = experiment_key.split("_", maxsplit=1)[0].upper()
    return prefix if prefix in _PROMPT_COLUMNS else None


def _summary_identity(summary_path: Path, summary: dict[str, Any]) -> tuple[str, str, str] | None:
    metadata = summary.get("analysis_metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    input_metadata = metadata.get("input", {})
    if not isinstance(input_metadata, dict):
        input_metadata = {}
    generation_metadata = metadata.get("generation_run_metadata", {})
    if not isinstance(generation_metadata, dict):
        generation_metadata = {}

    experiment_keys = input_metadata.get("experiment_keys", [])
    experiment_key = (
        experiment_keys[0]
        if isinstance(experiment_keys, list) and experiment_keys
        else generation_metadata.get("experiment_key")
    )
    prompt_column = _prompt_column(experiment_key)
    if prompt_column is None:
        return None

    dataset_key = generation_metadata.get("dataset_key") or summary_path.parent.parent.name
    model_key = (
        input_metadata.get("model_key")
        or generation_metadata.get("model_key")
        or summary_path.parent.name
    )
    return str(dataset_key), str(model_key), prompt_column


def _numeric_summary_metrics(summary: dict[str, Any]) -> dict[tuple[str, str], int | float]:
    metrics: dict[tuple[str, str], int | float] = {}
    for section, values in summary.items():
        if section in _COMPARISON_EXCLUDED_SECTIONS:
            continue
        if (
            section == "row_count"
            and isinstance(values, (int, float))
            and not isinstance(values, bool)
        ):
            metrics[("meta", "rows_analyzed")] = values
            continue
        if not isinstance(values, dict):
            continue
        for metric, value in values.items():
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and not pd.isna(value)
            ):
                metrics[(section, metric)] = value
    return metrics


def _infer_comparison_root(output_dir: Path) -> Path | None:
    for directory in (output_dir, *output_dir.parents):
        if directory.name in {"baseline", "experiments"}:
            return directory.parent
    return None


def write_prompt_comparisons(results_root: str | Path) -> dict[str, str]:
    """Write one P0-P5 comparison TSV for each dataset/model under a result root."""
    root = Path(results_root)
    grouped_summaries: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for summary_path in sorted(root.rglob("analysis_summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable analysis summary %s: %s", summary_path, exc)
            continue
        identity = _summary_identity(summary_path, summary)
        if identity is None:
            logger.warning(
                "Skipping comparison entry without a P0-P5 experiment key: %s", summary_path
            )
            continue
        dataset_key, model_key, prompt_column = identity
        grouped_summaries.setdefault((dataset_key, model_key), {})[prompt_column] = summary

    written: dict[str, str] = {}
    for (dataset_key, model_key), summaries_by_prompt in grouped_summaries.items():
        metrics_by_prompt = {
            prompt: _numeric_summary_metrics(summary)
            for prompt, summary in summaries_by_prompt.items()
        }
        metric_keys = set().union(*(metrics.keys() for metrics in metrics_by_prompt.values()))
        rows: list[dict[str, Any]] = []
        for section, metric in sorted(
            metric_keys,
            key=lambda item: (
                _comparison_section_label(item[0]),
                _comparison_metric_label(item[0], item[1]),
            ),
        ):
            row: dict[str, Any] = {
                "metric_group": _comparison_section_label(section),
                "metric": _comparison_metric_label(section, metric),
                "metric_key": f"{section}.{metric}",
                "interpretation": _comparison_interpretation(section, metric),
            }
            for prompt in _PROMPT_COLUMNS:
                row[prompt] = metrics_by_prompt.get(prompt, {}).get((section, metric))
            rows.append(row)

        comparison_path = root / "summary" / dataset_key / model_key / "prompt_comparison.tsv"
        save_dataframe_tsv(pd.DataFrame(rows), comparison_path)
        written[f"{dataset_key}/{model_key}"] = str(comparison_path)
        logger.info("Prompt comparison TSV written to %s", comparison_path)
    return written


def _format_metric_value(value: Any) -> str:
    return str(value)


def _optional_request_context(summary: dict[str, Any]) -> dict[str, Any]:
    return summary.get("report_context", {}).get("optional_requests", {})


def _section_has_visible_metrics(section_key: str, values: dict[str, Any]) -> bool:
    if section_key == "stanza_metrics":
        return any(value is not None for value in values.values())
    if section_key == "synonym_frequency_analysis":
        return (
            any(
                values.get(metric) not in (None, 0)
                for metric in (
                    "sfa_syn_ttr",
                    "ptf",
                    "cdu",
                    "source_lemmas_considered",
                    "source_lemmas_with_matches",
                    "sentences_with_candidates",
                    "dictionary_entries",
                )
            )
            or values.get("dictionary_path") is not None
        )
    if section_key in {"astred", "astred_source_prediction"}:
        return (
            any(
                values.get(metric) not in (None, 0)
                for metric in (
                    "ted",
                    "word_cross",
                    "seq_cross",
                    "sacr_cross",
                    "aligned_word_pairs",
                    "aligned_source_ratio",
                    "aligned_target_ratio",
                    "rows_scored",
                    "rows_failed",
                )
            )
            or values.get("parser") is not None
        )
    return True


def _render_section(
    title: str,
    section_key: str,
    summary: dict[str, Any],
) -> list[str]:
    values = summary.get(section_key, {})
    if not isinstance(values, dict):
        return []

    metric_specs = next(
        (
            metrics
            for report_title, key, metrics in _REPORT_SECTIONS
            if report_title == title and key == section_key
        ),
        [],
    )
    lines: list[str] = []

    for metric_key, label in metric_specs:
        if metric_key in values and values[metric_key] is not None:
            lines.append(f"- {label}: {_format_metric_value(values[metric_key])}")
        elif section_key == "repeated_source_translation_choice" and metric_key in {"ptf", "cdu"}:
            if values.get("groups_analyzed", 0) == 0:
                lines.append(f"- {label}: not defined (no repeated source groups)")

    if section_key in _OPTIONAL_SECTION_KEYS and not _section_has_visible_metrics(
        section_key, values
    ):
        return []

    if not lines:
        return []

    return [f"## {title}", "", *lines, ""]


def _build_report(summary: dict[str, Any]) -> str:
    lines = ["# Baseline Translationese Analysis", "", f"Rows analyzed: {summary['row_count']}", ""]

    metadata = summary.get("analysis_metadata", {})
    if isinstance(metadata, dict) and metadata:
        input_meta = metadata.get("input", {}) if isinstance(metadata.get("input"), dict) else {}
        trace_lines = [
            "## Traceability",
            "",
            f"- Predictions: {input_meta.get('predictions_path')}",
            f"- Output directory: {metadata.get('output_dir')}",
            f"- Analysis started: {metadata.get('analysis_started_at_utc')}",
            f"- Analysis finished: {metadata.get('analysis_finished_at_utc')}",
            f"- Analysis runtime seconds: {metadata.get('analysis_runtime_seconds')}",
            f"- Model key: {input_meta.get('model_key')}",
            f"- Model id: {input_meta.get('model_id')}",
            f"- Experiment keys: {input_meta.get('experiment_keys')}",
            f"- Prompt variants: {input_meta.get('prompt_variants') or input_meta.get('initial_prompt_variants') or input_meta.get('refinement_prompt_variants')}",
            f"- Prompt columns: {input_meta.get('prompt_columns')}",
            "",
        ]
        lines.extend(trace_lines)

    for title, section_key, _metric_specs in _REPORT_SECTIONS:
        lines.extend(_render_section(title, section_key, summary))

    skipped_optional: list[str] = []
    optional_requests = _optional_request_context(summary)
    adequacy = summary.get("adequacy", {})
    for metric_key in _OPTIONAL_ADEQUACY_KEYS:
        if adequacy.get(metric_key) is not None:
            continue
        label = next(
            label
            for report_title, report_section_key, metric_specs in _REPORT_SECTIONS
            if report_title == "Adequacy" and report_section_key == "adequacy"
            for current_key, label in metric_specs
            if current_key == metric_key
        )
        status = (
            "requested but unavailable"
            if optional_requests.get(metric_key)
            else "not requested for this run"
        )
        skipped_optional.append(f"- {label}: {status}")

    for section_key, request_key in _OPTIONAL_SECTION_KEYS.items():
        section_values = summary.get(section_key, {})
        if not isinstance(section_values, dict):
            continue
        if _section_has_visible_metrics(section_key, section_values):
            continue
        title = next(
            report_title
            for report_title, current_section_key, _metric_specs in _REPORT_SECTIONS
            if current_section_key == section_key
        )
        status = (
            "requested but unavailable"
            if optional_requests.get(request_key)
            else "not requested for this run"
        )
        skipped_optional.append(f"- {title}: {status}")

    if skipped_optional:
        lines.extend(["## Skipped Optional Metrics", "", *skipped_optional, ""])

    lines.extend(["## Notes", "", f"- {summary['implemented_metrics_note']}"])
    return "\n".join(lines) + "\n"


def _empty_morphological_diversity() -> dict[str, float | int | None]:
    return {
        "morph_shannon_entropy": None,
        "morph_simpson_d": None,
        "morph_inverse_simpson_optional": None,
        "morph_lemmas_used_for_entropy": 0,
        "morph_single_wordform_lemmas": 0,
        "morph_multi_wordform_lemmas": 0,
    }


_MORPHOLOGICAL_DIVERSITY_KEYS = frozenset(
    {
        "morph_shannon_entropy",
        "morph_simpson_d",
        "morph_inverse_simpson_optional",
        "morph_lemmas_used_for_entropy",
        "morph_single_wordform_lemmas",
        "morph_multi_wordform_lemmas",
    }
)


def _split_lexical_metrics(metrics: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    result = dict(metrics)
    return result, result.pop("lexical_frequency_profile")


def analyze_predictions(
    predictions_df: pd.DataFrame,
    comet_model: str | None = None,
    xcomet_model: str | None = None,
    comet_qe_model: str | None = None,
    cometkiwi_model: str | None = None,
    metricx_model: str | None = None,
    background_texts: list[str] | None = None,
    comet_batch_size: int = 8,
    comet_gpus: int | None = None,
    stanza_lang: str | None = None,
    sfa_target_lang: str | None = None,
    sfa_source_lang: str = "en",
    sfa_dictionary_path: str | Path | None = None,
    sfa_dictionary_invert: bool | None = None,
    astred_model_or_lang: str | None = None,
    astred_parser: str = "stanza",
    astred_source_column: str = "prediction",
    astred_target_column: str = "reference",
    astred_alignment_column: str | None = None,
    astred_source_prediction: bool = False,
    astred_source_prediction_source_model_or_lang: str = "en",
    astred_source_prediction_target_model_or_lang: str | None = None,
    astred_source_prediction_source_column: str = "source",
    astred_source_prediction_target_column: str = "prediction",
    astred_source_prediction_alignment_column: str | None = None,
    astred_source_reference: bool = False,
) -> dict[str, Any]:
    """Compute adequacy plus matched target-side naturalness metrics."""
    if "reference" not in predictions_df:
        raise ValueError("Naturalness analysis requires a reference target column.")

    reference_texts = predictions_df["reference"].fillna("").astype(str).tolist()
    prediction_texts = predictions_df["prediction"].fillna("").astype(str).tolist()
    frequency_reference_texts = (
        background_texts if background_texts is not None else reference_texts
    )
    frequency_reference_source = (
        "configured_natural_target_corpus"
        if background_texts is not None
        else "evaluation_references"
    )

    per_example = compute_per_example_metrics(
        predictions_df, background_texts=frequency_reference_texts
    )
    summary = compute_automatic_metrics(
        predictions_df,
        comet_model=comet_model,
        xcomet_model=xcomet_model,
        comet_qe_model=comet_qe_model,
        cometkiwi_model=cometkiwi_model,
        metricx_model=metricx_model,
        comet_batch_size=comet_batch_size,
        comet_gpus=comet_gpus,
    )
    if "sentence_boundary_contract_valid" in predictions_df:
        contract_values = predictions_df["sentence_boundary_contract_valid"].fillna(False).astype(bool)
        valid_outputs = int(contract_values.sum())
        expected_counts = predictions_df.get("sentence_boundary_count")
        expected_count = None
        if expected_counts is not None:
            non_null_counts = expected_counts.dropna().astype(int).unique().tolist()
            if len(non_null_counts) == 1:
                expected_count = int(non_null_counts[0])
        summary["sentence_boundary_contract"] = {
            "expected_count": expected_count,
            "rows_total": int(len(predictions_df)),
            "valid_outputs": valid_outputs,
            "invalid_outputs": int(len(predictions_df) - valid_outputs),
            "valid_ratio": valid_outputs / len(predictions_df) if len(predictions_df) else None,
        }
    reference_lexical, reference_lfp = _split_lexical_metrics(
        compute_target_lexical_metrics(reference_texts, frequency_reference_texts)
    )
    prediction_lexical, prediction_lfp = _split_lexical_metrics(
        compute_target_lexical_metrics(prediction_texts, frequency_reference_texts)
    )
    summary.update(
        {
            "frequency_profile_reference": {
                "source": frequency_reference_source,
                "text_count": len(frequency_reference_texts),
                "tokenization": "lowercase surface forms; punctuation removed",
            },
            "reference_lexical_diversity": reference_lexical,
            "prediction_lexical_diversity": prediction_lexical,
            "reference_lexical_frequency_profile": reference_lfp,
            "prediction_lexical_frequency_profile": prediction_lfp,
            "reference_morphological_diversity": _empty_morphological_diversity(),
            "prediction_morphological_diversity": _empty_morphological_diversity(),
            "reference_synonym_frequency_analysis": empty_sfa_summary(),
            "prediction_synonym_frequency_analysis": empty_sfa_summary(),
            "astred": empty_astred_summary(),
            "astred_source_prediction": empty_astred_summary(),
            "astred_source_reference": empty_astred_summary(),
            "report_context": {
                "optional_requests": {
                    "comet": comet_model is not None,
                    "xcomet": xcomet_model is not None,
                    "cometkiwi": (cometkiwi_model or comet_qe_model) is not None,
                    "metricx": metricx_model is not None,
                    "stanza": stanza_lang is not None,
                    "sfa": sfa_target_lang is not None,
                    "astred": astred_model_or_lang is not None,
                    "astred_source_reference": astred_source_reference,
                }
            },
        }
    )

    target_annotations: dict[str, list[list[dict[str, Any]]]] = {}
    if stanza_lang:
        try:
            target_annotations = {
                "reference": annotate_texts(reference_texts, lang=stanza_lang),
                "prediction": annotate_texts(prediction_texts, lang=stanza_lang),
            }
            summary["reference_morphological_diversity"] = {
                key: value
                for key, value in compute_stanza_summary_metrics(
                    target_annotations["reference"]
                ).items()
                if key in _MORPHOLOGICAL_DIVERSITY_KEYS
            }
            summary["prediction_morphological_diversity"] = {
                key: value
                for key, value in compute_stanza_summary_metrics(
                    target_annotations["prediction"]
                ).items()
                if key in _MORPHOLOGICAL_DIVERSITY_KEYS
            }
        except StanzaAnalysisError as exc:
            logger.warning("Skipping Stanza morphology: %s", exc)

    if sfa_target_lang:
        try:
            dictionary_spec = (
                {
                    "path": Path(sfa_dictionary_path),
                    "invert": bool(sfa_dictionary_invert),
                    "resource": "custom",
                }
                if sfa_dictionary_path is not None
                else resolve_default_apertium_dictionary(sfa_target_lang)
            )
            if sfa_dictionary_invert is not None:
                dictionary_spec["invert"] = bool(sfa_dictionary_invert)
            translation_options = load_apertium_translation_options(
                dictionary_spec["path"], invert=bool(dictionary_spec["invert"])
            )
            source_annotations = annotate_texts(
                predictions_df["source"].fillna("").astype(str).tolist(), lang=sfa_source_lang
            )
            if not target_annotations or stanza_lang != sfa_target_lang:
                target_annotations = {
                    "reference": annotate_texts(reference_texts, lang=sfa_target_lang),
                    "prediction": annotate_texts(prediction_texts, lang=sfa_target_lang),
                }
            for role in ("reference", "prediction"):
                counts = collect_sfa_option_counts(
                    source_annotations, target_annotations[role], translation_options
                )
                sfa_summary = summarize_sfa_metrics(
                    counts["counts_by_source_lemma"],
                    sentences_with_candidates=int(counts["sentences_with_candidates"]),
                    dictionary_path=dictionary_spec["path"],
                    dictionary_resource=dictionary_spec.get("resource"),
                    dictionary_inverted=bool(dictionary_spec["invert"]),
                )
                sfa_summary["dictionary_entries"] = len(translation_options)
                summary[f"{role}_synonym_frequency_analysis"] = sfa_summary
        except (KeyError, SFAAnalysisError, StanzaAnalysisError, ValueError) as exc:
            logger.warning("Skipping SFA metrics: %s", exc)

    if astred_model_or_lang:
        try:
            astred_result = compute_astred_metrics(
                predictions_df,
                source_model_or_lang=astred_model_or_lang,
                parser=astred_parser,
                source_column=astred_source_column,
                target_column=astred_target_column,
                alignment_column=astred_alignment_column,
            )
            per_example = per_example.join(astred_result["per_example"])
            summary["astred"] = astred_result["summary"]
        except (ASTrEDAnalysisError, ValueError) as exc:
            logger.warning("Skipping ASTrED metrics: %s", exc)

    if astred_source_prediction:
        target_model_or_lang = (
            astred_source_prediction_target_model_or_lang or astred_model_or_lang
        )
        if not target_model_or_lang:
            raise ValueError(
                "Source-to-prediction ASTrED requires "
                "--astred-source-prediction-target-model-or-lang."
            )
        try:
            source_prediction_result = compute_astred_metrics(
                predictions_df,
                source_model_or_lang=astred_source_prediction_source_model_or_lang,
                target_model_or_lang=target_model_or_lang,
                parser=astred_parser,
                source_column=astred_source_prediction_source_column,
                target_column=astred_source_prediction_target_column,
                alignment_column=astred_source_prediction_alignment_column,
            )
            per_example = per_example.join(
                _rename_astred_columns(
                    source_prediction_result["per_example"],
                    prefix="astred_source_prediction_",
                )
            )
            summary["astred_source_prediction"] = source_prediction_result["summary"]
        except (ASTrEDAnalysisError, ValueError) as exc:
            logger.warning("Skipping source-to-prediction ASTrED metrics: %s", exc)

    if astred_source_reference:
        target_model_or_lang = (
            astred_source_prediction_target_model_or_lang or astred_model_or_lang
        )
        if not target_model_or_lang:
            raise ValueError(
                "Source-to-reference ASTrED requires "
                "--astred-source-prediction-target-model-or-lang."
            )
        try:
            source_reference_result = compute_astred_metrics(
                predictions_df,
                source_model_or_lang=astred_source_prediction_source_model_or_lang,
                target_model_or_lang=target_model_or_lang,
                parser=astred_parser,
                source_column=astred_source_prediction_source_column,
                target_column="reference",
                alignment_column=None,
            )
            per_example = per_example.join(
                _rename_astred_columns(
                    source_reference_result["per_example"],
                    prefix="astred_source_reference_",
                )
            )
            summary["astred_source_reference"] = source_reference_result["summary"]
        except (ASTrEDAnalysisError, ValueError) as exc:
            logger.warning("Skipping source-to-reference ASTrED metrics: %s", exc)

    summary["implemented_metrics_note"] = (
        "Adequacy is reported separately (BLEU, TER, chrF++, optional COMET-family metrics, and MetricX-24). "
        "Target-side reference and prediction sections report corpus-level TTR, Yule's K/I, MTLD, "
        "fixed B1/B2/B3 lexical-frequency bands, lemma-to-wordform Shannon/Simpson morphology, "
        "and dictionary-grounded SFA when Stanza and an Apertium dictionary are available."
    )
    return {
        "summary": summary,
        "summary_table": _flatten_summary(summary),
        "per_example": per_example,
        "report_markdown": _build_report(summary),
    }


def persist_analysis_outputs(
    analysis_result: dict[str, Any],
    output_dir: str | Path,
    force: bool = False,
) -> dict[str, str]:
    """Persist baseline analysis outputs next to the prediction run."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "summary_json": output_dir / "analysis_summary.json",
        "summary_csv": output_dir / "analysis_summary.csv",
        "per_example_jsonl": output_dir / "per_example_metrics.jsonl",
        "per_example_csv": output_dir / "per_example_metrics.csv",
        "report_md": output_dir / "analysis_report.md",
        "analysis_metadata_json": output_dir / "analysis_metadata.json",
    }
    if not force:
        existing = [path for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing analysis outputs without --force: "
                + ", ".join(str(path) for path in existing)
            )

    save_json(analysis_result["summary"], paths["summary_json"])
    save_dataframe_csv(analysis_result["summary_table"], paths["summary_csv"])
    save_dataframe_jsonl(analysis_result["per_example"], paths["per_example_jsonl"])
    save_dataframe_csv(analysis_result["per_example"], paths["per_example_csv"])
    paths["report_md"].write_text(analysis_result["report_markdown"], encoding="utf-8")
    save_json(analysis_result.get("analysis_metadata", {}), paths["analysis_metadata_json"])
    return {key: str(value) for key, value in paths.items()}


def refresh_astred_outputs(
    predictions_df: pd.DataFrame,
    *,
    predictions_path: Path,
    output_dir: Path,
    astred_model_or_lang: str,
    astred_parser: str,
    astred_source_column: str,
    astred_target_column: str,
    astred_alignment_column: str | None,
    started_at_utc: str,
    finished_at_utc: str,
    runtime_seconds: float,
    astred_refresh_reference: bool = True,
    astred_source_prediction: bool = False,
    astred_source_prediction_source_model_or_lang: str = "en",
    astred_source_prediction_target_model_or_lang: str | None = None,
    astred_source_prediction_source_column: str = "source",
    astred_source_prediction_target_column: str = "prediction",
    astred_source_prediction_alignment_column: str | None = None,
    astred_source_reference: bool = False,
) -> dict[str, str]:
    """Refresh only ASTrED outputs while preserving every existing metric."""
    summary_path = output_dir / "analysis_summary.json"
    per_example_path = output_dir / "per_example_metrics.jsonl"
    metadata_path = output_dir / "analysis_metadata.json"
    required_paths = (summary_path, per_example_path, metadata_path)
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "ASTrED-only refresh requires existing analysis outputs: "
            + ", ".join(str(path) for path in missing_paths)
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    per_example = pd.read_json(per_example_path, lines=True)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if len(per_example) != len(predictions_df):
        raise ValueError(
            "Existing per-example metrics do not match predictions row count: "
            f"{len(per_example)} metrics rows vs {len(predictions_df)} predictions rows."
        )
    if "sentence_id" in predictions_df and "sentence_id" in per_example:
        if (
            predictions_df["sentence_id"].astype(str).tolist()
            != per_example["sentence_id"].astype(str).tolist()
        ):
            raise ValueError(
                "Existing per-example metrics do not match prediction sentence_id order."
            )

    if astred_refresh_reference:
        astred_result = compute_astred_metrics(
            predictions_df,
            source_model_or_lang=astred_model_or_lang,
            parser=astred_parser,
            source_column=astred_source_column,
            target_column=astred_target_column,
            alignment_column=astred_alignment_column,
        )
        for column in _ASTRED_COLUMNS:
            per_example[column] = astred_result["per_example"][column].to_numpy()
        summary["astred"] = astred_result["summary"]
    if astred_source_prediction:
        target_model_or_lang = (
            astred_source_prediction_target_model_or_lang or astred_model_or_lang
        )
        source_prediction_result = compute_astred_metrics(
            predictions_df,
            source_model_or_lang=astred_source_prediction_source_model_or_lang,
            target_model_or_lang=target_model_or_lang,
            parser=astred_parser,
            source_column=astred_source_prediction_source_column,
            target_column=astred_source_prediction_target_column,
            alignment_column=astred_source_prediction_alignment_column,
        )
        renamed = _rename_astred_columns(
            source_prediction_result["per_example"],
            prefix="astred_source_prediction_",
        )
        for column in _ASTRED_SOURCE_PREDICTION_COLUMNS:
            per_example[column] = renamed[column].to_numpy()
        summary["astred_source_prediction"] = source_prediction_result["summary"]
    if astred_source_reference:
        target_model_or_lang = (
            astred_source_prediction_target_model_or_lang or astred_model_or_lang
        )
        source_reference_result = compute_astred_metrics(
            predictions_df,
            source_model_or_lang=astred_source_prediction_source_model_or_lang,
            target_model_or_lang=target_model_or_lang,
            parser=astred_parser,
            source_column=astred_source_prediction_source_column,
            target_column="reference",
            alignment_column=None,
        )
        renamed = _rename_astred_columns(
            source_reference_result["per_example"],
            prefix="astred_source_reference_",
        )
        for column in _ASTRED_SOURCE_REFERENCE_COLUMNS:
            per_example[column] = renamed[column].to_numpy()
        summary["astred_source_reference"] = source_reference_result["summary"]
    report_context = summary.setdefault("report_context", {})
    optional_requests = report_context.setdefault("optional_requests", {})
    optional_requests["astred"] = True
    optional_requests["astred_source_prediction"] = astred_source_prediction
    optional_requests["astred_source_reference"] = astred_source_reference
    metadata["astred_refresh"] = {
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "runtime_seconds": runtime_seconds,
        "predictions_path": str(predictions_path),
        "astred_model_or_lang": astred_model_or_lang,
        "astred_parser": astred_parser,
        "astred_source_column": astred_source_column,
        "astred_target_column": astred_target_column,
        "astred_alignment_column": astred_alignment_column,
        "astred_refresh_reference": astred_refresh_reference,
        "astred_source_prediction": astred_source_prediction,
        "astred_source_prediction_source_model_or_lang": (
            astred_source_prediction_source_model_or_lang
        ),
        "astred_source_prediction_target_model_or_lang": (
            astred_source_prediction_target_model_or_lang
        ),
        "astred_source_prediction_source_column": astred_source_prediction_source_column,
        "astred_source_prediction_target_column": astred_source_prediction_target_column,
        "astred_source_prediction_alignment_column": astred_source_prediction_alignment_column,
        "astred_source_reference": astred_source_reference,
        "multi_sentence_policy": "pair_by_position_when_sentence_counts_match",
    }
    summary["analysis_metadata"] = metadata

    save_json(summary, summary_path)
    save_dataframe_csv(_flatten_summary(summary), output_dir / "analysis_summary.csv")
    save_dataframe_jsonl(per_example, per_example_path)
    save_dataframe_csv(per_example, output_dir / "per_example_metrics.csv")
    (output_dir / "analysis_report.md").write_text(_build_report(summary), encoding="utf-8")
    save_json(metadata, metadata_path)
    return {
        "summary_json": str(summary_path),
        "summary_csv": str(output_dir / "analysis_summary.csv"),
        "per_example_jsonl": str(per_example_path),
        "per_example_csv": str(output_dir / "per_example_metrics.csv"),
        "report_md": str(output_dir / "analysis_report.md"),
        "analysis_metadata_json": str(metadata_path),
    }


_LEXICAL_DIVERSITY_SUMMARY_KEYS = frozenset(
    {
        "token_count",
        "type_count",
        "ttr",
        "yules_k",
        "yules_i",
        "mtld",
        "mattr_50",
        "mtld_status",
    }
)


def refresh_corpus_lexical_diversity_outputs(
    predictions_df: pd.DataFrame,
    *,
    predictions_path: Path,
    output_dir: Path,
) -> dict[str, str]:
    """Refresh corpus MTLD/MATTR while preserving all non-lexical metrics."""
    summary_path = output_dir / "analysis_summary.json"
    per_example_path = output_dir / "per_example_metrics.jsonl"
    metadata_path = output_dir / "analysis_metadata.json"
    required_paths = (summary_path, per_example_path, metadata_path)
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Lexical-diversity-only refresh requires existing analysis outputs: "
            + ", ".join(str(path) for path in missing_paths)
        )
    required_columns = {"prediction", "reference"}
    missing_columns = sorted(required_columns - set(predictions_df.columns))
    if missing_columns:
        raise ValueError(
            "Lexical-diversity-only refresh requires prediction columns: "
            + ", ".join(missing_columns)
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    per_example = pd.read_json(per_example_path, lines=True)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if len(per_example) != len(predictions_df):
        raise ValueError(
            "Existing per-example metrics do not match predictions row count: "
            f"{len(per_example)} metrics rows vs {len(predictions_df)} predictions rows."
        )
    if "sentence_id" in predictions_df and "sentence_id" in per_example:
        if (
            predictions_df["sentence_id"].astype(str).tolist()
            != per_example["sentence_id"].astype(str).tolist()
        ):
            raise ValueError(
                "Existing per-example metrics do not match prediction sentence_id order."
            )

    for side, column in (("reference", "reference"), ("prediction", "prediction")):
        texts = predictions_df[column].fillna("").astype(str).tolist()
        corpus_metrics = compute_target_lexical_metrics(texts, [])
        lexical_summary = summary.setdefault(f"{side}_lexical_diversity", {})
        lexical_summary.update(
            {
                key: corpus_metrics[key]
                for key in _LEXICAL_DIVERSITY_SUMMARY_KEYS
            }
        )

    # MTLD and MATTR are corpus measures. Remove values previously calculated
    # for individual aligned sentences so they cannot be interpreted as such.
    per_example = per_example.drop(
        columns=["prediction_mattr", "prediction_mtld"], errors="ignore"
    )
    metadata["lexical_diversity_refresh"] = {
        "predictions_path": str(predictions_path),
        "row_count": int(len(predictions_df)),
        "tokenization": "existing tokenize_text: lowercase surface forms; punctuation removed",
        "mtld": {"implementation": "lexical_diversity.lex_div.mtld", "minimum_tokens": 100},
        "mattr_50": {
            "implementation": "lexical_diversity.lex_div.mattr",
            "window_size": 50,
            "short_sequence_value": None,
        },
        "scope": "corpus-level target-side outputs; no sentence-level MTLD or MATTR",
    }
    summary["analysis_metadata"] = metadata

    save_json(summary, summary_path)
    save_dataframe_csv(_flatten_summary(summary), output_dir / "analysis_summary.csv")
    save_dataframe_jsonl(per_example, per_example_path)
    save_dataframe_csv(per_example, output_dir / "per_example_metrics.csv")
    (output_dir / "analysis_report.md").write_text(_build_report(summary), encoding="utf-8")
    save_json(metadata, metadata_path)
    return {
        "summary_json": str(summary_path),
        "summary_csv": str(output_dir / "analysis_summary.csv"),
        "per_example_jsonl": str(per_example_path),
        "per_example_csv": str(output_dir / "per_example_metrics.csv"),
        "report_md": str(output_dir / "analysis_report.md"),
        "analysis_metadata_json": str(metadata_path),
    }

def refresh_metricx_outputs(
    predictions_df: pd.DataFrame,
    *,
    predictions_path: Path,
    output_dir: Path,
    metricx_model: str,
) -> tuple[dict[str, str], float]:
    """Refresh only MetricX while preserving existing evaluation results."""
    started_at_utc = _utc_now_iso()
    start_time = time.perf_counter()
    analysis_summary_path = output_dir / "analysis_summary.json"
    automatic_metrics_path = output_dir / "automatic_metrics.json"
    summary_path = (
        analysis_summary_path if analysis_summary_path.exists() else automatic_metrics_path
    )
    if not summary_path.exists():
        raise FileNotFoundError(
            "MetricX-only refresh requires analysis_summary.json or automatic_metrics.json in "
            f"{output_dir}"
        )

    required_columns = {"source", "prediction", "reference"}
    missing_columns = sorted(required_columns - set(predictions_df.columns))
    if missing_columns:
        raise ValueError(
            "MetricX-only refresh requires prediction columns: " + ", ".join(missing_columns)
        )

    score = compute_optional_metricx(
        predictions_df["source"].fillna("").astype(str).tolist(),
        predictions_df["prediction"].fillna("").astype(str).tolist(),
        predictions_df["reference"].fillna("").astype(str).tolist(),
        metricx_model,
    )
    if score is None:
        raise RuntimeError("MetricX inference did not return a score.")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.setdefault("adequacy", {})["metricx"] = score
    written = {"summary_json": str(summary_path)}
    metadata_path = output_dir / "analysis_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    )
    metadata["metricx_refresh"] = {
        "started_at_utc": started_at_utc,
        "finished_at_utc": _utc_now_iso(),
        "runtime_seconds": time.perf_counter() - start_time,
        "predictions_path": str(predictions_path),
        "model": metricx_model,
        "row_count": int(len(predictions_df)),
        "score": score,
    }

    if summary_path == analysis_summary_path:
        summary.setdefault("report_context", {}).setdefault("optional_requests", {})["metricx"] = (
            True
        )
        summary["analysis_metadata"] = metadata
        save_dataframe_csv(_flatten_summary(summary), output_dir / "analysis_summary.csv")
        (output_dir / "analysis_report.md").write_text(_build_report(summary), encoding="utf-8")
        save_json(metadata, metadata_path)
        written.update(
            {
                "summary_csv": str(output_dir / "analysis_summary.csv"),
                "report_md": str(output_dir / "analysis_report.md"),
                "analysis_metadata_json": str(metadata_path),
            }
        )
    else:
        refresh_metadata_path = output_dir / "metricx_refresh_metadata.json"
        save_json(metadata["metricx_refresh"], refresh_metadata_path)
        written["metricx_refresh_metadata_json"] = str(refresh_metadata_path)

    save_json(summary, summary_path)
    return written, score


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze baseline translationese metrics.")
    parser.add_argument(
        "--predictions", required=True, help="Path to predictions.jsonl or predictions.parquet"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for analysis outputs; defaults to the predictions parent directory",
    )
    parser.add_argument(
        "--comparison-root",
        default=None,
        help="Result root where P0-P5 comparison TSVs are written; inferred from the output path when omitted.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing analysis outputs")
    parser.add_argument(
        "--astred-only",
        action="store_true",
        help="Refresh ASTrED fields in existing analysis outputs without recomputing other metrics.",
    )
    parser.add_argument(
        "--metricx-only",
        action="store_true",
        help="Refresh only MetricX in existing outputs without recomputing other metrics.",
    )
    parser.add_argument(
        "--lexical-diversity-only",
        action="store_true",
        help="Refresh corpus-level MTLD/MATTR and remove sentence-level MTLD/MATTR fields.",
    )
    parser.add_argument("--comet-model", default=None, help="Optional COMET model identifier")
    parser.add_argument(
        "--xcomet-model",
        default=None,
        help=f"Optional xCOMET model identifier, for example {DEFAULT_XCOMET_XXL_MODEL}",
    )
    parser.add_argument(
        "--comet-qe-model",
        default=None,
        help="Backward-compatible alias for a reference-free COMET model identifier",
    )
    parser.add_argument(
        "--cometkiwi-model",
        default=None,
        help=f"Optional COMETKiwi model identifier, for example {DEFAULT_COMETKIWI_MODEL}",
    )
    parser.add_argument(
        "--metricx-model",
        default=None,
        help=f"Optional MetricX-24 model identifier, for example {DEFAULT_METRICX_MODEL}",
    )
    parser.add_argument(
        "--comet-batch-size",
        type=int,
        default=8,
        help="Batch size used by COMET-family metrics",
    )
    parser.add_argument(
        "--comet-gpus",
        type=int,
        default=None,
        help="Number of GPUs for COMET-family metrics; defaults to 1 when CUDA is available, else 0",
    )
    parser.add_argument(
        "--background-file",
        default=None,
        help="Optional JSONL or Parquet file used as a lexical-frequency background corpus",
    )
    parser.add_argument(
        "--background-column", default="target", help="Text column to read from --background-file"
    )
    parser.add_argument(
        "--stanza-lang",
        default=None,
        help="Optional Stanza language code for lexical density, morphology, and the UPOS:DEPREL TTR proxy",
    )
    parser.add_argument(
        "--sfa-target-lang",
        default=None,
        help="Optional target-language code for dictionary-grounded Synonym Frequency Analysis",
    )
    parser.add_argument(
        "--sfa-source-lang",
        default="en",
        help="Source-language code used to annotate the English side for SFA",
    )
    parser.add_argument(
        "--sfa-dictionary-path",
        default=None,
        help="Optional path to an Apertium XML dictionary for SFA; if omitted, the default resource for --sfa-target-lang is used",
    )
    parser.add_argument(
        "--sfa-invert-dictionary",
        action="store_true",
        help="Invert the Apertium dictionary direction when building SFA translation options",
    )
    parser.add_argument(
        "--astred-model-or-lang",
        default=None,
        help="Optional ASTrED parser identifier. Use a Stanza language code (for example `ca`) or a spaCy model name.",
    )
    parser.add_argument(
        "--astred-parser",
        default="stanza",
        choices=["stanza", "spacy"],
        help="Parser backend used by ASTrED.",
    )
    parser.add_argument(
        "--astred-source-column", default="prediction", help="Source column for ASTrED comparison"
    )
    parser.add_argument(
        "--astred-target-column", default="reference", help="Target column for ASTrED comparison"
    )
    parser.add_argument(
        "--astred-alignment-column",
        default=None,
        help="Optional column containing word alignments in GIZA format for ASTrED.",
    )
    parser.add_argument(
        "--astred-source-prediction",
        action="store_true",
        help="Also compute bilingual ASTrED from the English source to the generated translation.",
    )
    parser.add_argument(
        "--astred-source-reference",
        action="store_true",
        help="Also compute bilingual ASTrED from the English source to the human reference translation.",
    )
    parser.add_argument(
        "--astred-source-prediction-only",
        action="store_true",
        help="With --astred-only, retain existing prediction-reference ASTrED and refresh only bilingual ASTrED.",
    )
    parser.add_argument(
        "--astred-source-prediction-source-model-or-lang",
        default="en",
        help="Parser language/model for the source side of bilingual ASTrED.",
    )
    parser.add_argument(
        "--astred-source-prediction-target-model-or-lang",
        default=None,
        help="Parser language/model for the generated target side of bilingual ASTrED.",
    )
    parser.add_argument(
        "--astred-source-prediction-source-column",
        default="source",
        help="Source-text column for bilingual ASTrED.",
    )
    parser.add_argument(
        "--astred-source-prediction-target-column",
        default="prediction",
        help="Generated-translation column for bilingual ASTrED.",
    )
    parser.add_argument(
        "--astred-source-prediction-alignment-column",
        default=None,
        help="Optional Pharaoh/GIZA alignment column for bilingual ASTrED.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser


def run_baseline_analysis(args: argparse.Namespace | None = None) -> dict[str, Any]:
    """Run the baseline evaluation pipeline."""
    if args is None:
        args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    started_at_utc = _utc_now_iso()
    start_time = time.perf_counter()
    predictions_path = Path(args.predictions)
    predictions_df = load_predictions(predictions_path)
    output_dir = Path(args.output_dir) if args.output_dir else predictions_path.resolve().parent
    comparison_root_value = getattr(args, "comparison_root", None)
    comparison_root = (
        Path(comparison_root_value) if comparison_root_value else _infer_comparison_root(output_dir)
    )

    if args.astred_only:
        if args.metricx_only or args.lexical_diversity_only:
            raise ValueError(
                "--astred-only cannot be used with --metricx-only or --lexical-diversity-only."
            )
        if not args.force:
            raise ValueError(
                "--astred-only requires --force because it updates existing analysis outputs."
            )
        if not args.astred_model_or_lang:
            raise ValueError("--astred-only requires --astred-model-or-lang.")
        written = refresh_astred_outputs(
            predictions_df,
            predictions_path=predictions_path,
            output_dir=output_dir,
            astred_model_or_lang=args.astred_model_or_lang,
            astred_parser=args.astred_parser,
            astred_source_column=args.astred_source_column,
            astred_target_column=args.astred_target_column,
            astred_alignment_column=args.astred_alignment_column,
            astred_refresh_reference=not args.astred_source_prediction_only,
            astred_source_prediction=args.astred_source_prediction,
            astred_source_prediction_source_model_or_lang=(
                args.astred_source_prediction_source_model_or_lang
            ),
            astred_source_prediction_target_model_or_lang=(
                args.astred_source_prediction_target_model_or_lang
            ),
            astred_source_prediction_source_column=(
                args.astred_source_prediction_source_column
            ),
            astred_source_prediction_target_column=(
                args.astred_source_prediction_target_column
            ),
            astred_source_prediction_alignment_column=(
                args.astred_source_prediction_alignment_column
            ),
            astred_source_reference=(
                args.astred_source_reference or args.astred_source_prediction
            ),
            started_at_utc=started_at_utc,
            finished_at_utc=_utc_now_iso(),
            runtime_seconds=time.perf_counter() - start_time,
        )
        comparison_tsvs = write_prompt_comparisons(comparison_root) if comparison_root else {}
        logger.info("ASTrED-only analysis refresh written to %s", output_dir)
        return {
            "written": written,
            "summary": json.loads(
                (output_dir / "analysis_summary.json").read_text(encoding="utf-8")
            ),
            "comparison_tsvs": comparison_tsvs,
        }

    if args.lexical_diversity_only:
        if not args.force:
            raise ValueError(
                "--lexical-diversity-only requires --force because it updates existing analysis outputs."
            )
        written = refresh_corpus_lexical_diversity_outputs(
            predictions_df,
            predictions_path=predictions_path,
            output_dir=output_dir,
        )
        comparison_tsvs = write_prompt_comparisons(comparison_root) if comparison_root else {}
        logger.info("Corpus lexical-diversity refresh written to %s", output_dir)
        return {
            "written": written,
            "summary": json.loads(
                (output_dir / "analysis_summary.json").read_text(encoding="utf-8")
            ),
            "comparison_tsvs": comparison_tsvs,
        }

    if args.metricx_only:
        if not args.force:
            raise ValueError(
                "--metricx-only requires --force because it updates existing analysis outputs."
            )
        if not args.metricx_model:
            raise ValueError("--metricx-only requires --metricx-model.")
        written, score = refresh_metricx_outputs(
            predictions_df,
            predictions_path=predictions_path,
            output_dir=output_dir,
            metricx_model=args.metricx_model,
        )
        comparison_tsvs = write_prompt_comparisons(comparison_root) if comparison_root else {}
        logger.info("MetricX-only refresh written to %s with score %.6f", output_dir, score)
        summary_path = (
            output_dir / "analysis_summary.json"
            if (output_dir / "analysis_summary.json").exists()
            else output_dir / "automatic_metrics.json"
        )
        return {
            "written": written,
            "summary": json.loads(summary_path.read_text(encoding="utf-8")),
            "metricx": score,
            "comparison_tsvs": comparison_tsvs,
        }

    background_texts = (
        load_background_texts(args.background_file, args.background_column)
        if args.background_file
        else None
    )
    analysis_result = analyze_predictions(
        predictions_df,
        comet_model=args.comet_model,
        xcomet_model=args.xcomet_model,
        comet_qe_model=args.comet_qe_model,
        cometkiwi_model=args.cometkiwi_model,
        metricx_model=args.metricx_model,
        background_texts=background_texts,
        comet_batch_size=args.comet_batch_size,
        comet_gpus=args.comet_gpus,
        stanza_lang=args.stanza_lang,
        sfa_target_lang=args.sfa_target_lang,
        sfa_source_lang=args.sfa_source_lang,
        sfa_dictionary_path=args.sfa_dictionary_path,
        sfa_dictionary_invert=True if args.sfa_invert_dictionary else None,
        astred_model_or_lang=args.astred_model_or_lang,
        astred_parser=args.astred_parser,
        astred_source_column=args.astred_source_column,
        astred_target_column=args.astred_target_column,
        astred_alignment_column=args.astred_alignment_column,
        astred_source_prediction=args.astred_source_prediction,
        astred_source_prediction_source_model_or_lang=(
            args.astred_source_prediction_source_model_or_lang
        ),
        astred_source_prediction_target_model_or_lang=(
            args.astred_source_prediction_target_model_or_lang
        ),
        astred_source_prediction_source_column=args.astred_source_prediction_source_column,
        astred_source_prediction_target_column=args.astred_source_prediction_target_column,
        astred_source_prediction_alignment_column=(
            args.astred_source_prediction_alignment_column
        ),
        astred_source_reference=(
            args.astred_source_reference or args.astred_source_prediction
        ),
    )
    finished_at_utc = _utc_now_iso()
    runtime_seconds = time.perf_counter() - start_time
    metadata = _analysis_metadata(
        predictions_df=predictions_df,
        predictions_path=predictions_path,
        output_dir=output_dir,
        args=args,
        started_at_utc=started_at_utc,
        finished_at_utc=finished_at_utc,
        runtime_seconds=runtime_seconds,
    )
    analysis_result["analysis_metadata"] = metadata
    analysis_result["summary"]["analysis_metadata"] = metadata
    analysis_result["summary_table"] = _flatten_summary(analysis_result["summary"])
    analysis_result["report_markdown"] = _build_report(analysis_result["summary"])
    written = persist_analysis_outputs(analysis_result, output_dir, force=args.force)
    comparison_tsvs = write_prompt_comparisons(comparison_root) if comparison_root else {}
    logger.info("Analysis outputs written to %s", output_dir)
    return {
        "written": written,
        "summary": analysis_result["summary"],
        "comparison_tsvs": comparison_tsvs,
    }


if __name__ == "__main__":
    run_baseline_analysis()
