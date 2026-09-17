"""Run translation-quality and naturalness evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.evaluation.metrics.astred import (
    compute_astred_metrics,
    empty_astred_summary,
)
from src.evaluation.metrics.automatic import (
    DEFAULT_COMETKIWI_MODEL,
    DEFAULT_METRICX_MODEL,
    DEFAULT_XCOMET_XXL_MODEL,
    compute_automatic_metrics,
    compute_per_example_metrics,
    compute_target_lexical_metrics,
)
from src.evaluation.metrics.linguistic import (
    annotate_texts,
    compute_stanza_summary_metrics,
)
from src.evaluation.metrics.synonyms import (
    collect_sfa_option_counts,
    empty_sfa_summary,
    load_apertium_translation_options,
    resolve_default_apertium_dictionary,
    summarize_sfa_metrics,
)
from src.utils.errors import PipelineError
from src.utils.io import save_dataframe_csv, save_dataframe_jsonl, save_json

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
    column.replace("astred_", "astred_source_prediction_", 1) for column in _ASTRED_COLUMNS
)
_ASTRED_SOURCE_REFERENCE_COLUMNS = tuple(
    column.replace("astred_", "astred_source_reference_", 1) for column in _ASTRED_COLUMNS
)


def _rename_astred_columns(per_example: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    """Prefix columns from an additional ASTrED comparison."""
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
        "analysis_module": "src.evaluation.run",
        "output_dir": str(output_dir),
        "input": _prediction_traceability(predictions_df, predictions_path),
        "generation_run_metadata": _load_generation_metadata(predictions_path),
        "metric_requests": {
            "comet_model": args.comet_model,
            "xcomet_model": args.xcomet_model,
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
            "astred_source_prediction_source_column": (args.astred_source_prediction_source_column),
            "astred_source_prediction_target_column": (args.astred_source_prediction_target_column),
        },
    }


def load_predictions(path: str | Path) -> pd.DataFrame:
    """Load predictions from JSONL or Parquet."""
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


def _optional_metric(label: str, compute: Callable[[], Any]) -> Any | None:
    """Return None when an optional metric cannot be computed."""
    try:
        return compute()
    except (KeyError, PipelineError, ValueError) as exc:
        logger.warning("Skipping %s: %s", label, exc)
        return None


def _compute_morphology(
    reference_texts: list[str], prediction_texts: list[str], language: str
) -> tuple[dict[str, list[list[dict[str, Any]]]], dict[str, dict[str, Any]]]:
    annotations = {
        "reference": annotate_texts(reference_texts, lang=language),
        "prediction": annotate_texts(prediction_texts, lang=language),
    }
    summaries = {
        role: {
            key: value
            for key, value in compute_stanza_summary_metrics(role_annotations).items()
            if key in _MORPHOLOGICAL_DIVERSITY_KEYS
        }
        for role, role_annotations in annotations.items()
    }
    return annotations, summaries


def _compute_sfa(
    predictions_df: pd.DataFrame,
    reference_texts: list[str],
    prediction_texts: list[str],
    source_language: str,
    target_language: str,
    dictionary_path: str | Path | None,
    dictionary_invert: bool | None,
    target_annotations: dict[str, list[list[dict[str, Any]]]],
    annotation_language: str | None,
) -> dict[str, dict[str, Any]]:
    dictionary = (
        {"path": Path(dictionary_path), "invert": bool(dictionary_invert), "resource": "custom"}
        if dictionary_path is not None
        else resolve_default_apertium_dictionary(target_language)
    )
    if dictionary_invert is not None:
        dictionary["invert"] = bool(dictionary_invert)
    options = load_apertium_translation_options(
        dictionary["path"], invert=bool(dictionary["invert"])
    )
    source_annotations = annotate_texts(
        predictions_df["source"].fillna("").astype(str).tolist(), lang=source_language
    )
    if not target_annotations or annotation_language != target_language:
        target_annotations = {
            "reference": annotate_texts(reference_texts, lang=target_language),
            "prediction": annotate_texts(prediction_texts, lang=target_language),
        }

    summaries = {}
    for role in ("reference", "prediction"):
        counts = collect_sfa_option_counts(source_annotations, target_annotations[role], options)
        role_summary = summarize_sfa_metrics(
            counts["counts_by_source_lemma"],
            sentences_with_candidates=int(counts["sentences_with_candidates"]),
            dictionary_path=dictionary["path"],
            dictionary_resource=dictionary.get("resource"),
            dictionary_inverted=bool(dictionary["invert"]),
        )
        role_summary["dictionary_entries"] = len(options)
        summaries[role] = role_summary
    return summaries


def analyze_predictions(
    predictions_df: pd.DataFrame,
    comet_model: str | None = None,
    xcomet_model: str | None = None,
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
    """Compute translation-quality and target-side naturalness metrics."""
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
        cometkiwi_model=cometkiwi_model,
        metricx_model=metricx_model,
        comet_batch_size=comet_batch_size,
        comet_gpus=comet_gpus,
    )
    if "sentence_boundary_contract_valid" in predictions_df:
        contract_values = (
            predictions_df["sentence_boundary_contract_valid"].fillna(False).astype(bool)
        )
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
        }
    )

    target_annotations: dict[str, list[list[dict[str, Any]]]] = {}
    if stanza_lang:
        morphology = _optional_metric(
            "Stanza morphology",
            lambda: _compute_morphology(reference_texts, prediction_texts, stanza_lang),
        )
        if morphology is not None:
            target_annotations, morphology_summaries = morphology
            for role, role_summary in morphology_summaries.items():
                summary[f"{role}_morphological_diversity"] = role_summary

    if sfa_target_lang:
        sfa_summaries = _optional_metric(
            "SFA metrics",
            lambda: _compute_sfa(
                predictions_df,
                reference_texts,
                prediction_texts,
                sfa_source_lang,
                sfa_target_lang,
                sfa_dictionary_path,
                sfa_dictionary_invert,
                target_annotations,
                stanza_lang,
            ),
        )
        if sfa_summaries is not None:
            for role, role_summary in sfa_summaries.items():
                summary[f"{role}_synonym_frequency_analysis"] = role_summary

    if astred_model_or_lang:
        result = _optional_metric(
            "ASTrED metrics",
            lambda: compute_astred_metrics(
                predictions_df,
                source_model_or_lang=astred_model_or_lang,
                parser=astred_parser,
                source_column=astred_source_column,
                target_column=astred_target_column,
                alignment_column=astred_alignment_column,
            ),
        )
        if result is not None:
            per_example = per_example.join(result["per_example"])
            summary["astred"] = result["summary"]

    target_model_or_lang = astred_source_prediction_target_model_or_lang or astred_model_or_lang
    if (astred_source_prediction or astred_source_reference) and not target_model_or_lang:
        raise ValueError(
            "Source-side ASTrED requires --astred-source-prediction-target-model-or-lang."
        )

    if astred_source_prediction:
        result = _optional_metric(
            "source-to-prediction ASTrED metrics",
            lambda: compute_astred_metrics(
                predictions_df,
                source_model_or_lang=astred_source_prediction_source_model_or_lang,
                target_model_or_lang=target_model_or_lang,
                parser=astred_parser,
                source_column=astred_source_prediction_source_column,
                target_column=astred_source_prediction_target_column,
                alignment_column=astred_source_prediction_alignment_column,
            ),
        )
        if result is not None:
            per_example = per_example.join(
                _rename_astred_columns(result["per_example"], prefix="astred_source_prediction_")
            )
            summary["astred_source_prediction"] = result["summary"]

    if astred_source_reference:
        result = _optional_metric(
            "source-to-reference ASTrED metrics",
            lambda: compute_astred_metrics(
                predictions_df,
                source_model_or_lang=astred_source_prediction_source_model_or_lang,
                target_model_or_lang=target_model_or_lang,
                parser=astred_parser,
                source_column=astred_source_prediction_source_column,
                target_column="reference",
                alignment_column=None,
            ),
        )
        if result is not None:
            per_example = per_example.join(
                _rename_astred_columns(result["per_example"], prefix="astred_source_reference_")
            )
            summary["astred_source_reference"] = result["summary"]

    summary["implemented_metrics_note"] = (
        "Translation quality is reported separately (BLEU, TER, chrF++, optional COMET-family metrics, and MetricX-24). "
        "Target-side reference and prediction sections report corpus-level TTR, Yule's K/I, MTLD, "
        "fixed B1/B2/B3 lexical-frequency bands, lemma-to-wordform Shannon/Simpson morphology, "
        "and dictionary-grounded SFA when Stanza and an Apertium dictionary are available."
    )
    return {
        "summary": summary,
        "summary_table": _flatten_summary(summary),
        "per_example": per_example,
    }


def persist_analysis_outputs(
    analysis_result: dict[str, Any],
    output_dir: str | Path,
    force: bool = False,
) -> dict[str, str]:
    """Write the summary and per-example metric files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "summary_json": output_dir / "analysis_summary.json",
        "summary_csv": output_dir / "analysis_summary.csv",
        "per_example_jsonl": output_dir / "per_example_metrics.jsonl",
        "per_example_csv": output_dir / "per_example_metrics.csv",
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
    save_json(analysis_result.get("analysis_metadata", {}), paths["analysis_metadata_json"])
    return {key: str(value) for key, value in paths.items()}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate translation quality and naturalness.")
    parser.add_argument(
        "--predictions", required=True, help="Path to predictions.jsonl or predictions.parquet"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for analysis outputs; defaults to the predictions parent directory",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing analysis outputs")
    parser.add_argument("--comet-model", default=None, help="Optional COMET model identifier")
    parser.add_argument(
        "--xcomet-model",
        default=None,
        help=f"Optional xCOMET model identifier, for example {DEFAULT_XCOMET_XXL_MODEL}",
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


def run_evaluation(args: argparse.Namespace | None = None) -> dict[str, Any]:
    """Evaluate one prediction file and write its metrics."""
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
    background_texts = (
        load_background_texts(args.background_file, args.background_column)
        if args.background_file
        else None
    )
    analysis_result = analyze_predictions(
        predictions_df,
        comet_model=args.comet_model,
        xcomet_model=args.xcomet_model,
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
        astred_source_prediction_alignment_column=(args.astred_source_prediction_alignment_column),
        astred_source_reference=(args.astred_source_reference or args.astred_source_prediction),
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
    written = persist_analysis_outputs(analysis_result, output_dir, force=args.force)
    logger.info("Analysis outputs written to %s", output_dir)
    return {
        "written": written,
        "summary": analysis_result["summary"],
    }


if __name__ == "__main__":
    run_evaluation()
