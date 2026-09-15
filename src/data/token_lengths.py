"""Tokenizer-based length analysis and filtering utilities for MT datasets."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from src.prompting._shared import load_processed_split
from src.utils.config import (
    ConfigError,
    get_dataset_entry,
    get_experiment_entry,
    get_model_entry,
)
from src.utils.hf_auth import get_hf_token
from src.utils.io import save_dataframe_csv, save_dataframe_jsonl, save_json
from src.utils.model_adapters import build_prompt_text, build_supervised_translation_text


_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")
_DEFAULT_RESULTS_DIR = Path("results/token_lengths")
_SUMMARY_COLUMNS = [
    "metric",
    "count",
    "mean",
    "std",
    "min",
    "max",
    "p50",
    "p75",
    "p90",
    "p95",
    "p99",
]


@dataclass(frozen=True)
class InputSpec:
    """Description of one corpus to analyse."""

    label: str
    source_name: str
    records: pd.DataFrame
    source_lang: str
    target_lang: str
    split: str | None = None


def _slugify(value: str) -> str:
    return _SLUG_RE.sub("_", value).strip("_") or "run"


def _round_up(value: float | int, multiple: int) -> int:
    value = max(float(value), 1.0)
    return int(math.ceil(value / multiple) * multiple)


def _token_count(tokenizer: Any, text: Any) -> int:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return 0
    return int(len(tokenizer.encode(str(text), add_special_tokens=False)))


def _summary_stats(values: Iterable[int]) -> dict[str, float | int | None]:
    array = np.array(list(values), dtype=np.int64)
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": int(array.min()),
        "max": int(array.max()),
        "p50": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def _fraction_at_or_below(values: Iterable[int], threshold: int) -> float | None:
    series = pd.Series(list(values), dtype="int64")
    if series.empty:
        return None
    return float((series <= int(threshold)).mean())


def _summary_to_frame(summary: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for metric, stats in summary.items():
        row = {"metric": metric}
        row.update(stats)
        rows.append(row)
    return pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)


def _load_tokenizer(
    model_name_or_key: str, models_config_path: str | Path = "configs/models.yaml"
) -> tuple[Any, str, int | None]:
    try:
        model_entry = get_model_entry(model_name_or_key, models_config_path)
        model_id = model_entry.hf_id
        model_context_length = model_entry.max_context_length
    except ConfigError:
        model_id = model_name_or_key
        model_context_length = None

    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "transformers is required for token-length analysis. Install the project dependencies first."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=get_hf_token(),
        trust_remote_code=True,
        padding_side="left",
    )
    if (
        getattr(tokenizer, "pad_token_id", None) is None
        and getattr(tokenizer, "eos_token", None) is not None
    ):
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer, model_id, model_context_length


def _resolve_prompt_spec(
    experiment_key: str | None,
    experiments_config_path: str | Path = "configs/experiments.yaml",
) -> tuple[dict[str, Any] | None, str | None]:
    if experiment_key is None:
        return None, None
    experiment_entry = get_experiment_entry(experiment_key, experiments_config_path)
    if "prompt" not in experiment_entry:
        raise ValueError(
            f"Experiment '{experiment_key}' does not define a single prompt mapping and cannot be used for length analysis."
        )
    return dict(experiment_entry["prompt"]), experiment_key


def _build_input_spec_from_dataset(
    dataset_key: str,
    split: str,
    processed_dir: str | Path,
    dataset_config_path: str | Path,
) -> InputSpec:
    dataset_entry = get_dataset_entry(dataset_key, dataset_config_path)
    records = load_processed_split(dataset_key, split, processed_dir).copy()
    return InputSpec(
        label=f"{dataset_key}_{split}",
        source_name=dataset_key,
        records=records,
        source_lang=str(dataset_entry["source_lang"]),
        target_lang=str(dataset_entry["target_lang"]),
        split=split,
    )


def _build_input_spec_from_jsonl(
    jsonl_path: str | Path,
    source_column: str,
    target_column: str,
    target_lang: str,
    source_lang: str = "en",
) -> InputSpec:
    path = Path(jsonl_path)
    records = pd.read_json(path, lines=True)
    missing = {source_column, target_column} - set(records.columns)
    if missing:
        raise ValueError(
            f"JSONL input {path} is missing columns {sorted(missing)}. "
            f"Available columns: {list(records.columns)}"
        )
    return InputSpec(
        label=path.stem,
        source_name=str(path),
        records=records,
        source_lang=source_lang,
        target_lang=target_lang,
        split=None,
    )


def _build_input_spec_from_text(
    source_file: str | Path,
    target_file: str | Path,
    target_lang: str,
    source_lang: str = "en",
) -> InputSpec:
    source_path = Path(source_file)
    target_path = Path(target_file)
    source_lines = source_path.read_text(encoding="utf-8").splitlines()
    target_lines = target_path.read_text(encoding="utf-8").splitlines()
    if len(source_lines) != len(target_lines):
        raise ValueError(
            f"Plain-text inputs must be aligned line-by-line, but {source_path} has {len(source_lines)} lines "
            f"and {target_path} has {len(target_lines)} lines."
        )
    records = pd.DataFrame(
        {
            "line_number": list(range(1, len(source_lines) + 1)),
            "source": source_lines,
            "target": target_lines,
        }
    )
    return InputSpec(
        label=source_path.stem,
        source_name=f"{source_path}|{target_path}",
        records=records,
        source_lang=source_lang,
        target_lang=target_lang,
        split=None,
    )


def _resolve_record_identifier(records: pd.DataFrame) -> pd.Series:
    for column in ("sentence_id", "id", "example_id", "line_number"):
        if column in records.columns:
            return records[column].astype(str)
    return pd.Series([str(index) for index in range(len(records))], index=records.index)


def analyze_input_spec(
    input_spec: InputSpec,
    *,
    tokenizer: Any,
    model_name_or_key: str,
    prompt_spec: Mapping[str, Any] | None = None,
    experiment_key: str | None = None,
    models_config_path: str | Path = "configs/models.yaml",
    limit: int | None = None,
) -> dict[str, Any]:
    records = input_spec.records.copy()
    if limit is not None:
        records = records.head(limit).copy()
    if records.empty:
        raise ValueError(f"No records available for analysis in '{input_spec.label}'.")

    try:
        model_entry = get_model_entry(model_name_or_key, models_config_path)
    except ConfigError:
        model_entry = None

    source_column = "source" if "source" in records.columns else "source_raw"
    target_column = "target" if "target" in records.columns else "target_raw"
    record_ids = _resolve_record_identifier(records)

    per_example: list[dict[str, Any]] = []
    for index, row in records.iterrows():
        source_text = str(row[source_column])
        target_text = str(row[target_column])
        prompt_text = ""
        full_text = ""
        if model_entry is not None:
            prompt_text = build_prompt_text(
                model_entry,
                source_text=source_text,
                target_lang=input_spec.target_lang,
                tokenizer=tokenizer,
                enable_thinking=False,
                prompt_spec=prompt_spec,
            )
            full_text = build_supervised_translation_text(
                model_entry,
                source_text=source_text,
                target_text=target_text,
                target_lang=input_spec.target_lang,
                tokenizer=tokenizer,
                enable_thinking=False,
                prompt_spec=prompt_spec,
            )

        source_tokens = _token_count(tokenizer, source_text)
        target_tokens = _token_count(tokenizer, target_text)
        prompt_tokens = _token_count(tokenizer, prompt_text) if prompt_text else None
        total_tokens = _token_count(tokenizer, full_text) if full_text else None
        completion_tokens = (
            int(total_tokens - prompt_tokens)
            if total_tokens is not None and prompt_tokens is not None
            else None
        )

        per_example.append(
            {
                "record_id": str(record_ids.loc[index]),
                "source_name": input_spec.source_name,
                "split": input_spec.split,
                "source_lang": input_spec.source_lang,
                "target_lang": input_spec.target_lang,
                "experiment_key": experiment_key,
                "source_text": source_text,
                "target_text": target_text,
                "prompt_text": prompt_text or None,
                "source_tokens": source_tokens,
                "target_tokens": target_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
        )

    frame = pd.DataFrame(per_example)
    summary = {
        "source_tokens": _summary_stats(frame["source_tokens"].astype(int).tolist()),
        "target_tokens": _summary_stats(frame["target_tokens"].astype(int).tolist()),
    }
    if model_entry is not None:
        summary["prompt_tokens"] = _summary_stats(
            frame["prompt_tokens"].fillna(0).astype(int).tolist()
        )
        summary["completion_tokens"] = _summary_stats(
            frame["completion_tokens"].fillna(0).astype(int).tolist()
        )
        summary["total_tokens"] = _summary_stats(
            frame["total_tokens"].fillna(0).astype(int).tolist()
        )

    recommendations = build_recommendations(
        summary,
        per_example_frame=frame,
        model_context_length=(model_entry.max_context_length if model_entry is not None else None),
    )

    return {
        "label": input_spec.label,
        "source_name": input_spec.source_name,
        "rows": int(len(frame)),
        "source_lang": input_spec.source_lang,
        "target_lang": input_spec.target_lang,
        "split": input_spec.split,
        "model_name_or_key": model_name_or_key,
        "experiment_key": experiment_key,
        "summary": summary,
        "recommendations": recommendations,
        "per_example": frame,
    }


def build_recommendations(
    summary: Mapping[str, Mapping[str, Any]],
    *,
    per_example_frame: pd.DataFrame | None = None,
    model_context_length: int | None = None,
) -> dict[str, Any]:
    prompt_stats = summary.get("prompt_tokens")
    completion_stats = summary.get("completion_tokens")
    total_stats = summary.get("total_tokens")
    if not prompt_stats or not completion_stats or not total_stats:
        return {
            "notes": [
                "Prompt/completion/total recommendations require a project model key so the real prompt template can be rendered."
            ]
        }

    balanced_prompt = _round_up(float(prompt_stats["p95"]), 64)
    conservative_prompt = _round_up(float(prompt_stats["p99"]), 64)
    balanced_completion = _round_up(float(completion_stats["p95"]), 64)
    conservative_completion = _round_up(float(completion_stats["p99"]), 64)
    balanced_total = _round_up(float(total_stats["p95"]), 128)
    conservative_total = _round_up(float(total_stats["p99"]), 128)

    if model_context_length is not None:
        balanced_total = min(balanced_total, int(model_context_length))
        conservative_total = min(conservative_total, int(model_context_length))

    balanced_seq_length = max(balanced_total, balanced_prompt + balanced_completion)
    conservative_seq_length = max(conservative_total, conservative_prompt + conservative_completion)
    if model_context_length is not None:
        balanced_seq_length = min(int(balanced_seq_length), int(model_context_length))
        conservative_seq_length = min(int(conservative_seq_length), int(model_context_length))

    prompt_values = (
        per_example_frame["prompt_tokens"].fillna(0).astype(int).tolist()
        if per_example_frame is not None and "prompt_tokens" in per_example_frame
        else []
    )
    completion_values = (
        per_example_frame["completion_tokens"].fillna(0).astype(int).tolist()
        if per_example_frame is not None and "completion_tokens" in per_example_frame
        else []
    )
    total_values = (
        per_example_frame["total_tokens"].fillna(0).astype(int).tolist()
        if per_example_frame is not None and "total_tokens" in per_example_frame
        else []
    )

    balanced = {
        "max_new_tokens": balanced_completion,
        "max_prompt_length": balanced_prompt,
        "max_completion_length": balanced_completion,
        "max_seq_length": balanced_seq_length,
        "coverage": {
            "prompt_lte_threshold": _fraction_at_or_below(prompt_values, balanced_prompt),
            "completion_lte_threshold": _fraction_at_or_below(
                completion_values, balanced_completion
            ),
            "total_lte_threshold": _fraction_at_or_below(total_values, balanced_seq_length),
        },
        "based_on": {"prompt": "p95", "completion": "p95", "total": "p95"},
    }
    conservative = {
        "max_new_tokens": conservative_completion,
        "max_prompt_length": conservative_prompt,
        "max_completion_length": conservative_completion,
        "max_seq_length": conservative_seq_length,
        "coverage": {
            "prompt_lte_threshold": _fraction_at_or_below(prompt_values, conservative_prompt),
            "completion_lte_threshold": _fraction_at_or_below(
                completion_values, conservative_completion
            ),
            "total_lte_threshold": _fraction_at_or_below(total_values, conservative_seq_length),
        },
        "based_on": {"prompt": "p99", "completion": "p99", "total": "p99"},
    }

    return {
        "balanced": balanced,
        "conservative": conservative,
        "notes": [
            "Balanced settings track p95 and suit efficient iteration.",
            "Conservative settings track p99 and are safer for GRPO/SFT runs where truncation is costly.",
            "Completion-based recommendations should be used for max_new_tokens; prompt/completion/seq settings are intended for supervised or RL training.",
        ],
    }


def compare_analyses(analyses: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if len(analyses) < 2:
        return pd.DataFrame()

    for left_index in range(len(analyses)):
        for right_index in range(left_index + 1, len(analyses)):
            left = analyses[left_index]
            right = analyses[right_index]
            for metric in (
                "source_tokens",
                "target_tokens",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
            ):
                left_stats = left["summary"].get(metric)
                right_stats = right["summary"].get(metric)
                if not left_stats or not right_stats:
                    continue
                for percentile_key in ("p50", "p95", "p99", "mean"):
                    left_value = left_stats.get(percentile_key)
                    right_value = right_stats.get(percentile_key)
                    if left_value in (None, 0) or right_value is None:
                        ratio = None
                    else:
                        ratio = float(right_value) / float(left_value)
                    rows.append(
                        {
                            "analysis_a": left["label"],
                            "analysis_b": right["label"],
                            "metric": metric,
                            "statistic": percentile_key,
                            "value_a": left_value,
                            "value_b": right_value,
                            "ratio_b_over_a": ratio,
                            "absolute_difference": (
                                None
                                if left_value is None or right_value is None
                                else float(right_value) - float(left_value)
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def assess_pairwise_comparability(comparisons: pd.DataFrame) -> list[dict[str, Any]]:
    if comparisons.empty:
        return []
    summaries: list[dict[str, Any]] = []
    grouped = comparisons.groupby(["analysis_a", "analysis_b"], dropna=False)
    for (analysis_a, analysis_b), group in grouped:
        focus = group[
            group["metric"].isin(["target_tokens", "completion_tokens", "total_tokens"])
            & group["statistic"].isin(["p50", "p95"])
        ].copy()
        ratios = focus["ratio_b_over_a"].dropna().astype(float)
        comparable = bool(not ratios.empty and ratios.between(0.85, 1.15).all())
        summaries.append(
            {
                "analysis_a": analysis_a,
                "analysis_b": analysis_b,
                "heuristically_comparable": comparable,
                "criterion": "All target/completion/total p50 and p95 ratios must stay within [0.85, 1.15].",
                "min_ratio": (None if ratios.empty else float(ratios.min())),
                "max_ratio": (None if ratios.empty else float(ratios.max())),
            }
        )
    return summaries


def _ensure_input_specs(
    *,
    datasets: list[str] | None,
    split: str,
    processed_dir: str | Path,
    dataset_config_path: str | Path,
    jsonl_path: str | None,
    source_column: str,
    target_column: str,
    source_file: str | None,
    target_file: str | None,
    source_lang: str,
    target_lang: str | None,
) -> list[InputSpec]:
    specs: list[InputSpec] = []
    if datasets:
        for dataset_key in datasets:
            specs.append(
                _build_input_spec_from_dataset(
                    dataset_key,
                    split=split,
                    processed_dir=processed_dir,
                    dataset_config_path=dataset_config_path,
                )
            )
        return specs

    if jsonl_path:
        if not target_lang:
            raise ValueError("--target-lang is required when analysing a standalone JSONL file.")
        return [
            _build_input_spec_from_jsonl(
                jsonl_path,
                source_column=source_column,
                target_column=target_column,
                target_lang=target_lang,
                source_lang=source_lang,
            )
        ]

    if source_file and target_file:
        if not target_lang:
            raise ValueError("--target-lang is required when analysing paired plain-text files.")
        return [
            _build_input_spec_from_text(
                source_file,
                target_file,
                target_lang=target_lang,
                source_lang=source_lang,
            )
        ]

    raise ValueError("Provide either --dataset, --jsonl, or both --source-file and --target-file.")


def run_token_length_analysis(
    *,
    datasets: list[str] | None = None,
    split: str = "train",
    processed_dir: str | Path = "data/processed",
    dataset_config_path: str | Path = "configs/datasets.yaml",
    jsonl_path: str | None = None,
    source_column: str = "source",
    target_column: str = "target",
    source_file: str | None = None,
    target_file: str | None = None,
    source_lang: str = "en",
    target_lang: str | None = None,
    model_name_or_key: str = "latxa_8b_instruct",
    experiment_key: str | None = "p0_baseline",
    experiments_config_path: str | Path = "configs/experiments.yaml",
    models_config_path: str | Path = "configs/models.yaml",
    results_dir: str | Path = _DEFAULT_RESULTS_DIR,
    run_name: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    input_specs = _ensure_input_specs(
        datasets=datasets,
        split=split,
        processed_dir=processed_dir,
        dataset_config_path=dataset_config_path,
        jsonl_path=jsonl_path,
        source_column=source_column,
        target_column=target_column,
        source_file=source_file,
        target_file=target_file,
        source_lang=source_lang,
        target_lang=target_lang,
    )
    prompt_spec, resolved_experiment_key = _resolve_prompt_spec(
        experiment_key,
        experiments_config_path=experiments_config_path,
    )
    tokenizer, resolved_model_id, model_context_length = _load_tokenizer(
        model_name_or_key,
        models_config_path=models_config_path,
    )

    base_name = run_name or "__".join(spec.label for spec in input_specs)
    output_dir = Path(results_dir) / _slugify(base_name) / _slugify(model_name_or_key)
    output_dir.mkdir(parents=True, exist_ok=True)

    analyses = []
    for input_spec in input_specs:
        analysis = analyze_input_spec(
            input_spec,
            tokenizer=tokenizer,
            model_name_or_key=model_name_or_key,
            prompt_spec=prompt_spec,
            experiment_key=resolved_experiment_key,
            models_config_path=models_config_path,
            limit=limit,
        )
        if "conservative" in analysis["recommendations"] and model_context_length is not None:
            analysis["recommendations"]["model_context_length"] = model_context_length
        analyses.append(analysis)

        per_example_path = output_dir / f"{_slugify(input_spec.label)}_per_example.jsonl"
        summary_json_path = output_dir / f"{_slugify(input_spec.label)}_summary.json"
        summary_csv_path = output_dir / f"{_slugify(input_spec.label)}_summary.csv"
        save_dataframe_jsonl(analysis["per_example"], per_example_path)
        save_json(
            {
                "label": analysis["label"],
                "source_name": analysis["source_name"],
                "rows": analysis["rows"],
                "source_lang": analysis["source_lang"],
                "target_lang": analysis["target_lang"],
                "split": analysis["split"],
                "model_name_or_key": model_name_or_key,
                "resolved_model_id": resolved_model_id,
                "experiment_key": resolved_experiment_key,
                "summary": analysis["summary"],
                "recommendations": analysis["recommendations"],
            },
            summary_json_path,
        )
        save_dataframe_csv(_summary_to_frame(analysis["summary"]), summary_csv_path)

    comparisons = compare_analyses(analyses)
    comparisons_path = None
    pairwise_summary_path = None
    if not comparisons.empty:
        comparisons_path = save_dataframe_csv(
            comparisons, output_dir / "cross_dataset_comparisons.csv"
        )
        pairwise_summary_path = save_json(
            assess_pairwise_comparability(comparisons),
            output_dir / "cross_dataset_comparability.json",
        )

    manifest = {
        "output_dir": str(output_dir),
        "datasets": [analysis["label"] for analysis in analyses],
        "model_name_or_key": model_name_or_key,
        "resolved_model_id": resolved_model_id,
        "experiment_key": resolved_experiment_key,
        "limit": limit,
        "comparison_csv": (None if comparisons_path is None else str(comparisons_path)),
        "comparability_json": (
            None if pairwise_summary_path is None else str(pairwise_summary_path)
        ),
    }
    save_json(manifest, output_dir / "manifest.json")
    return manifest


def _load_thresholds_from_analysis(
    analysis_json_path: str | Path,
    recommendation_profile: str,
) -> dict[str, int]:
    payload = json.loads(Path(analysis_json_path).read_text(encoding="utf-8"))
    recommendations = payload.get("recommendations", {})
    if recommendation_profile not in recommendations:
        raise ValueError(
            f"Analysis summary {analysis_json_path} does not contain recommendation profile '{recommendation_profile}'."
        )
    profile = recommendations[recommendation_profile]
    return {
        "max_prompt_length": int(profile["max_prompt_length"]),
        "max_completion_length": int(profile["max_completion_length"]),
        "max_seq_length": int(profile["max_seq_length"]),
    }


def run_length_filter(
    *,
    dataset_key: str | None = None,
    split: str = "train",
    processed_dir: str | Path = "data/processed",
    dataset_config_path: str | Path = "configs/datasets.yaml",
    jsonl_path: str | None = None,
    source_column: str = "source",
    target_column: str = "target",
    source_file: str | None = None,
    target_file: str | None = None,
    source_lang: str = "en",
    target_lang: str | None = None,
    model_name_or_key: str = "latxa_8b_instruct",
    experiment_key: str | None = "p0_baseline",
    experiments_config_path: str | Path = "configs/experiments.yaml",
    models_config_path: str | Path = "configs/models.yaml",
    results_dir: str | Path = "data/filtered",
    run_name: str | None = None,
    max_prompt_length: int | None = None,
    max_completion_length: int | None = None,
    max_seq_length: int | None = None,
    analysis_json_path: str | None = None,
    recommendation_profile: str = "conservative",
    limit: int | None = None,
) -> dict[str, Any]:
    if split == "test":
        raise ValueError(
            "Length filtering is only intended for training or development data, never test data."
        )
    if analysis_json_path:
        thresholds = _load_thresholds_from_analysis(analysis_json_path, recommendation_profile)
        max_prompt_length = max_prompt_length or thresholds["max_prompt_length"]
        max_completion_length = max_completion_length or thresholds["max_completion_length"]
        max_seq_length = max_seq_length or thresholds["max_seq_length"]
    if max_prompt_length is None or max_completion_length is None:
        raise ValueError(
            "Provide --max-prompt-length and --max-completion-length, or pass --analysis-json with a recommendation profile."
        )

    input_specs = _ensure_input_specs(
        datasets=([dataset_key] if dataset_key else None),
        split=split,
        processed_dir=processed_dir,
        dataset_config_path=dataset_config_path,
        jsonl_path=jsonl_path,
        source_column=source_column,
        target_column=target_column,
        source_file=source_file,
        target_file=target_file,
        source_lang=source_lang,
        target_lang=target_lang,
    )
    if len(input_specs) != 1:
        raise ValueError("Filtering supports exactly one input at a time.")

    prompt_spec, resolved_experiment_key = _resolve_prompt_spec(
        experiment_key,
        experiments_config_path=experiments_config_path,
    )
    tokenizer, _, _ = _load_tokenizer(model_name_or_key, models_config_path=models_config_path)
    analysis = analyze_input_spec(
        input_specs[0],
        tokenizer=tokenizer,
        model_name_or_key=model_name_or_key,
        prompt_spec=prompt_spec,
        experiment_key=resolved_experiment_key,
        models_config_path=models_config_path,
        limit=limit,
    )
    if "prompt_tokens" not in analysis["summary"] or "completion_tokens" not in analysis["summary"]:
        raise ValueError(
            "Filtering requires a project model key from configs/models.yaml so prompt/completion lengths can be rendered consistently."
        )

    trace = analysis["per_example"].copy()
    trace["keep"] = trace["prompt_tokens"].fillna(0).astype(int).le(int(max_prompt_length)) & trace[
        "completion_tokens"
    ].fillna(0).astype(int).le(int(max_completion_length))
    if max_seq_length is not None:
        trace["keep"] = trace["keep"] & trace["total_tokens"].fillna(0).astype(int).le(
            int(max_seq_length)
        )

    def _drop_reason(row: pd.Series) -> str | None:
        reasons = []
        if int(row["prompt_tokens"]) > int(max_prompt_length):
            reasons.append("prompt_too_long")
        if int(row["completion_tokens"]) > int(max_completion_length):
            reasons.append("completion_too_long")
        if max_seq_length is not None and int(row["total_tokens"]) > int(max_seq_length):
            reasons.append("total_too_long")
        if not reasons:
            return None
        return ",".join(reasons)

    trace["drop_reason"] = trace.apply(_drop_reason, axis=1)

    original_records = input_specs[0].records.copy()
    filtered_records = original_records.loc[trace["keep"].to_numpy()].copy()

    base_name = run_name or input_specs[0].label
    output_dir = Path(results_dir) / _slugify(base_name) / _slugify(model_name_or_key)
    output_dir.mkdir(parents=True, exist_ok=True)

    if dataset_key or jsonl_path:
        filtered_path = output_dir / f"{_slugify(base_name)}_filtered.jsonl"
        save_dataframe_jsonl(filtered_records, filtered_path)
    else:
        assert source_file is not None and target_file is not None
        filtered_source_path = output_dir / f"{Path(source_file).stem}.filtered.txt"
        filtered_target_path = output_dir / f"{Path(target_file).stem}.filtered.txt"
        filtered_source_path.write_text(
            "\n".join(filtered_records["source"].astype(str).tolist()) + "\n",
            encoding="utf-8",
        )
        filtered_target_path.write_text(
            "\n".join(filtered_records["target"].astype(str).tolist()) + "\n",
            encoding="utf-8",
        )
        filtered_path = output_dir / "filtered_text_paths.json"
        save_json(
            {
                "source_file": str(filtered_source_path),
                "target_file": str(filtered_target_path),
            },
            filtered_path,
        )

    summary = {
        "input_label": input_specs[0].label,
        "rows_before": int(len(original_records)),
        "rows_after": int(len(filtered_records)),
        "rows_removed": int(len(original_records) - len(filtered_records)),
        "fraction_kept": float(trace["keep"].mean()) if not trace.empty else None,
        "max_prompt_length": int(max_prompt_length),
        "max_completion_length": int(max_completion_length),
        "max_seq_length": (None if max_seq_length is None else int(max_seq_length)),
        "experiment_key": resolved_experiment_key,
        "model_name_or_key": model_name_or_key,
    }

    save_json(summary, output_dir / "filter_summary.json")
    save_dataframe_csv(pd.DataFrame([summary]), output_dir / "filter_summary.csv")
    save_dataframe_jsonl(trace, output_dir / "filter_trace.jsonl")
    return {
        "output_dir": str(output_dir),
        "filtered_output": str(filtered_path),
        "summary_json": str(output_dir / "filter_summary.json"),
        "trace_jsonl": str(output_dir / "filter_trace.jsonl"),
    }


def _build_analyze_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("analyze", help="Run tokenizer-based length analysis.")
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        help="Dataset key from configs/datasets.yaml. Repeat to compare multiple corpora.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--dataset-config", default="configs/datasets.yaml")
    parser.add_argument("--jsonl", default=None)
    parser.add_argument("--source-column", default="source")
    parser.add_argument("--target-column", default="target")
    parser.add_argument("--source-file", default=None)
    parser.add_argument("--target-file", default=None)
    parser.add_argument("--source-lang", default="en")
    parser.add_argument("--target-lang", default=None)
    parser.add_argument("--model", default="latxa_8b_instruct")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--experiment", default="p0_baseline")
    parser.add_argument("--experiments-config", default="configs/experiments.yaml")
    parser.add_argument("--results-dir", default=str(_DEFAULT_RESULTS_DIR))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--limit", type=int, default=None)


def _build_filter_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "filter", help="Filter training/development data by tokenized length."
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--dataset-config", default="configs/datasets.yaml")
    parser.add_argument("--jsonl", default=None)
    parser.add_argument("--source-column", default="source")
    parser.add_argument("--target-column", default="target")
    parser.add_argument("--source-file", default=None)
    parser.add_argument("--target-file", default=None)
    parser.add_argument("--source-lang", default="en")
    parser.add_argument("--target-lang", default=None)
    parser.add_argument("--model", default="latxa_8b_instruct")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--experiment", default="p0_baseline")
    parser.add_argument("--experiments-config", default="configs/experiments.yaml")
    parser.add_argument("--results-dir", default="data/filtered")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-prompt-length", type=int, default=None)
    parser.add_argument("--max-completion-length", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--analysis-json", default=None)
    parser.add_argument(
        "--recommendation-profile", default="conservative", choices=["balanced", "conservative"]
    )
    parser.add_argument("--limit", type=int, default=None)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Token-length analysis and filtering for MT data.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _build_analyze_parser(subparsers)
    _build_filter_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.command == "analyze":
        result = run_token_length_analysis(
            datasets=args.datasets,
            split=args.split,
            processed_dir=args.processed_dir,
            dataset_config_path=args.dataset_config,
            jsonl_path=args.jsonl,
            source_column=args.source_column,
            target_column=args.target_column,
            source_file=args.source_file,
            target_file=args.target_file,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            model_name_or_key=args.model,
            experiment_key=args.experiment,
            experiments_config_path=args.experiments_config,
            models_config_path=args.models_config,
            results_dir=args.results_dir,
            run_name=args.run_name,
            limit=args.limit,
        )
    else:
        result = run_length_filter(
            dataset_key=args.dataset,
            split=args.split,
            processed_dir=args.processed_dir,
            dataset_config_path=args.dataset_config,
            jsonl_path=args.jsonl,
            source_column=args.source_column,
            target_column=args.target_column,
            source_file=args.source_file,
            target_file=args.target_file,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            model_name_or_key=args.model,
            experiment_key=args.experiment,
            experiments_config_path=args.experiments_config,
            models_config_path=args.models_config,
            results_dir=args.results_dir,
            run_name=args.run_name,
            max_prompt_length=args.max_prompt_length,
            max_completion_length=args.max_completion_length,
            max_seq_length=args.max_seq_length,
            analysis_json_path=args.analysis_json,
            recommendation_profile=args.recommendation_profile,
            limit=args.limit,
        )
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
