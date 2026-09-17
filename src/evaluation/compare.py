"""Build one P0-P5 comparison CSV per language."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import save_dataframe_csv

_EXPERIMENTS: dict[str, tuple[str, str | None]] = {
    "P0": ("p0", None),
    "P1": ("p1", None),
    "P2": ("p2", None),
    "P3": ("p3", None),
    "P4 SFT": ("p4_sft", None),
    "P5 GRPO A0": ("p5_grpo", "a0"),
    "P5 GRPO A1": ("p5_grpo", "a1"),
    "P5 GRPO A2": ("p5_grpo", "a2"),
    "P5 GRPO A3": ("p5_grpo", "a3"),
    "P5 GRPO A4": ("p5_grpo", "a4"),
    "P5 GRPO A3v2": ("p5_grpo", "a3v2"),
    "P5 GRPO A3v3": ("p5_grpo", "a3v3"),
    "P5 GRPO A3v4": ("p5_grpo", "a3v4"),
    "P5 GRPO A3v5": ("p5_grpo", "a3v5"),
    "P5 GRPO A5": ("p5_grpo", "a5"),
}

_PROMPTING_MODELS = {
    "latxa_8b_instruct",
    "salamandra_7b_instruct",
    "latxa_70b_instruct",
    "gemma3_27b_it",
}
_POST_TRAINING_MODELS = {"latxa_8b_instruct", "salamandrata_7b_instruct"}
_PROMPTING_LABELS = {"P1", "P2", "P3"}
_POST_TRAINING_LABELS = {
    label for label in _EXPERIMENTS if label == "P4 SFT" or label.startswith("P5 GRPO")
}
_FINAL_TEST_LABELS = {"P0", "P4 SFT", "P5 GRPO A2", "P5 GRPO A5"}
_EXCLUDED_SECTIONS = {"analysis_metadata", "report_context", "implemented_metrics_note"}


_EXCLUDED_METRIC_KEYS = {
    "prediction_lexical_frequency_profile.beyond_2000",
    "reference_lexical_frequency_profile.beyond_2000",
}
_EXCLUDED_METRIC_LEAVES_BY_SECTION = {
    "synonym_frequency_analysis": {
        "cdu",
        "ptf",
        "dictionary_entries",
        "dictionary_inverted",
        "dictionary_resource",
        "sentences_with_candidates",
        "source_lemmas_considered",
        "source_lemmas_with_matches",
        "source_lemmas_with_dictionary_entries",
        "sfa_source_lemmas_considered",
        "sfa_source_lemmas_with_dictionary_entries",
        "sfa_valid_source_lemmas",
    },
    "morphological_diversity": {
        "morph_lemmas_used_for_entropy",
        "morph_multi_wordform_lemmas",
        "morph_single_wordform_lemmas",
    },
}

_REFERENCE_PREFIX = "reference_"
_PREDICTION_PREFIX = "prediction_"
_PAIRED_NATURALNESS_SECTIONS = {
    "lexical_diversity",
    "lexical_frequency_profile",
    "morphological_diversity",
    "synonym_frequency_analysis",
}


def _is_thesis_facing_metric(metric_key: str) -> bool:
    if metric_key in _EXCLUDED_METRIC_KEYS:
        return False
    section, _, leaf = metric_key.rpartition(".")
    base_section = section.removeprefix("reference_").removeprefix("prediction_")
    return leaf not in _EXCLUDED_METRIC_LEAVES_BY_SECTION.get(base_section, set())


def _metric_path(
    results_root: Path,
    experiment: str,
    ablation: str | None,
    dataset: str,
    model: str,
    split: str,
) -> Path:
    if experiment in {"p0", "p1", "p2", "p3"}:
        base = results_root / "p0_p3" / experiment / dataset / model
        split_path = base / split / "analysis_summary.json"
        return split_path if split_path.exists() else base / "analysis_summary.json"
    if experiment == "p4_sft":
        output_dir = results_root / "p4_sft" / dataset / model / split
        analysis_path = output_dir / "analysis_summary.json"
        return analysis_path if analysis_path.exists() else output_dir / "automatic_metrics.json"
    if experiment == "p5_grpo":
        assert ablation is not None
        output_dir = results_root / "p5_grpo" / ablation / dataset / model / split
        analysis_path = output_dir / "analysis_summary.json"
        return analysis_path if analysis_path.exists() else output_dir / "automatic_metrics.json"
    raise ValueError(f"Unsupported experiment: {experiment}")


def _numeric_metrics(value: Any, prefix: tuple[str, ...] = ()) -> dict[str, float | int]:
    if isinstance(value, dict):
        metrics: dict[str, float | int] = {}
        for key, nested in value.items():
            if not prefix and key in _EXCLUDED_SECTIONS:
                continue
            metrics.update(_numeric_metrics(nested, (*prefix, str(key))))
        return metrics
    if isinstance(value, (int, float)) and not isinstance(value, bool) and pd.notna(value):
        return {".".join(prefix): value}
    return {}


def _interpretation(metric_key: str) -> str:
    section, _, leaf = metric_key.rpartition(".")
    base_section = section.removeprefix("reference_").removeprefix("prediction_")

    if section == "adequacy":
        return "lower is better" if leaf in {"ter", "metricx"} else "higher is better"
    if section in _PAIRED_NATURALNESS_SECTIONS:
        return "closer to reference is better"
    if base_section == "lexical_frequency_profile":
        if leaf.startswith("b1_"):
            return "share in the most-frequent 1k band; match reference distribution"
        if leaf.startswith("b2_"):
            return "share in the 1k-2k band; match reference distribution"
        if leaf.startswith(("b3_", "beyond_")):
            return "share beyond the top 2k (least-frequent band); match reference distribution"
        return "frequency-band share; match reference distribution"
    if base_section == "synonym_frequency_analysis":
        if leaf in {"sfa_ptf", "sfa_cdu", "ptf", "cdu"}:
            return "lower means more diverse choice"
        if leaf == "sfa_syn_ttr":
            return "higher means more diverse choice"
        return ""
    if base_section == "morphological_diversity":
        if leaf == "morph_simpson_d":
            return "lower means more diversity"
        if leaf in {"morph_shannon_entropy", "morph_inverse_simpson_optional"}:
            return "higher means more diversity"
        return ""
    if section in {"astred", "astred_source_prediction", "astred_source_reference"}:
        if leaf in {"ted", "word_cross", "seq_cross", "sacr_cross"}:
            if section == "astred_source_reference":
                return "human reference structural distance from source; comparison baseline"
            return (
                "lower means closer to source structure"
                if section == "astred_source_prediction"
                else "lower means closer to reference structure"
            )
        return ""

    if base_section == "classifier_reference_likeness":
        return "higher means more reference-like according to the named classifier"
    if leaf in {
        "ter",
        "metricx",
        "ted",
        "word_cross",
        "seq_cross",
        "sacr_cross",
        "yules_k",
        "simpson_d",
    }:
        return "lower is better"
    if leaf in {
        "bleu",
        "chrf_pp",
        "comet",
        "xcomet",
        "cometkiwi",
        "ttr",
        "mtld",
        "yules_i",
    }:
        return "higher is better"
    return ""


def _comparison_metric_key(metric_key: str) -> tuple[str, str]:
    """Map reference and prediction values to the same metric row."""
    section, separator, leaf = metric_key.rpartition(".")
    if section.startswith(_REFERENCE_PREFIX):
        return (f"{section.removeprefix(_REFERENCE_PREFIX)}{separator}{leaf}", "reference")
    if section.startswith(_PREDICTION_PREFIX):
        return (f"{section.removeprefix(_PREDICTION_PREFIX)}{separator}{leaf}", "translation")
    if section == "astred_source_reference":
        return (metric_key, "reference")
    return (metric_key, "translation")


def _metric_labels(metric_key: str) -> tuple[str, str]:
    group, _, metric = metric_key.rpartition(".")
    return (group.replace("_", " ").title() or "Overall", metric.replace("_", " "))


def _discover_models(results_root: Path, dataset: str) -> tuple[str, ...]:
    models: set[str] = set()
    for _, (experiment, ablation) in _EXPERIMENTS.items():
        if experiment in {"p0", "p1", "p2", "p3"}:
            parent = results_root / "p0_p3" / experiment / dataset
        elif experiment == "p4_sft":
            parent = results_root / experiment / dataset
        else:
            assert ablation is not None
            parent = results_root / experiment / ablation / dataset
        if parent.is_dir():
            models.update(path.name for path in parent.iterdir() if path.is_dir())
    return tuple(sorted(models))


def _is_planned_experiment(
    dataset: str,
    split: str,
    model: str,
    experiment_label: str,
) -> bool:
    if split == "test":
        if experiment_label == "P0":
            return True
        if experiment_label in _PROMPTING_LABELS:
            return model in _PROMPTING_MODELS
        if experiment_label in _POST_TRAINING_LABELS:
            return model in _POST_TRAINING_MODELS
        return False

    if split == "flores_test":
        if model == "latxa_8b_instruct":
            return True
        if model == "salamandrata_7b_instruct":
            return experiment_label == "P0" or experiment_label in _POST_TRAINING_LABELS
        return experiment_label == "P0"

    if split in {"news_test", "literary_test"}:
        return model in _POST_TRAINING_MODELS and experiment_label in _FINAL_TEST_LABELS

    if split in {
        "literary_5sent_test",
        "literary_5sent_boundary_test",
        "literary_5sent_boundary_v2_test",
    }:
        return model == "latxa_8b_instruct" and experiment_label in {
            "P5 GRPO A2",
            "P5 GRPO A3v2",
            "P5 GRPO A3v5",
        }

    return False


def _load_metrics(path: Path) -> dict[str, float | int]:
    if not path.exists():
        return {}
    try:
        return _numeric_metrics(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return {}


def build_comparison_eval(
    results_root: str | Path = "results",
    datasets: tuple[str, ...] = ("en_ca", "en_eu"),
    models: tuple[str, ...] | None = None,
    split: str = "test",
    output_dir: str | Path = "results/comparisons",
) -> tuple[Path, ...]:
    """Write one metric-by-condition comparison CSV per language."""
    root = Path(results_root)
    destination_dir = Path(output_dir)
    outputs: list[Path] = []

    for dataset in datasets:
        dataset_models = tuple(models) if models is not None else _discover_models(root, dataset)
        reference_metrics: dict[str, float | int] = {}
        translation_metrics_by_column: dict[str, dict[str, float | int]] = {}

        for model in dataset_models:
            for experiment_label, (experiment, ablation) in _EXPERIMENTS.items():
                column = f"{model} | {experiment_label} translation score"
                loaded_metrics = _load_metrics(
                    _metric_path(root, experiment, ablation, dataset, model, split)
                )
                translation_metrics: dict[str, float | int] = {}
                for metric_key, value in loaded_metrics.items():
                    if not _is_thesis_facing_metric(metric_key):
                        continue
                    comparison_key, score_role = _comparison_metric_key(metric_key)
                    if score_role == "reference":
                        # The reference set is shared by every condition for a language and split.
                        reference_metrics.setdefault(comparison_key, value)
                    else:
                        translation_metrics[comparison_key] = value
                # Keep planned conditions visible when their metrics are still pending.
                if translation_metrics or _is_planned_experiment(
                    dataset, split, model, experiment_label
                ):
                    translation_metrics_by_column[column] = translation_metrics

        metric_keys = sorted(
            {key for metrics in translation_metrics_by_column.values() for key in metrics}
            | set(reference_metrics)
        )
        rows: list[dict[str, Any]] = []
        for metric_key in metric_keys:
            group, metric = _metric_labels(metric_key)
            row: dict[str, Any] = {
                "Metric group": group,
                "Metric": metric,
                "Direction": _interpretation(metric_key),
                # These metrics already compare two texts, so no separate reference score applies.
                "Reference score": reference_metrics.get(metric_key),
            }
            for column, metrics in translation_metrics_by_column.items():
                row[column] = metrics.get(metric_key)
            rows.append(row)

        output = destination_dir / f"{dataset}_p0_p5_comparison.csv"
        save_dataframe_csv(pd.DataFrame(rows), output)
        outputs.append(output)

    return tuple(outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one P0-P5 comparison CSV per language.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--datasets", nargs="+", default=["en_ca", "en_eu"])
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, default=Path("results/comparisons"))
    args = parser.parse_args()
    outputs = build_comparison_eval(
        args.results_root,
        tuple(args.datasets),
        tuple(args.models) if args.models else None,
        args.split,
        args.output_dir,
    )
    for output in outputs:
        print(f"Wrote comparison CSV to {output}")


if __name__ == "__main__":
    main()
