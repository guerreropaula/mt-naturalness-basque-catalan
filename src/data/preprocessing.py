"""Minimal ordered preprocessing for the EN-CA and EN-EU training corpora."""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.loaders import load_dataset
from src.data.normalization import normalize_text
from src.utils.config import get_dataset_entry, load_preprocessing_config
from src.utils.io import save_dataframe_jsonl, save_json

logger = logging.getLogger(__name__)


class LanguageIdError(RuntimeError):
    """Raised when the configured fastText language-ID model is unavailable."""


def load_fasttext_model(model_path: str):
    try:
        import fasttext
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise LanguageIdError("fasttext-wheel is required for preprocessing.") from exc
    path = Path(model_path)
    if not path.exists():
        raise LanguageIdError(
            f"fastText language-ID model not found: {path}. "
            "Download lid.176.ftz before running preprocessing."
        )
    return fasttext.load_model(str(path))


def predict_language(model: Any, text: str) -> tuple[str, float]:
    labels, scores = model.predict(text.replace("\n", " "), k=1)
    return labels[0].replace("__label__", ""), float(scores[0])


def _domain_filter_active(
    dataset_config: dict[str, Any], preprocessing_config: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    domain_config = dict(preprocessing_config.get("domain_filtering", {}))
    if not bool(domain_config.get("enabled", False)):
        return False, domain_config
    corpus_ids = {str(item) for item in domain_config.get("corpus_ids", [])}
    if corpus_ids and str(dataset_config["corpus_id"]) not in corpus_ids:
        return False, domain_config
    return True, domain_config


def basic_text_rejection_reason(text: str, config: dict[str, Any], side: str) -> str | None:
    if not text:
        return f"{side}_empty"
    if len(text) < int(config["min_chars"]):
        return f"{side}_too_short"
    if len(text) > int(config["max_chars"]):
        return f"{side}_too_long"
    if config.get("require_alphabetic", True) and not any(
        character.isalpha() for character in text
    ):
        return f"{side}_no_alphabetic"
    return None


def length_rejection_reason(source: str, target: str, config: dict[str, Any]) -> str | None:
    source_tokens = max(len(source.split()), 1)
    target_tokens = max(len(target.split()), 1)
    min_tokens = int(config.get("min_tokens", 1))
    max_tokens = int(config.get("max_tokens", 10**9))
    if source_tokens < min_tokens:
        return "source_too_few_tokens"
    if target_tokens < min_tokens:
        return "target_too_few_tokens"
    if source_tokens > max_tokens:
        return "source_too_many_tokens"
    if target_tokens > max_tokens:
        return "target_too_many_tokens"
    token_ratio = target_tokens / source_tokens
    char_ratio = len(target) / max(len(source), 1)
    if not float(config["min_token_ratio"]) <= token_ratio <= float(config["max_token_ratio"]):
        return "token_ratio_out_of_range"
    if not float(config["min_char_ratio"]) <= char_ratio <= float(config["max_char_ratio"]):
        return "char_ratio_out_of_range"
    return None


def _row_id(prefix: str, original_index: int) -> str:
    return f"{prefix}_{original_index + 1:06d}"


def preprocess_loaded_dataframe(
    df: pd.DataFrame,
    dataset_config: dict[str, Any],
    preprocessing_config: dict[str, Any],
    *,
    require_full: bool = True,
) -> pd.DataFrame:
    """Filter rows in order and stop after the configured ordered split size."""
    required_columns = {"original_index", "source", "target", "corpus"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Loaded corpus is missing columns: {sorted(missing)}")

    split_config = preprocessing_config["splits"]
    requested_rows = sum(
        int(split_config.get(key, 0))
        for key in (
            "train_size",
            "test_size",
            "lfp_background_size",
            "evaluation_reserve_size",
        )
    )
    lid_config = preprocessing_config["language_id"]
    language_model = load_fasttext_model(str(lid_config["model_path"]))
    threshold = float(lid_config["confidence_threshold"])
    normalization = preprocessing_config["normalization"]
    text_filtering = preprocessing_config["text_filtering"]
    length_filtering = preprocessing_config["length_filtering"]
    prefix = str(dataset_config.get("id_prefix", dataset_config["corpus_id"]))
    metadata_columns = list(dict(dataset_config.get("metadata_mapping", {})))
    domain_active, domain_config = _domain_filter_active(dataset_config, preprocessing_config)
    domain_column = str(domain_config.get("column", "domain"))
    if domain_active and domain_column not in df.columns:
        raise ValueError(
            f"Domain filtering requires column '{domain_column}' for {dataset_config['corpus_id']}."
        )
    allowed_domains = {str(value) for value in domain_config.get("allowed_values", [])}
    if domain_active and not allowed_domains:
        raise ValueError("Domain filtering is enabled but allowed_values is empty.")

    rows: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    valid_rows = 0
    for row in df.itertuples(index=False):
        original_index = int(getattr(row, "original_index"))
        source = normalize_text(getattr(row, "source"), normalization)
        target = normalize_text(getattr(row, "target"), normalization)
        source = "" if pd.isna(source) else str(source)
        target = "" if pd.isna(target) else str(target)
        metadata = {
            column: getattr(row, column)
            for column in metadata_columns
            if hasattr(row, column)
        }
        row_id = _row_id(prefix, original_index)
        reason = None
        if domain_active and str(metadata.get(domain_column, "")) not in allowed_domains:
            reason = "domain_not_allowed"
        reason = reason or basic_text_rejection_reason(source, text_filtering, "source")
        reason = reason or basic_text_rejection_reason(target, text_filtering, "target")
        if reason is None:
            source_lang, source_score = predict_language(language_model, source)
            target_lang, target_score = predict_language(language_model, target)
            if source_lang != dataset_config["source_lang"] or source_score < threshold:
                reason = "source_language_mismatch"
            elif target_lang != dataset_config["target_lang"] or target_score < threshold:
                reason = "target_language_mismatch"
        if reason is None:
            reason = length_rejection_reason(source, target, length_filtering)

        if reason is None:
            pair = (source, target)
            if pair in seen_pairs:
                reason = "duplicate_pair"
            else:
                seen_pairs.add(pair)
                valid_rows += 1


        rows.append(
            {
                "id": row_id,
                "source": source,
                "target": target,
                "status": "valid" if reason is None else "invalid",
                "reason": reason,
                "corpus": dataset_config["corpus_id"],
                "original_index": original_index,
                **metadata,
            }
        )
        if valid_rows >= requested_rows:
            break


    result = pd.DataFrame(rows)
    result.attrs["summary"] = {
        "input_rows_available": int(len(df)),
        "rows_scanned": int(len(result)),
        "valid_rows": int((result["status"] == "valid").sum()),
        "invalid_rows": int((result["status"] == "invalid").sum()),
        "invalid_reasons": dict(Counter(result.loc[result["status"] == "invalid", "reason"])),
        "requested_rows": requested_rows,
        "split_sizes": {
            "train": int(split_config["train_size"]),
            "test": int(split_config["test_size"]),
            "lfp_background": int(split_config.get("lfp_background_size", 0)),
            "evaluation_reserve": int(split_config.get("evaluation_reserve_size", 0)),
        },
        "domain_filtering": {
            "enabled": domain_active,
            "column": domain_column if domain_active else None,
            "allowed_values": sorted(allowed_domains) if domain_active else [],
            "valid_domain_counts": (
                {
                    str(key): int(value)
                    for key, value in result.loc[result["status"] == "valid", domain_column]
                    .value_counts()
                    .items()
                }
                if domain_active
                else {}
            ),
        },
        "preserves_original_order": True,
        "complete": valid_rows >= requested_rows,
    }
    if require_full and valid_rows < requested_rows:
        raise ValueError(
            "Only "
            f"{valid_rows} valid rows found; {requested_rows} are required for "
            "train/test/LFP background."
        )
    return result

def _ordered_splits(
    processed_df: pd.DataFrame, split_config: dict[str, Any]
) -> dict[str, pd.DataFrame]:
    metadata_columns = [
        column
        for column in ("domain", "type", "corpus_alignment")
        if column in processed_df.columns
    ]
    valid = processed_df.loc[
        processed_df["status"] == "valid",
        ["id", "source", "target", *metadata_columns],
    ].copy()
    train_end = int(split_config["train_size"])
    test_end = train_end + int(split_config["test_size"])
    background_end = test_end + int(split_config.get("lfp_background_size", 0))
    reserve_end = background_end + int(split_config.get("evaluation_reserve_size", 0))
    splits = {
        "train": valid.iloc[:train_end],
        "test": valid.iloc[train_end:test_end],
    }
    if int(split_config.get("lfp_background_size", 0)):
        splits["lfp_background"] = valid.iloc[test_end:background_end]
    if int(split_config.get("evaluation_reserve_size", 0)):
        splits["evaluation_reserve"] = valid.iloc[background_end:reserve_end]
    return splits


def persist_preprocessed_outputs(
    processed_df: pd.DataFrame,
    dataset_key: str,
    preprocessing_config: dict[str, Any],
    *,
    force: bool = False,
) -> dict[str, str]:
    """Write ordered training JSONL splits plus a compact invalid-row audit."""
    output_dir = Path(preprocessing_config["output"]["processed_dir"]) / dataset_key
    paths = {
        "train": output_dir / "train.jsonl",
        "test": output_dir / "test.jsonl",
        "lfp_background": output_dir / "lfp_background.jsonl",
        "evaluation_reserve": output_dir / "evaluation_reserve.jsonl",
        "invalid": output_dir / "invalid.jsonl",
        "summary": output_dir / "summary.json",
    }
    if not int(preprocessing_config["splits"].get("lfp_background_size", 0)):
        paths.pop("lfp_background")
    if not int(preprocessing_config["splits"].get("evaluation_reserve_size", 0)):
        paths.pop("evaluation_reserve")
    if not force:
        existing = [path for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError("Existing preprocessing output: " + ", ".join(map(str, existing)))

    for name, frame in _ordered_splits(processed_df, preprocessing_config["splits"]).items():
        save_dataframe_jsonl(frame, paths[name])
    invalid_columns = ["id", "source", "target", "reason", "corpus", "original_index"]
    invalid_columns.extend(
        column
        for column in ("domain", "type", "corpus_alignment")
        if column in processed_df.columns
    )
    invalid = processed_df.loc[
        processed_df["status"] == "invalid",
        invalid_columns,
    ]
    save_dataframe_jsonl(invalid, paths["invalid"])
    save_json(processed_df.attrs["summary"], paths["summary"])
    return {name: str(path) for name, path in paths.items()}


def preprocess_dataset(
    dataset_key: str,
    dataset_config_path: str | Path = "configs/datasets.yaml",
    preprocessing_config_path: str | Path = "configs/preprocessing.yaml",
    limit: int | None = None,
    require_full: bool = True,
) -> pd.DataFrame:
    config = get_dataset_entry(dataset_key, dataset_config_path)
    preprocessing_config = load_preprocessing_config(preprocessing_config_path)
    loaded = load_dataset(dataset_key, dataset_config_path, limit=limit)
    return preprocess_loaded_dataframe(
        loaded,
        config,
        preprocessing_config,
        require_full=require_full,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create ordered JSONL training splits.")
    parser.add_argument("--dataset", required=True, choices=["en_ca", "en_eu"])
    parser.add_argument("--dataset-config", default="configs/datasets.yaml")
    parser.add_argument("--preprocessing-config", default="configs/preprocessing.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def run_preprocessing_cli(args: argparse.Namespace) -> dict[str, Any]:
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_preprocessing_config(args.preprocessing_config)
    processed = preprocess_dataset(
        args.dataset,
        dataset_config_path=args.dataset_config,
        preprocessing_config_path=args.preprocessing_config,
        limit=args.limit,
        require_full=not (args.dry_run and args.limit is not None),
    )
    result: dict[str, Any] = {"dataset": args.dataset, "summary": processed.attrs["summary"]}
    if not args.dry_run:
        result["written"] = persist_preprocessed_outputs(
            processed, args.dataset, config, force=args.force
        )
    logger.info("Preprocessing summary: %s", result["summary"])
    return result


if __name__ == "__main__":
    run_preprocessing_cli(build_arg_parser().parse_args())
