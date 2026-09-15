"""Conservative text normalisation for preprocessing."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd

_WHITESPACE_RE = re.compile(r"\s+")
_ZERO_WIDTH_CHARACTERS = (
    "\u200b",  # zero width space
    "\u200c",  # zero width non-joiner
    "\u200d",  # zero width joiner
    "\ufeff",  # zero width no-break space / BOM
)


def normalize_text(text: Any, config: dict[str, Any]) -> Any:
    """Apply conservative normalisation while preserving linguistic content."""
    if text is None or pd.isna(text):
        return pd.NA

    normalized = str(text)
    unicode_form = config.get("unicode_form")
    if unicode_form:
        normalized = unicodedata.normalize(unicode_form, normalized)
    if config.get("replace_nbsp", False):
        normalized = normalized.replace("\u00a0", " ")
    if config.get("remove_zero_width_spaces", False):
        for character in _ZERO_WIDTH_CHARACTERS:
            normalized = normalized.replace(character, "")
    if config.get("collapse_whitespace", False):
        normalized = _WHITESPACE_RE.sub(" ", normalized)
    if config.get("strip_edges", False):
        normalized = normalized.strip()
    return normalized


def add_normalized_text_columns(
    df: pd.DataFrame,
    normalization_config: dict[str, Any],
    source_raw_column: str = "source_raw",
    target_raw_column: str = "target_raw",
) -> pd.DataFrame:
    """Create `source` and `target` columns from the raw text columns."""
    result = df.copy()
    result["source"] = result[source_raw_column].map(
        lambda value: normalize_text(value, normalization_config)
    )
    result["target"] = result[target_raw_column].map(
        lambda value: normalize_text(value, normalization_config)
    )
    return result
