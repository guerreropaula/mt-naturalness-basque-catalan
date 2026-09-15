"""Helpers for resolving Hugging Face authentication."""

from __future__ import annotations

import os


def get_hf_token() -> str | None:
    """Return a Hugging Face token from the process environment."""
    token = os.environ.get("HF_TOKEN", "").strip()
    return token or None
