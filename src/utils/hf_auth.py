"""Resolve Hugging Face authentication from the environment."""

from __future__ import annotations

import os


def get_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN", "").strip()
    return token or None
