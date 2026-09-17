"""Target-side HT-likeness scoring for GRPO."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class TargetSideReferenceLikenessScorer:
    """Estimate P(human translation | target text)."""

    def __init__(
        self,
        model_dir: str | Path,
        max_length: int = 256,
        device_name: str | None = None,
    ) -> None:
        self.max_length = int(max_length)
        self.device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score(self, target_texts: list[str], batch_size: int = 64) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(target_texts), batch_size):
            batch = self.tokenizer(
                [str(text) for text in target_texts[start : start + batch_size]],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch = {name: value.to(self.device) for name, value in batch.items()}
            probabilities = torch.softmax(self.model(**batch).logits, dim=-1)[:, 1]
            scores.extend(probabilities.detach().cpu().tolist())
        return scores
