"""
Shared base for any Hugging Face sequence-classification model.
Subclasses set `model_id` and `positive_labels` (which output labels mean "injection").
Runs on GPU if torch sees one, else CPU (fine for a few hundred rows).
"""
from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class HFClassifierPredictor:
    name = "hf-classifier"
    model_id = ""
    positive_labels: set[str] = set()   # label names that mean injection
    max_length = 512
    batch_size = 16

    def __init__(self, model_id: str | None = None, threshold: float = 0.5):
        self.model_id = model_id or self.model_id
        self.threshold = threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id).to(self.device).eval()
        id2label = self.model.config.id2label
        self.pos_ids = [i for i, lbl in id2label.items() if lbl.upper() in {p.upper() for p in self.positive_labels}]
        if not self.pos_ids:
            raise ValueError(f"{self.model_id}: none of {self.positive_labels} found in labels {id2label}")

    @torch.no_grad()
    def scores(self, texts: list[str]) -> list[float]:
        """Probability of 'injection' for each text."""
        out: list[float] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            enc = self.tok(batch, truncation=True, max_length=self.max_length,
                           padding=True, return_tensors="pt").to(self.device)
            probs = self.model(**enc).logits.softmax(-1)
            out.extend(probs[:, self.pos_ids].sum(-1).tolist())
        return out

    def predict(self, texts: list[str]) -> list[int]:
        return [int(s >= self.threshold) for s in self.scores(texts)]
