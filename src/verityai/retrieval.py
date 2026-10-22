"""Local evidence retrieval using Hugging Face Transformers and FAISS.

The returned score is a vector similarity for ranking passages, not a truth probability.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_evidence(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL evidence records and validate required fields."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            for field in ("id", "title", "text"):
                if not isinstance(record.get(field), str) or not record[field].strip():
                    raise ValueError(f"Line {line_number}: required field {field!r} is missing.")
            records.append(record)
    if not records:
        raise ValueError("Evidence corpus is empty.")
    return records


class EvidenceRetriever:
    """Embed corpus passages and retrieve the most similar passages."""

    def __init__(self, evidence: list[dict[str, Any]], model_name: str = DEFAULT_MODEL):
        if not evidence:
            raise ValueError("Evidence corpus must contain at least one passage.")
        self.evidence = evidence
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        corpus_vectors = self._embed([self._passage_text(item) for item in evidence])
        self.index = faiss.IndexFlatIP(corpus_vectors.shape[1])
        self.index.add(corpus_vectors)

    @staticmethod
    def _passage_text(item: dict[str, Any]) -> str:
        return f"{item.get('title', '')}. {item.get('text', '')}"

    def _embed(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            output = self.model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).expand(output.size()).float()
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy().astype("float32")

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        vector = self._embed([query])
        scores, indices = self.index.search(vector, min(top_k, len(self.evidence)))
        results: list[dict[str, Any]] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            result = dict(self.evidence[int(index)])
            result["score"] = float(score)
            results.append(result)
        return results
