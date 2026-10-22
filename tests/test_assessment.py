import json

import pytest

from src.verityai.retrieval import load_evidence


def test_loads_valid_jsonl(tmp_path):
    path = tmp_path / "evidence.jsonl"
    path.write_text(
        json.dumps({"id": "1", "title": "Example", "text": "Passage"}) + "\n",
        encoding="utf-8",
    )
    assert load_evidence(path)[0]["id"] == "1"


def test_rejects_missing_required_field(tmp_path):
    path = tmp_path / "evidence.jsonl"
    path.write_text(json.dumps({"id": "1", "title": "Example"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="text"):
        load_evidence(path)


def test_rejects_empty_corpus(tmp_path):
    path = tmp_path / "evidence.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_evidence(path)
