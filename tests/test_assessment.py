import json

import pytest

from src.verityai.classifier import classify_article
from src.verityai.live_search import parse_gdelt_articles
from src.verityai.retrieval import load_evidence


def test_loads_valid_jsonl(tmp_path):
    path = tmp_path / "evidence.jsonl"
    path.write_text(json.dumps({"id": "1", "title": "Example", "text": "Passage"}) + "\n", encoding="utf-8")
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


def test_gdelt_parser_normalizes_metadata_and_filters_bad_urls():
    payload = {"articles": [
        {"title": "A headline", "url": "https://example.com/a", "domain": "example.com", "seendate": "20261022093000"},
        {"title": "Bad URL", "url": "javascript:alert(1)", "domain": "bad"},
        {"title": "Duplicate", "url": "https://example.com/a", "domain": "example.com"},
        {"title": "", "url": "https://example.com/empty"},
    ]}
    records = parse_gdelt_articles(payload)
    assert len(records) == 1
    assert records[0]["title"] == "A headline"
    assert records[0]["date"] == "2026-10-22"
    assert "not necessarily publication date" in records[0]["date_label"]


def test_gdelt_parser_handles_empty_and_invalid_payload():
    assert parse_gdelt_articles({"articles": []}) == []
    with pytest.raises(ValueError, match="JSON object"):
        parse_gdelt_articles([])


def test_classifier_rejects_empty_input():
    class NeverCalled:
        def predict(self, values):
            raise AssertionError("predict should not run for empty input")
    with pytest.raises(ValueError, match="Enter a title"):
        classify_article(NeverCalled(), " ", "\n")


def test_classifier_returns_label_and_predicted_class_score():
    class FakeModel:
        classes_ = ["Fake", "Real"]
        def predict(self, values):
            assert values == ["Headline body"]
            return ["Real"]
        def predict_proba(self, values):
            return [[0.2, 0.8]]
    result = classify_article(FakeModel(), "Headline", "body")
    assert result == {"label": "Real", "score": pytest.approx(0.8)}
