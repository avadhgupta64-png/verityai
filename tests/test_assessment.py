import json

import pytest

from src.verityai.assessment import assess_facts, find_numeric_discrepancies, split_claim_into_facts
from src.verityai.classifier import classify_article
from src.verityai.live_search import build_search_queries, parse_gdelt_articles
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


def test_live_search_builds_focused_query_variants():
    queries = build_search_queries(
        "A city planted 500 trees in 2025. Tree cover increased by 20%."
    )
    assert len(queries) >= 2
    assert len(queries) <= 3
    assert queries[0].startswith("A city planted")
    assert any("500" in query and "2025" in query for query in queries)
    assert build_search_queries("   ") == []


def test_gdelt_parser_filters_invalid_and_duplicate_urls():
    records = parse_gdelt_articles({"articles": [
        {"title": "A headline", "url": "https://example.com/a", "domain": "example.com", "seendate": "20261022093000"},
        {"title": "Bad URL", "url": "javascript:alert(1)"},
        {"title": "Duplicate", "url": "https://example.com/a"},
        {"title": "", "url": "https://example.com/empty"},
    ]})
    assert len(records) == 1
    assert records[0]["date"] == "2026-10-22"
    assert "not necessarily publication date" in records[0]["date_label"]


def test_gdelt_parser_handles_empty_and_invalid_payload():
    assert parse_gdelt_articles({"articles": []}) == []
    with pytest.raises(ValueError, match="JSON object"):
        parse_gdelt_articles([])


def test_splits_paragraph_into_atomic_sentence_statements():
    assert split_claim_into_facts("The council met on Monday. It approved two clinics.") == [
        "The council met on Monday.", "It approved two clinics."
    ]


def test_empty_claim_has_no_facts():
    assert split_claim_into_facts("   ") == []


def test_assessment_abstains_when_no_evidence_exists():
    result = assess_facts(["The city opened three clinics."], [])
    assert result[0]["status"] == "Unverified"
    assert result[0]["source"] is None


def test_assessment_does_not_attach_irrelevant_source():
    result = assess_facts(
        ["The council planted 500 trees in 2025."],
        [{"id": "1", "title": "School garden report", "text": "The garden contains twelve planter boxes."}],
    )
    assert result[0]["status"] == "Unverified"
    assert result[0]["source"] is None
    assert result[0]["coverage_score"] == 0
    assert "No sufficiently relevant passage" in result[0]["reason"]


def test_numeric_discrepancy_is_a_review_prompt_not_a_verdict():
    facts = [
        "The report claims that tree cover increased by 20% during the year.",
        "An independent environmental survey says tree cover increased by only 5%.",
    ]
    results = find_numeric_discrepancies(facts)
    assert len(results) == 1
    assert results[0]["figures"] == ["20%", "5%"]
    assert "Check whether" in results[0]["note"]


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
    assert classify_article(FakeModel(), "Headline", "body") == {
        "label": "Real", "score": pytest.approx(0.8)
    }
