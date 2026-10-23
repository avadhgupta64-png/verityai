"""Claim breakdown and conservative evidence assessment.

Coverage is a lexical relevance signal, never a probability of truth.
"""
from __future__ import annotations

import re
from typing import Any

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from",
    "has", "have", "in", "is", "it", "of", "on", "or", "that", "the", "their",
    "this", "to", "was", "were", "will", "with", "says", "said", "according",
    "reported", "reports", "claim", "claims", "claimed", "only", "during",
}


def split_claim_into_facts(claim: str) -> list[str]:
    """Split a paragraph into reviewable sentence-level statements."""
    cleaned = re.sub(r"\\s+", " ", (claim or "")).strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\\s+(?=[A-Z0-9“\"'])", cleaned)
    return [part.strip(" \\t\\n") for part in parts if part.strip()]


def _terms(text: str) -> set[str]:
    return {
        term.lower()
        for term in re.findall(r"[A-Za-z][A-Za-z'-]{2,}|\\d+(?:\\.\\d+)?", text)
        if term.lower() not in STOP_WORDS
    }


def assess_facts(facts: list[str], evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find candidate sources and describe evidence availability, not truth."""
    assessed: list[dict[str, Any]] = []
    for fact in facts:
        fact_terms = _terms(fact)
        candidates = []
        for item in evidence:
            evidence_terms = _terms(f"{item.get('title', '')} {item.get('text', '')}")
            overlap_terms = fact_terms & evidence_terms
            coverage = len(overlap_terms) / max(len(fact_terms), 1)
            candidates.append((coverage, len(overlap_terms), item, overlap_terms))
        candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)
        best_coverage, overlap, best, shared = candidates[0] if candidates else (0.0, 0, {}, set())

        # This is relevance only. It is not entailment or contradiction detection.
        relevant = bool(candidates and best_coverage >= 0.20 and overlap >= 2)
        strong_match = relevant and best_coverage >= 0.55 and overlap >= 3
        if not relevant:
            status = "Insufficient evidence found"
            reason = (
                "No sufficiently relevant candidate passage was found. This is a search/retrieval gap, "
                "not evidence that the statement is false."
            )
        elif strong_match:
            status = "Relevant source found — review needed"
            reason = (
                "A candidate passage shares several key terms with the statement. This does not show "
                "that the source supports or contradicts it; open the original article and check its "
                "full text, date, context, and source attribution."
            )
        else:
            status = "Possible source found — review needed"
            reason = (
                "Some related terms were found, but the match is weak. Inspect the original source "
                "before deciding whether it is relevant; word overlap is not verification."
            )

        assessed.append({
            "fact": fact,
            "status": status,
            "coverage_score": round(best_coverage if relevant else 0.0, 3),
            "matched_terms": sorted(shared) if relevant else [],
            "reason": reason,
            "source": best if relevant else None,
        })
    return assessed


def find_numeric_discrepancies(facts: list[str]) -> list[dict[str, Any]]:
    """Flag differing percentages in statements that discuss the same broad topic.

    This is a review prompt only; it cannot determine whether the figures conflict.
    """
    results: list[dict[str, Any]] = []
    percent_pattern = re.compile(r"(\\d+(?:\\.\\d+)?)\\s*%")
    for left_index, left in enumerate(facts):
        left_values = percent_pattern.findall(left)
        if not left_values:
            continue
        left_terms = _terms(left) - {"increased", "decreased", "growth", "change", "year"}
        for right_index in range(left_index + 1, len(facts)):
            right = facts[right_index]
            right_values = percent_pattern.findall(right)
            if not right_values or set(left_values) == set(right_values):
                continue
            right_terms = _terms(right) - {"increased", "decreased", "growth", "change", "year"}
            shared_topic = left_terms & right_terms
            if len(shared_topic) >= 2:
                results.append({
                    "statement_a": left,
                    "statement_b": right,
                    "figures": [f"{value}%" for value in left_values + right_values],
                    "shared_topic_terms": sorted(shared_topic),
                    "note": "These statements contain different percentage figures and share topic terms. Check whether they describe the same metric, area, period, and methodology before calling them contradictory.",
                })
    return results
