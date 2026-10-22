"""Claim breakdown and conservative evidence assessment.

The scores here describe retrieval coverage and textual signals. They are not
probabilities that a claim is true.
"""
from __future__ import annotations

import re
from typing import Any

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from",
    "has", "have", "in", "is", "it", "of", "on", "or", "that", "the", "their",
    "this", "to", "was", "were", "will", "with", "says", "said", "according",
    "reported", "reports", "claim", "claims", "claimed",
}


def split_claim_into_facts(claim: str) -> list[str]:
    """Split a paragraph into reviewable sentence-level factual statements."""
    cleaned = re.sub(r"\s+", " ", (claim or "")).strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9“\"'])", cleaned)
    facts = [part.strip(" \t\n") for part in parts if part.strip()]
    return facts[:12]


def _terms(text: str) -> set[str]:
    return {
        term.lower()
        for term in re.findall(r"[A-Za-z][A-Za-z'-]{2,}|\d+(?:\.\d+)?", text)
        if term.lower() not in STOP_WORDS
    }


def assess_facts(facts: list[str], evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare each fact with retrieved passages using transparent lexical signals.

    This intentionally abstains from a Verified verdict: semantic retrieval and
    word overlap alone cannot establish truth or source authority.
    """
    assessed: list[dict[str, Any]] = []
    for fact in facts:
        fact_terms = _terms(fact)
        candidates = []
        for item in evidence:
            text = f"{item.get('title', '')} {item.get('text', '')}"
            evidence_terms = _terms(text)
            overlap = len(fact_terms & evidence_terms)
            coverage = overlap / max(len(fact_terms), 1)
            candidates.append((coverage, overlap, item))
        candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)
        best_coverage, overlap, best = candidates[0] if candidates else (0.0, 0, {})
        enough = best_coverage >= 0.55 and overlap >= 3
        conflicting = False
        if enough:
            fact_lower = fact.lower()
            evidence_lower = f"{best.get('title', '')} {best.get('text', '')}".lower()
            negations = (" not ", " never ", " no ", " false ", " denied ", " disproved ")
            fact_negated = any(token in f" {fact_lower} " for token in negations)
            evidence_negated = any(token in f" {evidence_lower} " for token in negations)
            conflicting = fact_negated != evidence_negated
        if not candidates or best_coverage < 0.20 or overlap < 2:
            status = "Unverified"
            reason = "The retrieved material has too little meaningful term overlap to assess this statement."
        elif conflicting:
            status = "Potentially Misleading"
            reason = "The closest passage has overlapping terms but a possible negation mismatch. Read the original context; this is a warning, not a verdict."
        elif enough:
            status = "Unverified"
            reason = "A passage shares several key terms with this statement, but text overlap does not prove support. Source authority and context still need human review."
        else:
            status = "Unverified"
            reason = "Some related terms were found, but the available match is too weak to establish support or contradiction."
        assessed.append({
            "fact": fact,
            "status": status,
            "coverage_score": round(best_coverage, 3),
            "matched_terms": sorted(fact_terms & _terms(f"{best.get('title', '')} {best.get('text', '')}")),
            "reason": reason,
            "source": best or None,
        })
    return assessed
