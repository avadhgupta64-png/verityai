"""GDELT-powered candidate news discovery.

Search results are candidates for human review, not verified evidence.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any
from urllib.parse import urlparse

import requests

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
USER_AGENT = "VerityAI/1.0 (research prototype)"


def _valid_http_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _format_seen_date(value: Any) -> str:
    """Format GDELT's seen/indexed timestamp; never label it publication date."""
    raw = str(value or "")
    try:
        return datetime.strptime(raw[:8], "%Y%m%d").date().isoformat()
    except (TypeError, ValueError):
        return "Date not supplied"


def parse_gdelt_articles(payload: dict[str, Any]) -> list[dict[str, str]]:
    """Normalize GDELT article results and discard malformed/untraceable entries."""
    if not isinstance(payload, dict):
        raise ValueError("GDELT response must be a JSON object.")
    articles = payload.get("articles", [])
    if not isinstance(articles, list):
        raise ValueError("GDELT response has an invalid 'articles' field.")
    records: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for article in articles:
        if not isinstance(article, dict):
            continue
        title = str(article.get("title") or "").strip()
        url = str(article.get("url") or "").strip()
        if not title or not _valid_http_url(url) or url in seen_urls:
            continue
        seen_urls.add(url)
        domain = str(article.get("domain") or urlparse(url).netloc).strip()
        seen_date = _format_seen_date(article.get("seendate"))
        records.append({
            "id": f"gdelt-{len(records)}",
            "title": title,
            "publisher": domain or "Publisher not supplied",
            "url": url,
            "date": seen_date,
            "date_label": "GDELT indexed/seen date (not necessarily publication date)",
            "text": f"{title}. Publisher/domain: {domain}",
        })
    return records


def _request_gdelt(query: str, limit: int, timeout: int) -> requests.Response:
    """Call GDELT with bounded retries for transient gateway/rate-limit errors."""
    last_response = None
    last_error = None
    for attempt in range(3):
        try:
            response = requests.get(
                GDELT_URL,
                params={
                    "query": query,
                    "mode": "ArtList",
                    "format": "json",
                    "maxrecords": min(limit, 25),
                    "sort": "HybridRel",
                },
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                timeout=timeout,
            )
            last_response = response
            # Retry temporary server errors and rate limiting.
            if response.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                import time
                time.sleep(0.5 * (attempt + 1))
                continue
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt < 2:
                import time
                time.sleep(0.5 * (attempt + 1))
                continue
            raise
    if last_response is not None:
        last_response.raise_for_status()
        return last_response
    if last_error:
        raise last_error
    raise RuntimeError("GDELT request failed without a response.")


def search_live_news(query: str, limit: int = 10, timeout: int = 20) -> list[dict[str, str]]:
    """Search GDELT and return normalized article candidates.

    GDELT occasionally responds with an HTML/plain-text error page despite an
    HTTP 200. Retry once with a compact keyword query before reporting failure.
    """
    query = query.strip()
    if not query:
        raise ValueError("Enter a claim before searching.")
    if limit < 1:
        raise ValueError("limit must be at least 1.")

    # Keep the original sentence for the first attempt, then reduce it to a
    # compact set of words if GDELT returns a non-JSON response.
    response = _request_gdelt(query, limit, timeout)
    try:
        payload = response.json()
    except ValueError as first_exc:
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]{2,}", query)
        stop_words = {
            "the", "and", "for", "from", "with", "that", "this", "says",
            "said", "during", "only", "report", "claims", "claim",
        }
        compact_words = []
        for word in words:
            if word.lower() not in stop_words and word.lower() not in {
                value.lower() for value in compact_words
            }:
                compact_words.append(word)
        compact_query = " ".join(compact_words[:7])
        if not compact_query or compact_query.casefold() == query.casefold():
            compact_query = " ".join(words[:6]) or query[:80]
        retry_response = _request_gdelt(compact_query, limit, timeout)
        try:
            payload = retry_response.json()
        except ValueError as second_exc:
            content_type = retry_response.headers.get("Content-Type", "unknown")
            preview = " ".join((retry_response.text or "").split())[:180]
            raise ValueError(
                "GDELT returned a non-JSON response after retrying with a shorter query "
                f"(HTTP {retry_response.status_code}, Content-Type: {content_type}). "
                f"Response preview: {preview or '[empty response]'}. "
                "The live-news service may be temporarily unavailable; try again or switch to the local evidence corpus."
            ) from second_exc
    return parse_gdelt_articles(payload)
