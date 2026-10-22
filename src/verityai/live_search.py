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


def search_live_news(query: str, limit: int = 10, timeout: int = 20) -> list[dict[str, str]]:
    """Search GDELT and return normalized article candidates."""
    query = query.strip()
    if not query:
        raise ValueError("Enter a claim before searching.")
    if limit < 1:
        raise ValueError("limit must be at least 1.")
    response = requests.get(
        GDELT_URL,
        params={
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": min(limit, 25),
            "sort": "HybridRel",
        },
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError("GDELT returned a non-JSON response.") from exc
    return parse_gdelt_articles(payload)
