"""GDELT-powered candidate news discovery.

Search results are candidates for human review, not verified evidence.
"""
from __future__ import annotations

from datetime import datetime
import re
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus
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


def _search_google_news_rss(query: str, limit: int, timeout: int) -> list[dict[str, str]]:
    """Fallback news discovery using Google News RSS when GDELT is unavailable."""
    url = (
        "https://news.google.com/rss/search?q="
        + quote_plus(query)
        + "&hl=en-IN&gl=IN&ceid=IN:en"
    )
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/rss+xml, application/xml, text/xml"},
        timeout=timeout,
    )
    response.raise_for_status()
    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as exc:
        raise ValueError("Google News RSS returned invalid XML.") from exc

    records: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for item in root.findall("./channel/item"):
        title = (item.findtext("title") or "").strip()
        article_url = (item.findtext("link") or "").strip()
        if not title or not _valid_http_url(article_url) or article_url in seen_urls:
            continue
        seen_urls.add(article_url)
        publisher = (item.findtext("source") or "").strip() or urlparse(article_url).netloc
        pub_date = (item.findtext("pubDate") or "").strip()
        records.append({
            "id": f"google-news-{len(records)}",
            "title": title,
            "publisher": publisher,
            "url": article_url,
            "date": pub_date or "Date not supplied",
            "date_label": "Google News RSS publication date (as supplied)",
            "text": f"{title}. Publisher: {publisher}",
        })
        if len(records) >= limit:
            break
    return records


def search_live_news(query: str, limit: int = 10, timeout: int = 8) -> list[dict[str, str]]:
    """Discover candidate headlines using Google News RSS first, then GDELT.

    GDELT can rate-limit shared cloud-hosted apps. RSS-first avoids repeatedly
    hitting a provider already returning 429. Results are candidate links for
    human review, not verified evidence.
    """
    query = query.strip()
    if not query:
        raise ValueError("Enter a claim before searching.")
    if limit < 1:
        raise ValueError("limit must be at least 1.")

    provider_errors: list[str] = []

    # Try RSS first: GDELT was returning HTTP 429 from the hosted app. Do not
    # burn multiple retry attempts against a rate-limited shared endpoint.
    try:
        rss_records = _search_google_news_rss(query, limit, timeout)
        if rss_records:
            return rss_records
        provider_errors.append("Google News RSS: no usable article records.")
    except (requests.RequestException, ValueError) as exc:
        provider_errors.append(
            f"Google News RSS: {type(exc).__name__}: {str(exc)[:180]}"
        )

    try:
        response = _request_gdelt(query, limit, timeout)
        try:
            payload = response.json()
        except ValueError as exc:
            provider_errors.append(
                f"GDELT: non-JSON response ({type(exc).__name__}: {str(exc)[:100]})."
            )
        else:
            records = parse_gdelt_articles(payload)
            if records:
                return records
            provider_errors.append("GDELT: no usable article records.")
    except (requests.RequestException, ValueError, RuntimeError) as exc:
        provider_errors.append(
            f"GDELT: {type(exc).__name__}: {str(exc)[:180]}"
        )

    raise ValueError(
        "Live news providers failed. " + " | ".join(provider_errors)
    )
