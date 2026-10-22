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


def build_search_queries(query: str) -> list[str]:
    """Create focused variants so long claim sentences do not dominate search."""
    cleaned = re.sub(r"\s+", " ", query or "").strip()
    if not cleaned:
        return []
    stop_words = {
        "the", "and", "for", "from", "with", "that", "this", "was", "were",
        "are", "has", "have", "had", "into", "during", "according", "reported",
        "reports", "says", "said", "while", "only", "increased", "decreased",
        "claim", "claims", "city", "independent", "survey",
    }
    tokens = re.findall(r"[A-Za-z][A-Za-z'-]{2,}|\d+(?:\.\d+)?%?", cleaned)
    keywords = []
    for token in tokens:
        if token.lower() not in stop_words and token.lower() not in {x.lower() for x in keywords}:
            keywords.append(token)
    variants = [cleaned]
    # Keep quantities, years and distinctive nouns together where possible.
    numbers = [token for token in tokens if re.fullmatch(r"\d{4}|\d+(?:\.\d+)?%?", token)]
    focused = numbers[:3] + [token for token in keywords if token not in numbers][:5]
    if len(focused) >= 2:
        variants.append(" ".join(focused))
    distinctive = [token for token in keywords if token not in numbers][:6]
    if len(distinctive) >= 2:
        variants.append(" ".join(distinctive))
    unique = []
    for variant in variants:
        if variant and variant.lower() not in {item.lower() for item in unique}:
            unique.append(variant)
    return unique[:3]


def search_live_news(query: str, limit: int = 10, timeout: int = 8) -> list[dict[str, str]]:
    """Discover candidate headlines with multi-query RSS search and GDELT fallback.

    Search results remain candidates for human review, not verified evidence.
    """
    queries = build_search_queries(query)
    if not queries:
        raise ValueError("Enter a claim before searching.")
    if limit < 1:
        raise ValueError("limit must be at least 1.")

    provider_errors: list[str] = []
    rss_records: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    # Run a small bounded set of query variants and merge unique results.
    for search_query in queries:
        try:
            found = _search_google_news_rss(search_query, max(1, limit // len(queries)), timeout)
            for record in found:
                url = record.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    rss_records.append(record)
                if len(rss_records) >= limit:
                    break
        except (requests.RequestException, ValueError) as exc:
            provider_errors.append(
                f"Google News RSS ({search_query[:70]}): {type(exc).__name__}: {str(exc)[:120]}"
            )
        if len(rss_records) >= limit:
            break
    if rss_records:
        return rss_records[:limit]

    # Only hit GDELT if RSS yielded no usable candidates.
    try:
        response = _request_gdelt(queries[0], limit, timeout)
        try:
            payload = response.json()
        except ValueError as exc:
            provider_errors.append(
                f"GDELT: non-JSON response ({type(exc).__name__}: {str(exc)[:100]})."
            )
        else:
            records = parse_gdelt_articles(payload)
            if records:
                return records[:limit]
            provider_errors.append("GDELT: no usable article records.")
    except (requests.RequestException, ValueError, RuntimeError) as exc:
        provider_errors.append(
            f"GDELT: {type(exc).__name__}: {str(exc)[:180]}"
        )

    raise ValueError(
        "Live news providers failed. " + " | ".join(provider_errors)
    )
