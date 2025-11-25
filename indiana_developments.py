# indiana_developments.py
"""
Indiana developments helper using NewsAPI.org instead of Bing.

- search_indiana_developments(user_q, days=90) -> list[dict]
- render_developments_markdown(items) -> markdown string

Expected by heli_backup_ai.py in the /api/chat 'indiana_developments' mode.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List

import httpx  # make sure 'httpx' is in requirements.txt

import logging

log = logging.getLogger("indiana_intel")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

NEWS_API_KEY = os.getenv("NEWS_API_KEY")  # <-- set this in Render


def _build_query(user_q: str) -> str:
    """
    Take the user's question and build a NewsAPI search query that
    is biased toward Indiana warehouse / industrial developments.
    """
    base_terms = '("warehouse" OR "distribution center" OR "logistics" OR "industrial")'
    geo_terms = '(Indiana OR "IN" OR "Boone County" OR "Hendricks County")'

    q = (user_q or "").strip()
    if not q:
        return f"{base_terms} AND {geo_terms}"

    # Ensure Indiana context is in there somewhere
    q_lower = q.lower()
    if "indiana" not in q_lower and "boone" not in q_lower and "hendricks" not in q_lower:
        q = f"{q} Indiana"

    # Combine with our base terms
    return f"{q} AND {base_terms}"


def search_indiana_developments(user_q: str, days: int = 90) -> List[Dict[str, Any]]:
    """
    Call NewsAPI /v2/everything and return a simplified list of articles.
    """
    if not NEWS_API_KEY:
        log.warning("NEWS_API_KEY not set; returning empty result list.")
        return []

    q = _build_query(user_q)
    from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": q,
        "from": from_date,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 20,
        "apiKey": NEWS_API_KEY,
    }

    log.info("NewsAPI query: %s", q)

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.warning("NewsAPI request failed: %s", e)
        return []

    articles = data.get("articles") or []
    items: List[Dict[str, Any]] = []

    for art in articles:
        try:
            title = (art.get("title") or "").strip()
            if not title:
                continue
            desc = (art.get("description") or "").strip()
            url = art.get("url") or ""
            src = (art.get("source") or {}).get("name") or ""
            published = art.get("publishedAt") or ""
            # trim timestamp to date if present
            if "T" in published:
                published = published.split("T", 1)[0]

            items.append(
                {
                    "title": title,
                    "summary": desc,
                    "url": url,
                    "source": src,
                    "published": published,
                }
            )
        except Exception as e:
            log.debug("Skipping article row due to error: %s", e)
            continue

    return items


def render_developments_markdown(items: List[Dict[str, Any]]) -> str:
    """
    Turn the article list into markdown the AI can reason over.
    """
    if not items:
        return "No recent Indiana warehouse / distribution developments were found."

    lines: List[str] = []
    lines.append("Recent Indiana warehouse / logistics developments:")
    for it in items[:20]:
        title = it.get("title", "")
        src = it.get("source", "")
        published = it.get("published", "")
        url = it.get("url", "")
        summary = it.get("summary", "")

        header = f"- **{title}**"
        meta_bits = []
        if src:
            meta_bits.append(src)
        if published:
            meta_bits.append(published)
        if meta_bits:
            header += f" ({', '.join(meta_bits)})"
        if url:
            header += f" — {url}"

        lines.append(header)
        if summary:
            lines.append(f"  - Summary: {summary}")

    return "\n".join(lines)
