"""
indiana_intel.py

Indiana developments lead-finder for Tynan / Heli AI site.

Responsibilities:
- Take a natural-language question (e.g. "new warehouses in Boone County in the last 90 days")
- Hit a web/news search API (e.g. Bing News) for Indiana industrial / logistics / manufacturing projects
- Normalize results into simple dicts
- Provide markdown rendering for the chat UI and AI prompt

Note: You must supply a BING_NEWS_API_KEY (or change the provider logic) for live calls.
"""

from __future__ import annotations

import os
import re
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests

log = logging.getLogger("indiana_intel")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s"
    )

# ---------------------------------------------------------------------------
# Config: provider + defaults
# ---------------------------------------------------------------------------
BING_NEWS_ENDPOINT = os.environ.get(
    "BING_NEWS_ENDPOINT",
    "https://api.bing.microsoft.com/v7.0/news/search",
)
BING_NEWS_API_KEY = os.environ.get("BING_NEWS_API_KEY")

DEFAULT_DAYS = 60

# Simple keywords to bias toward forklift-relevant projects
BASE_KEYWORDS = (
    "Indiana (warehouse OR distribution center OR logistics OR "
    "manufacturing OR plant OR factory OR industrial OR fulfillment)"
)

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _lower(s: Any) -> str:
    return str(s or "").lower()


def _extract_geo_hint(q: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Very light-weight extractor:
    - Finds 'X County' → county='X County'
    - Tries to grab a city name after 'in ' if it looks like 'Greenwood' or 'Greenwood, IN'
    Returns (city, county).
    """
    if not q:
        return (None, None)
    text = q.strip()

    # County: e.g. "in Boone County", "Boone County leads", etc.
    m_county = re.search(r"\b([A-Za-z]+)\s+County\b", text)
    county = None
    if m_county:
        county = f"{m_county.group(1).strip()} County"

    # City: look for "in Greenwood", "around Greenwood, IN", etc.
    # This is intentionally simple but works for most phrasing.
    m_city = re.search(r"\b(?:in|around|near)\s+([A-Za-z\s]+?)(?:,|\?|\.|$)", text)
    city = None
    if m_city:
        # clean trailing words like "Indiana" / "IN"
        raw = m_city.group(1).strip()
        raw = re.sub(r"\b(Indiana|IN)\b\.?", "", raw, flags=re.I).strip()
        if raw:
            city = raw

    return (city, county)


def _build_query(user_q: str, city: Optional[str], county: Optional[str]) -> str:
    """
    Build a Bing-friendly query that always anchors on Indiana industrial keywords,
    with optional city / county bias.
    """
    parts = [BASE_KEYWORDS]

    if county:
        parts.append(f'"{county}"')
    if city:
        parts.append(f'"{city}"')

    # Add raw user text as a soft signal
    cleaned = user_q.strip()
    if cleaned:
        parts.append(f"({cleaned})")

    return " ".join(parts)


def _bing_news_search(
    query: str, days: int
) -> List[Dict[str, Any]]:
    """
    Thin wrapper around Bing News Search.
    Returns a list of raw results.
    """
    if not BING_NEWS_API_KEY:
        log.warning("BING_NEWS_API_KEY not set; returning empty result list.")
        return []

    params = {
        "q": query,
        "mkt": "en-US",
        "count": 25,
        "sortBy": "Date",
    }

    headers = {"Ocp-Apim-Subscription-Key": BING_NEWS_API_KEY}
    try:
        resp = requests.get(BING_NEWS_ENDPOINT, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.warning("Bing News request failed: %s", e)
        return []

    articles = data.get("value", []) or []

    # Filter by date window if possible
    since = datetime.utcnow() - timedelta(days=max(1, days))
    out: List[Dict[str, Any]] = []
    for a in articles:
        name = a.get("name") or ""
        desc = a.get("description") or ""
        url = a.get("url") or ""
        provider = ", ".join(
            p.get("name", "") for p in a.get("provider", []) if isinstance(p, dict)
        ).strip()
        date_published = a.get("datePublished")

        dt: Optional[datetime] = None
        if date_published:
            try:
                dt = datetime.fromisoformat(date_published.replace("Z", "+00:00"))
            except Exception:
                dt = None

        if dt and dt < since:
            continue

        out.append(
            {
                "title": name,
                "snippet": desc,
                "url": url,
                "provider": provider,
                "date": dt.isoformat() if dt else None,
            }
        )

    return out


def _looks_relevant(item: Dict[str, Any]) -> bool:
    """
    Quick relevance filter: look for industrial / facility terms in title/snippet.
    """
    text = _lower(item.get("title", "") + " " + item.get("snippet", ""))
    if not text:
        return False

    hits = 0
    for kw in (
        "warehouse",
        "distribution center",
        "logistics",
        "fulfillment",
        "industrial park",
        "industrial",
        "plant",
        "factory",
        "manufacturing",
        "facility",
        "expansion",
        "distribution hub",
        "cold storage",
    ):
        if kw in text:
            hits += 1

    return hits > 0


def search_indiana_developments(
    user_q: str,
    days: int = DEFAULT_DAYS,
) -> List[Dict[str, Any]]:
    """
    Main entrypoint.

    - Extracts simple city/county hints from the user question.
    - Builds a Bing News query anchored on Indiana industrial projects.
    - Returns a list of normalized dicts:
      {
        "title": str,
        "snippet": str,
        "url": str,
        "provider": str,
        "date": "YYYY-MM-DD" or None,
        "city_hint": Optional[str],
        "county_hint": Optional[str],
      }
    """
    city, county = _extract_geo_hint(user_q)
    query = _build_query(user_q, city, county)

    raw_results = _bing_news_search(query, days=days)
    filtered: List[Dict[str, Any]] = []

    for r in raw_results:
        if not _looks_relevant(r):
            continue

        # You could add harder city/county checks here if you want,
        # but for now we keep everything Bing thinks is relevant.
        r["city_hint"] = city
        r["county_hint"] = county
        filtered.append(r)

    # Sort by date desc (newest first)
    def _dt(item: Dict[str, Any]) -> float:
        d = item.get("date")
        if not d:
            return 0.0
        try:
            return datetime.fromisoformat(d).timestamp()
        except Exception:
            return 0.0

    filtered.sort(key=_dt, reverse=True)
    return filtered


def render_developments_markdown(items: List[Dict[str, Any]]) -> str:
    """
    Compact markdown summary for chat UI or as context into GPT.
    """
    if not items:
        return "_No recent Indiana developments found for that query. Try widening the date range or adjusting the city/county._"

    lines: List[str] = ["**Recent Indiana Developments (lead candidates):**"]
    for i, item in enumerate(items[:15], start=1):
        title = item.get("title") or "Untitled"
        snippet = (item.get("snippet") or "").strip()
        url = item.get("url") or ""
        provider = item.get("provider") or ""
        date = item.get("date") or ""
        if date:
            try:
                dt = datetime.fromisoformat(date)
                date = dt.strftime("%Y-%m-%d")
            except Exception:
                pass

        meta_bits = []
        if provider:
            meta_bits.append(provider)
        if date:
            meta_bits.append(date)
        meta = " • ".join(meta_bits)

        lines.append(f"{i}. **{title}**")
        if meta:
            lines.append(f"   - _{meta}_")
        if snippet:
            lines.append(f"   - {snippet}")
        if url:
            lines.append(f"   - {url}")

    return "\n".join(lines)


__all__ = [
    "search_indiana_developments",
    "render_developments_markdown",
]
