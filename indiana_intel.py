"""
indiana_intel.py

Indiana developments lead-finder for Tynan / Heli AI site.

Responsibilities:
- Take a natural-language question (e.g. "new warehouses in Boone County in the last 90 days")
- Hit NewsAPI.org for Indiana industrial / logistics / manufacturing projects
- Normalize results into simple dicts
- Provide markdown rendering for the chat UI and AI prompt

Note: You must supply NEWSAPI_KEY as an environment variable for live calls.
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
        format="%(levelname)s:%(name)s:%(message)s",
    )

# ---------------------------------------------------------------------------
# Config: provider + defaults
# ---------------------------------------------------------------------------
NEWSAPI_ENDPOINT = os.environ.get(
    "NEWSAPI_ENDPOINT",
    "https://newsapi.org/v2/everything",
)
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY") or os.environ.get("NEWS_API_KEY")

# How far back to look if the caller doesn’t specify
DEFAULT_DAYS = 60

# Base Indiana-focused keywords to bias toward forklift-relevant projects
BASE_TERMS = (
    "\"warehouse\" OR \"distribution center\" OR logistics OR "
    "manufacturing OR plant OR factory OR industrial OR fulfillment"
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
    m_city = re.search(r"\b(?:in|around|near)\s+([A-Za-z\s]+?)(?:,|\?|\.|$)", text)
    city = None
    if m_city:
        raw = m_city.group(1).strip()
        # strip trailing "Indiana"/"IN"
        raw = re.sub(r"\b(Indiana|IN)\b\.?", "", raw, flags=re.I).strip()
        if raw:
            city = raw

    return (city, county)


def _build_query(user_q: str, city: Optional[str], county: Optional[str]) -> str:
    """
    Build a NewsAPI-friendly query anchored on Indiana industrial keywords,
    with optional city / county bias and the raw user text as a soft signal.
    """
    parts: List[str] = []

    # Always anchor to Indiana and industrial-ish terms
    base = f"Indiana AND ({BASE_TERMS})"
    parts.append(base)

    # Add county / city hints if present
    if county:
        parts.append(f"AND \"{county}\"")
    if city:
        parts.append(f"AND \"{city}\"")

    # Add raw user text as a soft signal (wrapped to avoid breaking syntax)
    cleaned = user_q.strip()
    if cleaned:
        parts.append(f"AND ({cleaned})")

    query = " ".join(parts)
    log.info("NewsAPI query: %s", query)
    return query


def _newsapi_search(query: str, days: int) -> List[Dict[str, Any]]:
    """
    Thin wrapper around NewsAPI 'everything' endpoint.
    Returns a list of raw NewsAPI-style result dicts.
    """
    if not NEWSAPI_KEY:
        log.warning("NEWSAPI_KEY not set; returning empty result list.")
        return []

    from_date = (datetime.utcnow() - timedelta(days=max(1, days))).strftime("%Y-%m-%d")

    params = {
        "q": query,
        "from": from_date,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 20,
    }
    headers = {"X-Api-Key": NEWSAPI_KEY}

    try:
        resp = requests.get(
            NEWSAPI_ENDPOINT,
            params=params,
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.warning("NewsAPI request failed: %s", e)
        return []

    articles = data.get("articles", []) or []
    out: List[Dict[str, Any]] = []

    for a in articles:
        title = a.get("title") or ""
        desc = a.get("description") or ""
        url = a.get("url") or ""
        src = a.get("source") or {}
        provider = src.get("name") or ""
        date_published = a.get("publishedAt")

        dt: Optional[datetime] = None
        if date_published:
            try:
                dt = datetime.fromisoformat(date_published.replace("Z", "+00:00"))
            except Exception:
                dt = None

        out.append(
            {
                "title": title,
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
    - Builds a NewsAPI query anchored on Indiana industrial projects.
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

    raw_results = _newsapi_search(query, days=days)
    filtered: List[Dict[str, Any]] = []

    for r in raw_results:
        if not _looks_relevant(r):
            continue

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
        return (
            "_No recent Indiana developments found for that query. "
            "Try widening the date range or adjusting the city/county._"
        )

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

        meta_bits: List[str] = []
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
