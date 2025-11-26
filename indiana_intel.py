"""
indiana_intel.py

Indiana developments lead-finder for Tynan / Heli AI site.

Responsibilities:
- Take a natural-language question (e.g. "new warehouses in Boone County in the last 90 days")
- Hit Google Programmable Search (Custom Search JSON API) for Indiana
  industrial / logistics / manufacturing projects
- Normalize results into simple dicts
- Provide markdown rendering for the chat UI and AI prompt

NOTE:
- You MUST set these environment variables in Render:
    GOOGLE_CSE_API_KEY  -> your Google API key
    GOOGLE_CSE_CX       -> your Custom Search Engine ID
"""

from __future__ import annotations

import os
import re
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests

log = logging.getLogger("indiana_intel")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s"
    )

# ---------------------------------------------------------------------------
# Config: Google Programmable Search
# ---------------------------------------------------------------------------
GOOGLE_CSE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"
GOOGLE_CSE_API_KEY = os.environ.get("GOOGLE_CSE_API_KEY")
GOOGLE_CSE_CX = os.environ.get("GOOGLE_CSE_CX")

# This is just a semantic parameter now (Google API doesn't take "days")
DEFAULT_DAYS = 60

# Base keywords to bias toward forklift-relevant projects in Indiana
BASE_KEYWORDS = (
    "Indiana (warehouse OR \"distribution center\" OR logistics OR "
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
    - Tries to grab a city name after 'in ' if it looks like 'Greenwood'
      or 'Greenwood, IN'
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
        # Remove trailing "Indiana" / "IN"
        raw = re.sub(r"\b(Indiana|IN)\b\.?", "", raw, flags=re.I).strip()
        if raw:
            city = raw

    return (city, county)


def _build_query(user_q: str, city: Optional[str], county: Optional[str]) -> str:
    """
    Build a Google-friendly query that anchors on Indiana industrial keywords,
    with optional city / county bias.
    """
    parts = [BASE_KEYWORDS]

    if county:
        parts.append(f'"{county}"')
    if city:
        parts.append(f'"{city}"')

    cleaned = user_q.strip()
    if cleaned:
        parts.append(f"({cleaned})")

    return " ".join(parts)


def _google_cse_search(query: str, days: int) -> List[Dict[str, Any]]:
    """
    Thin wrapper around Google Programmable Search (Custom Search JSON API).
    Returns a list of *raw* result dicts.
    """
    if not GOOGLE_CSE_API_KEY or not GOOGLE_CSE_CX:
        log.warning("GOOGLE_CSE_API_KEY or GOOGLE_CSE_CX not set; returning empty result list.")
        return []

    params = {
        "key": GOOGLE_CSE_API_KEY,
        "cx": GOOGLE_CSE_CX,
        "q": query,
        "num": 10,  # up to 10 per request
    }

    log.info("Google CSE query: %s", query)

    try:
        resp = requests.get(GOOGLE_CSE_ENDPOINT, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.warning("Google CSE request failed: %s", e)
        return []

    items = data.get("items", []) or []
    out: List[Dict[str, Any]] = []

    for it in items:
        title = it.get("title") or ""
        snippet = it.get("snippet") or ""
        url = it.get("link") or ""

        # Rough "provider" = domain of the URL
        provider = ""
        if url:
            m_host = re.search(r"https?://([^/]+)/?", url)
            if m_host:
                provider = m_host.group(1)

        # Try to grab a published date from metadata if present
        date_str: Optional[str] = None
        try:
            pagemap = it.get("pagemap", {}) or {}
            metatags = pagemap.get("metatags", [])
            if isinstance(metatags, list) and metatags:
                meta0 = metatags[0]
                # Common fields websites use
                for key in (
                    "article:published_time",
                    "og:updated_time",
                    "date",
                    "pubdate",
                ):
                    if key in meta0:
                        date_str = str(meta0[key])
                        break
        except Exception:
            date_str = None

        dt_iso: Optional[str] = None
        if date_str:
            # Best-effort parse; many formats exist, so be defensive
            try:
                # Handle "2024-01-02T12:34:56Z" or similar
                ds = date_str.replace("Z", "+00:00")
                dt = datetime.fromisoformat(ds)
                dt_iso = dt.iso8601()
            except Exception:
                dt_iso = None

        out.append(
            {
                "title": title,
                "snippet": snippet,
                "url": url,
                "provider": provider,
                "date": dt_iso,
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
    - Builds a Google CSE query anchored on Indiana industrial projects.
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

    raw_results = _google_cse_search(query, days=days)
    filtered: List[Dict[str, Any]] = []

    for r in raw_results:
        if not _looks_relevant(r):
            continue

        r["city_hint"] = city
        r["county_hint"] = county
        filtered.append(r)

    # Sort by date desc if we have dates; otherwise keep Google order
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
            "Try adjusting the city/county or your keywords._"
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
