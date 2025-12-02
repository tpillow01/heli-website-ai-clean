"""
indiana_intel.py

Indiana developments lead-finder for Tynan / Heli AI site.

Responsibilities:
- Take a natural-language question (e.g. "new warehouses in Boone County in the last 90 days")
- Hit Google Custom Search JSON API (Programmable Search Engine) for
  Indiana industrial / logistics / manufacturing projects
- Normalize results into simple dicts
- Provide a compact text/markdown rendering for AI context or display

Environment variables required:
- GOOGLE_CSE_KEY : Google API key for Custom Search JSON API
- GOOGLE_CSE_CX  : Programmable Search Engine ID (cx) configured to search the web
"""

from __future__ import annotations

import os
import re
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

log = logging.getLogger("indiana_intel")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s"
    )

# ---------------------------------------------------------------------------
# Config: Google CSE
# ---------------------------------------------------------------------------
GOOGLE_CSE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"
GOOGLE_CSE_KEY = os.environ.get("GOOGLE_CSE_KEY")
GOOGLE_CSE_CX = os.environ.get("GOOGLE_CSE_CX")

# How far back we try to keep things by default
DEFAULT_DAYS = 365

# Base industrial / logistics keywords for Indiana
BASE_KEYWORDS = (
    'Indiana (warehouse OR "distribution center" OR logistics OR manufacturing '
    'OR plant OR factory OR industrial OR fulfillment)'
)

# Bias toward economic development / gov / business news domains
PREFERRED_DOMAIN_HINTS = [
    "site:.gov",
    "site:.us",
    "site:.org",
    "site:.edu",
    "site:.in.us",
    "site:insideindianabusiness.com",
    "site:ibj.com",
]

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
        raw = re.sub(r"\b(Indiana|IN)\b\.?", "", raw, flags=re.I).strip()
        if raw:
            city = raw

    return (city, county)


_STOPWORDS = {
    "what", "are", "there", "any", "new", "or", "in", "the", "last", "month",
    "months", "recent", "recently", "project", "projects", "have", "has",
    "been", "announced", "announcement", "for", "about", "on", "of", "a",
    "an", "county", "indiana", "logistics", "warehouse", "distribution",
    "center", "centers", "companies", "coming", "to", "area", "city",
}


def _build_query(user_q: str, city: Optional[str], county: Optional[str]) -> str:
    """
    Build a Google CSE query that always anchors on Indiana industrial keywords,
    with optional city / county bias, + a small keyword tail from the question.
    We also include one of the PREFERRED_DOMAIN_HINTS to bias toward news / econ dev.
    """
    parts: List[str] = [BASE_KEYWORDS]

    if county:
        parts.append(f'"{county}"')
    if city:
        parts.append(f'"{city}"')

    # Clean the user question and add a few non-boring keywords as a soft hint
    cleaned = re.sub(r"[“”\"']", " ", user_q or "")
    tokens = re.findall(r"[A-Za-z0-9]+", cleaned)
    extra_tokens: List[str] = []
    for tok in tokens:
        tl = tok.lower()
        if tl in _STOPWORDS:
            continue
        extra_tokens.append(tok)
    if extra_tokens:
        parts.append(" ".join(extra_tokens[:6]))

    # Add a domain hint to bias toward useful sources
    # (we just pick the first one for now)
    if PREFERRED_DOMAIN_HINTS:
        parts.append(PREFERRED_DOMAIN_HINTS[0])

    query = " ".join(parts)
    log.info("Google CSE query: %s", query)
    return query


def _parse_date_from_pagemap(it: Dict[str, Any]) -> Optional[str]:
    """
    Try to sniff out a publication/update date from Google CSE pagemap metatags.
    Returns ISO string (UTC, naive) or None.
    """
    pagemap = it.get("pagemap") or {}
    meta_list = pagemap.get("metatags") or []
    if not isinstance(meta_list, list):
        return None

    candidates: List[str] = []
    for m in meta_list:
        if not isinstance(m, dict):
            continue
        for key in (
            "article:published_time",
            "article:modified_time",
            "og:updated_time",
            "date",
            "dc.date",
            "pubdate",
        ):
            if key in m and m[key]:
                candidates.append(str(m[key]))

    for raw in candidates:
        val = raw.strip()
        # Try ISO first, then fall back to some common date patterns
        try:
            dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
        except Exception:
            # Loose parsing for common "YYYY-MM-DD" or "YYYY/MM/DD" forms
            for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
                try:
                    dt = datetime.strptime(val[:10], fmt)
                    # Treat as local → convert to UTC naive
                    dt = dt.replace(tzinfo=None)
                    break
                except Exception:
                    dt = None
            if not dt:
                continue

        # Normalize to UTC naive
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)

        return dt.isoformat()

    return None


def _google_cse_search(query: str, days: int) -> List[Dict[str, Any]]:
    """
    Thin wrapper around Google Custom Search JSON API.
    Returns a list of normalized raw results.
    """
    if not GOOGLE_CSE_KEY or not GOOGLE_CSE_CX:
        log.warning(
            "GOOGLE_CSE_KEY or GOOGLE_CSE_CX not set or empty; returning empty result list. "
            f"GOOGLE_CSE_KEY present={bool(GOOGLE_CSE_KEY)}, GOOGLE_CSE_CX present={bool(GOOGLE_CSE_CX)}"
        )
        return []

    params = {
        "key": GOOGLE_CSE_KEY,
        "cx": GOOGLE_CSE_CX,
        "q": query,
        "num": 10,
    }

    try:
        resp = requests.get(GOOGLE_CSE_ENDPOINT, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.warning("Google CSE request failed: %s", e)
        return []

    items = data.get("items", []) or []
    log.info("Google CSE returned %s items", len(items))

    out: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue

        title = it.get("title") or ""
        snippet = it.get("snippet") or it.get("htmlSnippet") or ""
        url = it.get("link") or ""

        # provider: use displayLink if available
        provider = it.get("displayLink") or ""

        # Date: from metatags if we can
        dt_iso = _parse_date_from_pagemap(it)

        out.append(
            {
                "title": title,
                "snippet": snippet,
                "url": url,
                "provider": provider,
                "date": dt_iso,  # may be None
            }
        )

    return out


def _looks_relevant(item: Dict[str, Any]) -> bool:
    """
    Quick relevance filter: look for industrial / facility terms in title/snippet.
    If nothing passes this filter, we will fall back to the raw items.
    """
    text = _lower(item.get("title", "") + " " + item.get("snippet", ""))
    if not text:
        return False

    hits = 0
    for kw in (
        "warehouse",
        "distribution center",
        "distribution facility",
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
        "supply chain",
        "logistics facility",
        "distribution campus",
    ):
        if kw in text:
            hits += 1

    return hits > 0


def _within_days(item: Dict[str, Any], days: int) -> bool:
    """
    True if item.date is within the last N days (when date is parseable).

    We normalize everything to UTC *naive* so we don't get
    'can't compare offset-naive and offset-aware datetimes'.
    If no date or parsing fails, we treat it as unknown (return True)
    so we don't accidentally throw away possibly-good results.
    """
    d = item.get("date")
    if not d:
        return True

    try:
        dt = datetime.fromisoformat(d.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    except Exception:
        return True

    cutoff = datetime.utcnow() - timedelta(days=max(1, days))
    return dt >= cutoff


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
        "date": ISO string or None,
        "city_hint": Optional[str],
        "county_hint": Optional[str],
      }
    """
    city, county = _extract_geo_hint(user_q)
    query = _build_query(user_q, city, county)

    raw_results = _google_cse_search(query, days=days)
    if not raw_results:
        return []

    # First pass: apply relevance filter
    relevant: List[Dict[str, Any]] = []
    for r in raw_results:
        if _looks_relevant(r):
            relevant.append(r)

    # If nothing looks relevant, fall back to everything
    if not relevant:
        log.info("No items passed relevance filter; falling back to raw Google items.")
        relevant = list(raw_results)

    # Second pass: try to keep items within the date window if we have dates
    recent_only = [r for r in relevant if _within_days(r, days)]
    if recent_only:
        chosen = recent_only
    else:
        # No date-qualified hits; keep the relevant set but log it
        log.info("No items had parseable dates within %s days; returning relevance-filtered set.", days)
        chosen = relevant

    # Annotate with geo hints
    for r in chosen:
        r["city_hint"] = city
        r["county_hint"] = county

    # Sort by date desc (newest first) when we have dates; otherwise keep API order
    def _dt(item: Dict[str, Any]) -> float:
        d = item.get("date")
        if not d:
            return 0.0
        try:
            dt = datetime.fromisoformat(d.replace("Z", "+00:00"))
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt.timestamp()
        except Exception:
            return 0.0

    chosen.sort(key=_dt, reverse=True)

    # Cap to a reasonable number for AI context
    return chosen[:20]


def render_developments_markdown(items: List[Dict[str, Any]]) -> str:
    """
    Compact markdown/text summary for chat UI or as context into GPT.
    (No *** or other markdown styling that will clash with your own prompt.)
    """
    if not items:
        return (
            "No clearly dated new warehouse / logistics project announcements were found "
            "in the search results for the requested time window."
        )

    lines: List[str] = []
    lines.append("Recent Indiana projects (raw web hits):")

    for i, item in enumerate(items[:15], start=1):
        title = item.get("title") or "Untitled"
        snippet = (item.get("snippet") or "").strip()
        url = item.get("url") or ""
        provider = item.get("provider") or ""
        date = item.get("date") or ""
        if date:
            try:
                dt = datetime.fromisoformat(date.replace("Z", "+00:00"))
                if dt.tzinfo is not None:
                    dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
                date = dt.strftime("%Y-%m-%d")
            except Exception:
                pass

        lines.append(f"{i}. {title}")
        if provider or date:
            pieces = []
            if provider:
                pieces.append(provider)
            if date:
                pieces.append(date)
            lines.append("   " + " • ".join(pieces))
        if snippet:
            lines.append(f"   {snippet}")
        if url:
            lines.append(f"   {url}")

    return "\n".join(lines)


__all__ = [
    "search_indiana_developments",
    "render_developments_markdown",
]
