"""
indiana_intel.py

Indiana developments lead-finder for Tynan / Heli AI site.

Responsibilities:
- Take a natural-language question (e.g. "new warehouses in Boone County in the last 12–18 months")
- Hit Google Custom Search JSON API (Programmable Search Engine) for
  Indiana industrial / logistics / manufacturing / warehouse projects
- Normalize results into simple dicts
- Provide markdown rendering for the chat UI and AI prompt

Environment variables required (set in Render):
- GOOGLE_CSE_KEY : Google API key for Custom Search JSON API
- GOOGLE_CSE_CX  : Programmable Search Engine ID (cx) configured to search the web
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
# Config: Google CSE
# ---------------------------------------------------------------------------
GOOGLE_CSE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"
GOOGLE_CSE_KEY = os.environ.get("GOOGLE_CSE_KEY")
GOOGLE_CSE_CX = os.environ.get("GOOGLE_CSE_CX")

# We don't really have reliable dates from CSE; keep "days" parameter for API compatibility.
DEFAULT_DAYS = 365

# Base industrial / logistics keywords for Indiana
BASE_KEYWORDS = (
    'Indiana (warehouse OR "distribution center" OR logistics OR manufacturing '
    'OR plant OR factory OR industrial OR fulfillment OR "industrial park")'
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
    - Tries to grab a city name after 'in ' / 'around ' / 'near '.
    Returns (city, county).
    """
    if not q:
        return (None, None)
    text = q.strip()

    # County: e.g. "in Boone County", "Boone County leads", etc.
    m_county = re.search(r"\b([A-Za-z]+)\s+County\b", text, flags=re.I)
    county = None
    if m_county:
        # Title-case for nicer logging/queries
        name = m_county.group(1).strip()
        county = f"{name.title()} County"

    # City: e.g. "in Greenwood", "around Plainfield, IN", etc.
    m_city = re.search(r"\b(?:in|around|near)\s+([A-Za-z\s]+?)(?:,|\?|\.|$)", text, flags=re.I)
    city = None
    if m_city:
        raw = m_city.group(1).strip()
        # Strip trailing "Indiana" / "IN"
        raw = re.sub(r"\b(Indiana|IN)\b\.?", "", raw, flags=re.I).strip()
        if raw:
            city = raw.title()

    return (city, county)


# Words from the user question we don't want to blindly shove into the Google query
_STOPWORDS = {
    "what", "are", "there", "any", "new", "or", "in", "the", "last", "month",
    "months", "recent", "recently", "project", "projects", "have", "has",
    "been", "announced", "announcement", "for", "about", "on", "of", "a",
    "an", "county", "indiana", "logistics", "warehouse", "warehouses",
    "distribution", "center", "centers", "developments", "development",
    "years", "year", "days", "day",
}


def _build_query(user_q: str, city: Optional[str], county: Optional[str]) -> str:
    """
    Build a Google CSE query that:
    - Always anchors on Indiana + industrial/warehouse keywords
    - Adds city / county if we found them
    - Adds a small cleaned tail from the question as a soft signal
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
        # Skip stopwords and pure numbers (12, 18, etc.)
        if tl in _STOPWORDS:
            continue
        if tl.isdigit():
            continue
        extra_tokens.append(tok)

    if extra_tokens:
        parts.append(" ".join(extra_tokens[:8]))

    query = " ".join(parts)
    log.info("indiana_intel: Google CSE query: %s", query)
    return query


def _google_cse_search(query: str, days: int) -> List[Dict[str, Any]]:
    """
    Thin wrapper around Google Custom Search JSON API.
    Returns a list of raw results.
    """
    key_present = bool(GOOGLE_CSE_KEY)
    cx_present = bool(GOOGLE_CSE_CX)
    if not key_present or not cx_present:
        log.warning(
            "GOOGLE_CSE_KEY or GOOGLE_CSE_CX not set or empty; returning empty result list. "
            "GOOGLE_CSE_KEY present=%s, GOOGLE_CSE_CX present=%s",
            key_present,
            cx_present,
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
    log.info("indiana_intel: Google CSE returned %s items", len(items))

    out: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue

        title = it.get("title") or ""
        snippet = it.get("snippet") or it.get("htmlSnippet") or ""
        url = it.get("link") or ""
        provider = ""  # CSE usually doesn't give us a clean provider name

        # Try to sniff out a date from metatags (best-effort only)
        dt_iso: Optional[str] = None
        pagemap = it.get("pagemap") or {}
        meta_list = pagemap.get("metatags") or []
        if isinstance(meta_list, list):
            for m in meta_list:
                if not isinstance(m, dict):
                    continue
                for key in (
                    "article:published_time",
                    "og:updated_time",
                    "date",
                    "dc.date",
                    "pubdate",
                ):
                    if key in m:
                        dt_iso = str(m[key])
                        break
                if dt_iso:
                    break

        # Normalize date string a bit
        if dt_iso:
            try:
                dt = datetime.fromisoformat(dt_iso.replace("Z", "+00:00"))
                dt_iso = dt.isoformat()
            except Exception:
                # keep raw if we can't parse
                pass

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
    If nothing passes this filter, the caller will fall back to raw items.
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
        "logistics park",
        "logistics center",
    ):
        if kw in text:
            hits += 1

    return hits > 0


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
        "provider": str (often empty with CSE),
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
    filtered: List[Dict[str, Any]] = []
    for r in raw_results:
        if _looks_relevant(r):
            r["city_hint"] = city
            r["county_hint"] = county
            filtered.append(r)

    # If our filter killed everything, just fall back to the top few raw items
    if not filtered:
        log.info("indiana_intel: No items passed relevance filter; falling back to raw top items.")
        filtered = raw_results[:5]
        for r in filtered:
            r["city_hint"] = city
            r["county_hint"] = county

    # Sort by date desc (newest first) when we have dates; otherwise keep API order
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
    Compact HTML summary for chat UI or as context into GPT.
    - No Markdown (** or __)
    - Project titles in bold dark red for easy scanning
    """
    if not items:
        return "No recent Indiana developments found for that query. Try widening the date range or adjusting the city/county."

    lines: List[str] = ['<div class="indiana-intel-list">']
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

        lines.append('  <div class="intel-item">')
        lines.append(
            '    <div class="intel-title">'
            f'<span style="color:#b00000;font-weight:bold;">{title}</span>'
            '</div>'
        )
        if meta:
            lines.append(f'    <div class="intel-meta">{meta}</div>')
        if snippet:
            lines.append(f'    <div class="intel-snippet">{snippet}</div>')
        if url:
            lines.append(f'    <div class="intel-link">{url}</div>')
        lines.append('  </div>')

    lines.append('</div>')
    return "\n".join(lines)

