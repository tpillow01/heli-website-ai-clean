"""
indiana_intel.py

Indiana developments lead-finder for Tynan / Heli AI site.

Responsibilities:
- Take a natural-language question (e.g. "new warehouses in Boone County in the last 90 days")
- Hit Google Custom Search JSON API (Programmable Search Engine) for
  Indiana industrial / logistics / manufacturing projects
- Normalize results into simple dicts
- Provide HTML rendering for the chat UI and AI prompt

Environment variables required:
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

# We keep this for future tuning; many CSE results don't have reliable dates.
DEFAULT_DAYS = 365

# ---------------------------------------------------------------------------
# Helper utilities
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
    "center", "centers",
}


def _build_query(user_q: str, city: Optional[str], county: Optional[str]) -> str:
    """
    Build a Google CSE query that strongly targets facility-type results.
    We REQUIRE facility-style terms in the Google query so we don't pull
    gardening clubs, tourism, or generic county pages.
    """
    parts: List[str] = []

    # Geo anchor first
    if county:
        parts.append(f'"{county}"')
    if city:
        parts.append(f'"{city}"')

    # Always anchor on Indiana
    parts.append("Indiana")

    # Facility-type keywords we actually care about
    facility_clause = (
        '("warehouse" OR "distribution center" OR "distribution facility" '
        'OR "logistics center" OR "logistics hub" OR "logistics park" '
        'OR "fulfillment center" OR "industrial park" OR "industrial park")'
    )
    parts.append(facility_clause)

    # Small tail from user question, minus boring words
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

    query = " ".join(parts).strip()
    log.info("Google CSE query: %s", query)
    return query


def _google_cse_search(query: str, days: int) -> List[Dict[str, Any]]:
    """
    Thin wrapper around Google Custom Search JSON API.
    Returns a list of raw results.
    """
    if not GOOGLE_CSE_KEY or not GOOGLE_CSE_CX:
        log.warning(
            "GOOGLE_CSE_KEY or GOOGLE_CSE_CX not set or empty; returning empty result list. "
            "GOOGLE_CSE_KEY present=%s, GOOGLE_CSE_CX present=%s",
            bool(GOOGLE_CSE_KEY),
            bool(GOOGLE_CSE_CX),
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
        provider = ""

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
    STRONGER relevance filter:
    - We only keep results that clearly look like a FACILITY or SITE,
      not generic county info or tourism pages.
    - To pass, the text must include at least one of a list of facility terms.
    """
    text = _lower(item.get("title", "") + " " + item.get("snippet", ""))
    if not text:
        return False

    facility_terms = (
        "warehouse",
        "distribution center",
        "distribution facility",
        "logistics center",
        "logistics hub",
        "logistics park",
        "fulfillment center",
        "industrial park",
        "industrial park",  # duplicated intentionally, it's cheap
        "spec building",
        "industrial building",
        "logistics campus",
        "industrial campus",
        "cold storage",
    )

    return any(term in text for term in facility_terms)


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
    - Builds a Google CSE query anchored on Indiana facility-type projects.
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

    IMPORTANT:
    - We DO NOT fall back to generic county/tourism/gardening pages.
      If nothing passes the facility filter, we return [] and let the
      caller say "no projects found" cleanly.
    """
    city, county = _extract_geo_hint(user_q)
    query = _build_query(user_q, city, county)

    raw_results = _google_cse_search(query, days=days)
    if not raw_results:
        log.info("No items returned by Google CSE.")
        return []

    filtered: List[Dict[str, Any]] = []
    for r in raw_results:
        if _looks_relevant(r):
            r["city_hint"] = city
            r["county_hint"] = county
            filtered.append(r)

    if not filtered:
        log.info("No items passed facility relevance filter; returning empty list.")
        return []

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
    We keep a neutral, structured HTML format; styling is handled by the caller.
    """
    if not items:
        return "_No recent Indiana developments found for that query._"

    # HTML only – no Markdown asterisks.
    lines: List[str] = []
    lines.append("<div>")
    lines.append("  <ol>")

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

        location_hint_parts = []
        city_hint = item.get("city_hint")
        county_hint = item.get("county_hint")
        if city_hint:
            location_hint_parts.append(city_hint)
        if county_hint:
            location_hint_parts.append(county_hint)
        location_hint = ", ".join(location_hint_parts) or "Indiana"

        lines.append("    <li>")
        # Name styled in dark red + bold; if your UI strips HTML, you'll just see the text.
        lines.append(
            f'      <span style="color:#b00000;font-weight:bold;">{title}</span><br>'
        )
        lines.append(f"      Location hint: {location_hint}<br>")
        if provider:
            lines.append(f"      Source: {provider}<br>")
        if date:
            lines.append(f"      Date (from metadata): {date}<br>")
        if snippet:
            lines.append(f"      Snippet: {snippet}<br>")
        if url:
            lines.append(f'      URL: <a href="{url}" target="_blank">{url}</a>')
        lines.append("    </li>")

    lines.append("  </ol>")
    lines.append("</div>")

    return "\n".join(lines)


__all__ = [
    "search_indiana_developments",
    "render_developments_markdown",
]
