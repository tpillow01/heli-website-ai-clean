"""
indiana_intel.py

Indiana developments lead-finder for Tynan / Heli AI site.

Responsibilities:
- Take a natural-language question (e.g. "new warehouses in Boone County in the last 90 days")
- Hit Google Custom Search JSON API (Programmable Search Engine) for
  Indiana industrial / logistics / manufacturing projects
- Normalize results into simple dicts
- Provide markdown rendering for the chat UI and AI prompt

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

# Conceptual window; Google CSE doesn't strictly filter by date but we keep this
DEFAULT_DAYS = 365

# Base industrial / logistics keywords for Indiana
BASE_KEYWORDS = (
    'Indiana (warehouse OR "distribution center" OR logistics OR manufacturing '
    'OR plant OR factory OR industrial OR fulfillment)'
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
        raw = re.sub(r"\b(Indiana|IN)\b\.?", "", raw, flags=re.I).strip()
        if raw:
            city = raw

    return (city, county)


_STOPWORDS = {
    "what", "are", "there", "any", "new", "or", "in", "the", "last", "month",
    "months", "recent", "recently", "project", "projects", "have", "has",
    "been", "announced", "announcement", "for", "about", "on", "of", "a",
    "an", "county", "indiana", "logistics", "warehouse", "distribution",
    "center", "centers", "today", "current", "currently",
}


def _build_query(user_q: str, city: Optional[str], county: Optional[str]) -> str:
    """
    Build a Google CSE query that always anchors on Indiana industrial keywords,
    with optional city / county bias, and a *small* extra keyword tail from the question.
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
        parts.append(" ".join(extra_tokens[:8]))

    query = " ".join(parts)
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
        "logistics facility",
        "supply chain",
    ):
        if kw in text:
            hits += 1

    return hits > 0


def _sort_key_date(item: Dict[str, Any]) -> float:
    d = item.get("date")
    if not d:
        return 0.0
    try:
        return datetime.fromisoformat(d).timestamp()
    except Exception:
        return 0.0


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
    - Builds one or more Google CSE queries anchored on Indiana industrial projects.
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

    # --- 1) Primary query (as before) ---
    primary_query = _build_query(user_q, city, county)
    all_raw: List[Dict[str, Any]] = _google_cse_search(primary_query, days=days)

    # --- 2) Extra queries to pull in more candidates for that county ---
    # We keep these pretty generic so Google can find:
    # - press releases
    # - town / county meeting notes
    # - industrial park announcements, etc.
    county_term = county or ""
    city_term = city or ""

    extra_queries: List[str] = []

    if county_term:
        extra_queries.append(
            f'Indiana "{county_term}" (warehouse OR '
            f'"distribution center" OR logistics OR "industrial park" OR '
            f'"logistics park" OR "fulfillment center" OR "industrial development")'
        )

    # If we have a city, bias another query to that city name.
    if city_term:
        extra_queries.append(
            f'"{city_term}" Indiana (warehouse OR "distribution center" '
            f'OR logistics OR "industrial park" OR "logistics park")'
        )

    # If user asked about 12 months / last year etc., add a generic
    # "project" query too to catch things that don’t say “warehouse”
    # in the snippet but are clearly developments.
    q_lower = (user_q or "").lower()
    if "12" in q_lower or "last year" in q_lower or "12 months" in q_lower:
        extra_queries.append(
            f'Indiana "{county_term or city_term}" '
            f'(project OR development OR expansion) '
            f'(industrial OR logistics OR warehouse OR distribution)'
        )

    # Run all extra queries and merge results
    for q in extra_queries:
        more = _google_cse_search(q, days=days)
        if more:
            all_raw.extend(more)

    if not all_raw:
        return []

    # --- 3) De-duplicate by URL so the same article doesn’t show up 3 times ---
    seen_urls: set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for r in all_raw:
        url = (r.get("url") or "").strip()
        if not url:
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)
        deduped.append(r)

    if not deduped:
        return []

    # --- 4) Apply a *soft* relevance filter ---
    filtered: List[Dict[str, Any]] = []
    for r in deduped:
        if _looks_relevant(r):
            r["city_hint"] = city
            r["county_hint"] = county
            filtered.append(r)

    # If our filter killed everything, fall back to the top few raw items
    if not filtered:
        log.info("No items passed relevance filter; falling back to raw top items.")
        filtered = deduped[:8]
        for r in filtered:
            r["city_hint"] = city
            r["county_hint"] = county

    # --- 5) Sort by date if we have it; otherwise keep Google’s order ---
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
    Compact text summary for chat UI or as context into GPT.
    This is NOT the final formatted answer to the user; it's just
    structured context the AI will read.
    """
    if not items:
        return "NO_RESULTS"

    lines: List[str] = ["RAW_INDIANA_DEVELOPMENTS_RESULTS"]
    for i, item in enumerate(items[:15], start=1):
        title = item.get("title") or "Untitled"
        snippet = (item.get("snippet") or "").strip()
        url = item.get("url") or ""
        provider = item.get("provider") or ""
        date = item.get("date") or ""
        county = item.get("county_hint") or ""
        city = item.get("city_hint") or ""

        if date:
            try:
                dt = datetime.fromisoformat(date)
                date = dt.strftime("%Y-%m-%d")
            except Exception:
                pass

        lines.append(f"ITEM {i}:")
        lines.append(f"  TITLE: {title}")
        if date:
            lines.append(f"  DATE: {date}")
        if city:
            lines.append(f"  CITY_HINT: {city}")
        if county:
            lines.append(f"  COUNTY_HINT: {county}")
        if provider:
            lines.append(f"  PROVIDER: {provider}")
        if snippet:
            lines.append(f"  SNIPPET: {snippet}")
        if url:
            lines.append(f"  URL: {url}")
        lines.append("")  # blank line between items

    return "\n".join(lines)


__all__ = [
    "search_indiana_developments",
    "render_developments_markdown",
]
