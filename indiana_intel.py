"""
indiana_intel.py

Indiana developments lead-finder for Tynan / Heli AI site.

Responsibilities:
- Take a natural-language question (e.g. "new warehouses in Boone County in the last 90 days")
- Hit Google Custom Search JSON API (Programmable Search Engine) for
  Indiana industrial / logistics / manufacturing projects
- Normalize results into simple dicts
- Provide HTML/markdown rendering for the chat UI and AI prompt

Environment variables required:
- GOOGLE_CSE_KEY : Google API key for Custom Search JSON API
- GOOGLE_CSE_CX  : Programmable Search Engine ID (cx) configured to search the web
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
# Config: Google CSE
# ---------------------------------------------------------------------------
GOOGLE_CSE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"
GOOGLE_CSE_KEY = os.environ.get("GOOGLE_CSE_KEY")
GOOGLE_CSE_CX = os.environ.get("GOOGLE_CSE_CX")

# Default lookback window when caller doesn't specify (in days)
DEFAULT_DAYS = 365  # aim for "last 12 months" style questions

# Base industrial / logistics keywords for Indiana
BASE_KEYWORDS = (
    'Indiana (warehouse OR "distribution center" OR logistics OR manufacturing '
    'OR plant OR factory OR industrial OR "industrial park" OR "fulfillment center")'
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
    m_county = re.search(r"\b([A-Za-z]+)\s+County\b", text, flags=re.I)
    county = None
    if m_county:
        county = f"{m_county.group(1).strip()} County"

    # City: look for "in Greenwood", "around Greenwood, IN", etc.
    m_city = re.search(r"\b(?:in|around|near)\s+([A-Za-z\s]+?)(?:,|\?|\.|$)", text, flags=re.I)
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
    "center", "centers", "developments",
}


def _build_query(user_q: str, city: Optional[str], county: Optional[str], days: int) -> str:
    """
    Build a Google CSE query that:
    - Anchors on Indiana industrial/warehouse keywords
    - Biases toward a specific county/city if present
    - Adds a soft 'after:' date tail when the user asks for recent activity
    - Adds a few non-boring tokens from the question
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

    # Try to bias Google toward recency using after: operator
    try:
        since = datetime.utcnow() - timedelta(days=max(30, days))  # minimum 30-day window
        parts.append(f"after:{since.strftime('%Y-%m-%d')}")
    except Exception:
        pass

    query = " ".join(parts)
    log.info("Google CSE query: %s", query)
    return query


def _google_cse_search(query: str, num: int = 10, start: int = 1) -> List[Dict[str, Any]]:
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
        "num": num,
        "start": start,
    }

    try:
        resp = requests.get(GOOGLE_CSE_ENDPOINT, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.warning("Google CSE request failed: %s", e)
        return []

    items = data.get("items", []) or []
    log.info("Google CSE returned %s items (start=%s)", len(items), start)

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
    ):
        if kw in text:
            hits += 1

    return hits > 0


def _within_days(item: Dict[str, Any], days: int) -> bool:
    """
    True if item.date is within the last N days (when date is parseable).
    If no date, we treat it as unknown (return True) so we don't accidentally
    throw away possibly-good results just because they lack date metadata.
    """
    d = item.get("date")
    if not d:
        return True
    try:
        dt = datetime.fromisoformat(d)
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

    Recency & quantity strategy:
    - Run a base query.
    - If a county is present, run a second county-focused query.
    - Pull up to ~20 results, de-duplicate by URL.
    - Filter by industrial relevance.
    - Prefer projects with dates within the requested window (days).
      If that leaves you with zero, fall back to older/undated items
      instead of returning nothing.
    """
    city, county = _extract_geo_hint(user_q)
    main_query = _build_query(user_q, city, county, days)

    all_raw: List[Dict[str, Any]] = []

    # 1) Main query (top 10)
    all_raw.extend(_google_cse_search(main_query, num=10, start=1))

    # 2) If we have a county, add a second, more focused query to widen net
    if county:
        county_query = (
            f'"{county}" Indiana '
            '(warehouse OR "distribution center" OR logistics OR "industrial park" '
            'OR "logistics park" OR "fulfillment center")'
        )
        log.info("Google CSE county-focused query: %s", county_query)
        all_raw.extend(_google_cse_search(county_query, num=10, start=1))

    # De-duplicate by URL
    dedup: Dict[str, Dict[str, Any]] = {}
    for r in all_raw:
        url = (r.get("url") or "").strip()
        if not url:
            continue
        if url not in dedup:
            dedup[url] = r

    raw_results = list(dedup.values())
    if not raw_results:
        return []

    # First pass: apply relevance filter
    relevant: List[Dict[str, Any]] = [r for r in raw_results if _looks_relevant(r)]
    if not relevant:
        log.info("No items passed relevance filter; falling back to raw items.")
        relevant = raw_results

    # Second pass: prefer projects within the requested date window
    recent_only = [r for r in relevant if _within_days(r, days)]
    if recent_only:
        chosen = recent_only
        log.info(
            "Using %s recent items within %s days (out of %s relevant)",
            len(recent_only),
            days,
            len(relevant),
        )
    else:
        # Nothing clearly within the window; keep the most relevant ones,
        # but caller can still see they might be older.
        chosen = relevant
        log.info(
            "No items clearly within %s days; falling back to %s relevant items.",
            days,
            len(relevant),
        )

    # Attach city/county hints
    for r in chosen:
        r.setdefault("city_hint", city)
        r.setdefault("county_hint", county)

    # Sort by date desc (newest first) when we have dates; otherwise keep API order
    def _dt(item: Dict[str, Any]) -> float:
        d = item.get("date")
        if not d:
            return 0.0
        try:
            return datetime.fromisoformat(d).timestamp()
        except Exception:
            return 0.0

    chosen.sort(key=_dt, reverse=True)
    return chosen


def render_developments_html(items: List[Dict[str, Any]]) -> str:
    """
    Compact HTML summary for chat UI.

    - Project names: bold + dark red for clear separation.
    - Simple card-like layout using basic inline styles so it works
      inside your existing chat bubble.
    """
    if not items:
        return (
            "<p>No clearly dated new warehouse or logistics project announcements were "
            "found in the search results for the requested time window.</p>"
        )

    parts: List[str] = []
    parts.append('<div style="display:flex;flex-direction:column;gap:10px;">')

    for item in items[:10]:
        title = item.get("title") or "Untitled project"
        snippet = (item.get("snippet") or "").strip()
        url = item.get("url") or ""
        provider = item.get("provider") or ""
        date = item.get("date") or ""
        city_hint = item.get("city_hint") or ""
        county_hint = item.get("county_hint") or ""

        # Format date if we have it
        if date:
            try:
                dt = datetime.fromisoformat(date)
                date = dt.strftime("%Y-%m-%d")
            except Exception:
                pass

        # Build location string
        loc_bits: List[str] = []
        if city_hint:
            loc_bits.append(city_hint)
        if county_hint and county_hint not in loc_bits:
            loc_bits.append(county_hint)
        location = ", ".join(loc_bits) if loc_bits else ""

        parts.append(
            '<div style="padding:8px 10px;border-radius:6px;'
            'border:1px solid #444;background:#111;margin-bottom:4px;">'
        )

        # Project name: bold + dark red
        parts.append(
            f'<div style="font-weight:bold;color:#b30000;margin-bottom:2px;">'
            f'{title}'
            f'</div>'
        )

        if location:
            parts.append(
                f'<div style="font-size:0.9rem;color:#ddd;">'
                f'<span style="font-weight:bold;">Location:</span> {location}'
                f'</div>'
            )

        if provider:
            parts.append(
                f'<div style="font-size:0.85rem;color:#aaa;">'
                f'<span style="font-weight:bold;">Source:</span> {provider}'
                f'</div>'
            )

        if snippet:
            parts.append(
                f'<div style="margin-top:4px;font-size:0.95rem;color:#eee;">'
                f'{snippet}'
                f'</div>'
            )

        if date:
            parts.append(
                f'<div style="margin-top:4px;font-size:0.85rem;color:#ccc;">'
                f'<span style="font-weight:bold;">Timeline:</span> {date}'
                f'</div>'
            )

        if url:
            parts.append(
                f'<div style="margin-top:4px;font-size:0.85rem;">'
                f'<a href="{url}" target="_blank" '
                f'style="color:#66b3ff;text-decoration:none;">'
                f'View source</a></div>'
            )

        parts.append("</div>")  # end card

    parts.append("</div>")
    return "".join(parts)


def render_developments_markdown(items: List[Dict[str, Any]]) -> str:
    """
    Markdown fallback (no *** bold markers, just simple bullets).
    """
    if not items:
        return (
            "No clearly dated new warehouse or logistics project announcements were "
            "found in the search results for the requested time window."
        )

    lines: List[str] = []
    for item in items[:10]:
        title = item.get("title") or "Untitled project"
        snippet = (item.get("snippet") or "").strip()
        url = item.get("url") or ""
        date = item.get("date") or ""
        if date:
            try:
                dt = datetime.fromisoformat(date)
                date = dt.strftime("%Y-%m-%d")
            except Exception:
                pass

        lines.append(f"- {title}")
        if date:
            lines.append(f"  - Timeline: {date}")
        if snippet:
            lines.append(f"  - {snippet}")
        if url:
            lines.append(f"  - {url}")

    return "\n".join(lines)


__all__ = [
    "search_indiana_developments",
    "render_developments_html",
    "render_developments_markdown",
]
