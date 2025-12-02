"""
indiana_intel.py

Indiana developments lead-finder for Tynan / Heli AI site.

Responsibilities:
- Take a natural-language question (e.g. "new warehouses in Boone County in the last 90 days")
- Hit Google Custom Search JSON API (Programmable Search Engine) for
  Indiana industrial / logistics / manufacturing / commercial projects
- Normalize results into simple dicts
- Provide a compact text summary for the chat UI or as context into GPT

Environment variables required:
- GOOGLE_CSE_KEY : Google API key for Custom Search JSON API
- GOOGLE_CSE_CX  : Programmable Search Engine ID (cx) configured to search the web
"""

from __future__ import annotations

import os
import re
import logging
from datetime import datetime, timezone, timedelta
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

# Base industrial / logistics / commercial keywords for Indiana
BASE_KEYWORDS = (
    'Indiana (warehouse OR "distribution center" OR logistics OR manufacturing '
    'OR plant OR factory OR industrial OR fulfillment OR "business park" '
    'OR "industrial park" OR "logistics park" OR headquarters OR facility)'
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
    "center", "centers", "companies", "coming", "to", "area", "city",
    "kind", "sort", "type",
}


def _build_query(user_q: str, city: Optional[str], county: Optional[str]) -> str:
    """
    Build a Google CSE query anchored on Indiana industrial/commercial keywords,
    with optional city / county bias, plus a small keyword tail from the question.
    (No domain restriction – we want as many relevant hits as possible.)
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
                raw = str(m[key]).strip()
                # Try ISO first
                try:
                    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                except Exception:
                    dt = None
                    # Fallback short formats
                    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
                        try:
                            dt = datetime.strptime(raw[:10], fmt)
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


def _score_relevance(user_q: str, title: str, snippet: str) -> int:
    """
    Crude relevance score based on overlapping non-stopword tokens between
    the user question and the result's title+snippet.
    """
    cleaned_q = re.sub(r"[“”\"']", " ", user_q or "")
    q_tokens = {
        t.lower()
        for t in re.findall(r"[A-Za-z0-9]+", cleaned_q)
        if t.lower() not in _STOPWORDS
    }

    haystack = f"{title} {snippet}".lower()
    score = 0
    for tok in q_tokens:
        if tok and tok in haystack:
            score += 1
    return score


def _infer_project_type(title: str, snippet: str) -> str:
    """
    Tiny heuristic to guess project type from title/snippet.
    Just used for UI/context – not mission critical.
    """
    text = f"{title} {snippet}".lower()

    if "distribution center" in text or "fulfillment center" in text:
        return "Distribution / fulfillment center"
    if "warehouse" in text or "logistics park" in text or "logistics center" in text:
        return "Warehouse / logistics"
    if "manufacturing" in text or "factory" in text or "plant" in text:
        return "Manufacturing / plant"
    if "headquarters" in text or "hq" in text:
        return "Headquarters / office"
    if "business park" in text or "industrial park" in text:
        return "Business / industrial park"
    return "Industrial / commercial project"


def _google_cse_search(
    query: str,
    max_items: int = 10,
    days: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Thin wrapper around Google Custom Search JSON API.
    Returns a list of normalized raw results (up to max_items).
    Uses pagination and optional dateRestrict (days).
    """
    if not GOOGLE_CSE_KEY or not GOOGLE_CSE_CX:
        log.warning(
            "GOOGLE_CSE_KEY or GOOGLE_CSE_CX not set or empty; returning empty result list. "
            f"GOOGLE_CSE_KEY present={bool(GOOGLE_CSE_KEY)}, GOOGLE_CSE_CX present={bool(GOOGLE_CSE_CX)}"
        )
        return []

    max_items = max(1, min(max_items, 30))  # hard safety cap

    out: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()
    start = 1

    while len(out) < max_items and start <= 91:
        remaining = max_items - len(out)
        num = min(10, remaining)

        params = {
            "key": GOOGLE_CSE_KEY,
            "cx": GOOGLE_CSE_CX,
            "q": query,
            "num": num,
            "start": start,
        }
        if days and days > 0:
            # "d90" = last 90 days
            params["dateRestrict"] = f"d{days}"

        try:
            resp = requests.get(GOOGLE_CSE_ENDPOINT, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.warning("Google CSE request failed (start=%s): %s", start, e)
            break

        items = data.get("items", []) or []
        log.info("Google CSE returned %s items for start=%s", len(items), start)
        if not items:
            break

        for it in items:
            if not isinstance(it, dict):
                continue

            title = it.get("title") or ""
            snippet = it.get("snippet") or it.get("htmlSnippet") or ""
            url = it.get("link") or ""
            provider = it.get("displayLink") or ""

            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

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

        start += len(items)
        # Conservative safety break to avoid looping forever if something weird happens
        if len(items) < num:
            break

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def search_indiana_developments(
    user_q: str,
    days: int = 365,
    max_items: int = 20,
) -> List[Dict[str, Any]]:
    """
    Main entrypoint.

    - Extracts simple city/county hints from the user question.
    - Builds a Google CSE query anchored on Indiana industrial/commercial projects.
    - Uses Google dateRestrict + local filtering to bias towards the last `days`.
    - Returns up to `max_items` normalized dicts:
      {
        "title": str,
        "snippet": str,
        "url": str,
        "provider": str,
        "date": ISO string or None,
        "city_hint": Optional[str],
        "county_hint": Optional[str],
        "score": int,                 # crude relevance score
        "project_type": str,          # e.g. "Warehouse / logistics"
      }
    """
    city, county = _extract_geo_hint(user_q)
    query = _build_query(user_q, city, county)

    raw_results = _google_cse_search(query, max_items=max_items, days=days)
    if not raw_results:
        return []

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    cutoff = now - timedelta(days=max(days, 1))

    enriched: List[Dict[str, Any]] = []
    for r in raw_results:
        title = r.get("title") or ""
        snippet = r.get("snippet") or ""
        dt_iso = r.get("date")

        # Parse date, if present
        parsed_dt: Optional[datetime] = None
        if dt_iso:
            try:
                parsed_dt = datetime.fromisoformat(dt_iso.replace("Z", "+00:00"))
                if parsed_dt.tzinfo is not None:
                    parsed_dt = parsed_dt.astimezone(timezone.utc).replace(tzinfo=None)
            except Exception:
                parsed_dt = None

        # Soft filter by recency only if we actually got a date
        if parsed_dt and parsed_dt < cutoff:
            # Too old for requested window; skip
            continue

        score = _score_relevance(user_q, title, snippet)
        proj_type = _infer_project_type(title, snippet)

        r["city_hint"] = city
        r["county_hint"] = county
        r["score"] = score
        r["project_type"] = proj_type
        r["_parsed_dt"] = parsed_dt  # private helper key for sorting

        enriched.append(r)

    # Sort by (score desc, date desc, title)
    def _sort_key(item: Dict[str, Any]) -> Tuple[int, datetime, str]:
        score = int(item.get("score") or 0)
        dt = item.get("_parsed_dt") or datetime.min
        title = item.get("title") or ""
        return (-score, dt, title.lower())

    enriched.sort(key=_sort_key, reverse=False)  # because score is negated

    # Strip internal helper key
    for r in enriched:
        r.pop("_parsed_dt", None)

    return enriched


def render_developments_markdown(items: List[Dict[str, Any]]) -> str:
    """
    Compact plain-text summary for chat UI or as context into GPT.
    (No *** or other markdown styling that will clash with your own prompt.)
    """
    if not items:
        return (
            "No web results were found for that location and timeframe. "
            "Try adjusting the date range or phrasing."
        )

    lines: List[str] = []
    lines.append("Recent Indiana projects (web search hits):")

    for i, item in enumerate(items[:15], start=1):
        title = item.get("title") or "Untitled"
        snippet = (item.get("snippet") or "").strip()
        url = item.get("url") or ""
        provider = item.get("provider") or ""
        date = item.get("date") or ""
        proj_type = item.get("project_type") or ""
        city_hint = item.get("city_hint") or ""
        county_hint = item.get("county_hint") or ""

        # Normalize date display
        if date:
            try:
                dt = datetime.fromisoformat(date.replace("Z", "+00:00"))
                if dt.tzinfo is not None:
                    dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
                date = dt.strftime("%Y-%m-%d")
            except Exception:
                pass

        lines.append(f"{i}. {title}")

        # Meta line: provider, date, type
        meta_bits = []
        if provider:
            meta_bits.append(provider)
        if date:
            meta_bits.append(date)
        if proj_type:
            meta_bits.append(proj_type)
        if meta_bits:
            lines.append("   " + " • ".join(meta_bits))

        # Location hint, if any
        loc_bits = []
        if city_hint:
            loc_bits.append(city_hint)
        if county_hint:
            loc_bits.append(county_hint)
        if loc_bits:
            lines.append("   Location focus: " + ", ".join(loc_bits))

        # Snippet + URL
        if snippet:
            lines.append(f"   {snippet}")
        if url:
            lines.append(f"   {url}")

    return "\n".join(lines)


__all__ = [
    "search_indiana_developments",
    "render_developments_markdown",
]
