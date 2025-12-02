"""
indiana_intel.py

Indiana developments lead-finder for Tynan / Heli AI site.

Responsibilities:
- Take a natural-language question (e.g. "new warehouses in Boone County in the last 90 days")
- Hit Google Custom Search JSON API (Programmable Search Engine) for
  Indiana industrial / logistics / manufacturing / commercial projects
- Normalize results into simple dicts that the chat layer can format
- Provide a plain-text summary (mostly for debugging or backup use)

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
    - Tries to grab a city name after 'in ' if it looks like 'Plainfield' or 'Plainfield, IN'
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

    # City: look for "in Plainfield", "around Plainfield, IN", etc.
    m_city = re.search(r"\b(?:in|around|near)\s+([A-Za-z\s]+?)(?:,|\?|\.|$)", text)
    city = None
    if m_city:
        raw = m_city.group(1).strip()
        raw = re.sub(r"\b(Indiana|IN)\b\.?", "", raw, flags=re.I).strip()
        if raw:
            city = raw

    return (city, county)


def _extract_year_from_question(q: str) -> Optional[int]:
    """
    Try to find a 4-digit year like 2024, 2025, 2026 in the user's question.
    Used only as a soft hint for timeline interpretation / filtering.
    """
    if not q:
        return None
    for m in re.finditer(r"\b(20[2-4][0-9])\b", q):
        try:
            year = int(m.group(1))
            if 2000 <= year <= 2100:
                return year
        except Exception:
            continue
    return None


_STOPWORDS = {
    "what", "are", "there", "any", "new", "or", "in", "the", "last", "month",
    "months", "recent", "recently", "project", "projects", "have", "has",
    "been", "announced", "announcement", "for", "about", "on", "of", "a",
    "an", "county", "indiana", "logistics", "warehouse", "distribution",
    "center", "centers", "companies", "coming", "to", "area", "city",
    "kind", "sort", "type", "planned", "plan", "announce", "announced",
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


def _parse_date_from_pagemap(it: Dict[str, Any]) -> Optional[datetime]:
    """
    Try to sniff out a publication/update date from Google CSE pagemap metatags.
    Returns a datetime (UTC-naive) or None.
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
                            break
                        except Exception:
                            dt = None
                    if not dt:
                        continue

                # Normalize to UTC naive
                if dt.tzinfo is not None:
                    dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
                return dt

    return None


def _google_cse_search(query: str) -> List[Dict[str, Any]]:
    """
    Thin wrapper around Google Custom Search JSON API.
    Returns a list of normalized raw results:
      { title, snippet, url, provider, date (datetime or None) }
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
        provider = it.get("displayLink") or ""

        dt = _parse_date_from_pagemap(it)

        out.append(
            {
                "title": title,
                "snippet": snippet,
                "url": url,
                "provider": provider,
                "date": dt,  # datetime or None
            }
        )

    return out

# ---------------------------------------------------------------------------
# Classification & filtering – forklift relevance + junk removal
# ---------------------------------------------------------------------------

# Forklift-relevant positive signals
_FORKLIFT_POSITIVE = [
    "warehouse",
    "distribution center",
    "distribution facility",
    "distribution hub",
    "fulfillment center",
    "logistics center",
    "logistics facility",
    "logistics hub",
    "logistics park",
    "supply chain",
    "cross-dock",
    "cross dock",
    "cold storage",
    "industrial park",
    "business park",
    "industrial complex",
    "manufacturing plant",
    "production plant",
    "factory",
    "assembly plant",
    "equipment plant",
]

# Obvious “not a forklift customer” signals (or low-value noise)
_PROJECT_NEGATIVE_TEXT = [
    "official website",
    "faq",
    "faqs",
    "civicengage",
    "visit hendricks county",
    "visit plainfield",
    "events, shopping & family fun",
    "tourism",
    "welcome to plainfield",
    "quarterly welcome",
    "police shooting",
    "shooting incident",
    "museum",
    "performing arts",
    "theatre",
    "theater",
    "stadium",
    "arena",
    "sports complex",
    "golf",
    "resort",
    "water park",
    "amusement park",
    "hotel",
    "conference center",
    "convention center",
    "park pavilion",
    "community center",
    "library",
    "school",
    "elementary school",
    "high school",
    "middle school",
    "university",
    "college",
    "campus",
    "hospital",
    "medical center",
    "clinic",
    "mental health",
    "addiction services",
    "behavioral health",
    "apartments",
    "apartment complex",
    "housing development",
    "subdivision",
    "condominiums",
    "senior living",
    "assisted living",
]

_PROJECT_NEGATIVE_URL = [
    "facebook.com",
    "instagram.com",
    "twitter.com",
    "x.com",
    "youtube.com",
    "visit hendrickscounty",
    "visithendrickscounty",
    "tripadvisor.com",
]


def _looks_like_project_hit(title: str, snippet: str, url: str) -> bool:
    """
    Decide whether this CSE hit looks like a concrete forklift-relevant project
    vs. generic marketing, tourism, FAQ, or random news.
    """
    text = _lower(f"{title} {snippet}")
    url_l = _lower(url or "")

    # Kill obvious junk domains
    for bad in _PROJECT_NEGATIVE_URL:
        if bad in url_l:
            return False

    # Kill obvious non-forklift-y content
    for neg in _PROJECT_NEGATIVE_TEXT:
        if neg in text:
            return False

    # Require at least one forklift-relevant positive signal
    if not any(pos in text for pos in _FORKLIFT_POSITIVE):
        return False

    return True


def _classify_scope_and_location(
    title: str,
    snippet: str,
    city: Optional[str],
    county: Optional[str],
) -> Tuple[str, str]:
    """
    Very light classification into:
      - scope: "local" (explicitly mentions city/county) or "statewide"
      - location_label: human-readable label for display.
    """
    text = _lower(title) + " " + _lower(snippet)
    loc_label = "Indiana"

    scope = "statewide"
    if county:
        base = county.split()[0].lower()
        if base and base in text:
            scope = "local"
        loc_label = county
    if city:
        c = city.lower()
        if c and c in text:
            scope = "local"
        if not county:
            loc_label = city

    return scope, loc_label


def _infer_project_type(title: str, snippet: str) -> str:
    """
    Heuristic to categorize project type from title/snippet.
    We keep this conservative – no wild guessing.
    """
    text = _lower(title + " " + snippet)

    if any(w in text for w in ("warehouse", "fulfillment center", "distribution center", "distribution facility", "distribution hub")):
        return "warehouse / logistics facility"
    if any(w in text for w in ("logistics hub", "logistics park", "logistics center", "logistics facility")):
        return "warehouse / logistics facility"
    if any(w in text for w in ("manufacturing plant", "factory", "plant expansion", "production plant", "assembly plant", "equipment plant")):
        return "manufacturing plant"
    if any(w in text for w in ("industrial park", "business park", "industrial complex")):
        return "business / industrial park"
    if any(w in text for w in ("headquarters", "hq", "office building")):
        return "HQ / office project"

    return "Industrial / commercial project"

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_projects(
    raw_items: List[Dict[str, Any]],
    city: Optional[str],
    county: Optional[str],
    user_q: str,
) -> List[Dict[str, Any]]:
    """
    Convert raw CSE results into the structured dicts used by the chat layer.
    We do NOT invent square footage, job counts, or dollars – those stay None.
    """
    inferred_year = _extract_year_from_question(user_q)
    original_area_label = county or city or "Indiana"

    projects: List[Dict[str, Any]] = []
    for it in raw_items:
        title = it.get("title") or ""
        snippet = it.get("snippet") or ""
        url = it.get("url") or ""
        provider = it.get("provider") or ""
        dt = it.get("date")  # datetime or None

        # Filter out obvious non-project / non-forklift hits
        if not _looks_like_project_hit(title, snippet, url):
            continue

        scope, location_label = _classify_scope_and_location(title, snippet, city, county)
        project_type = _infer_project_type(title, snippet)

        # Timeline
        timeline_year: Optional[int] = dt.year if isinstance(dt, datetime) else None
        if inferred_year and (timeline_year is not None) and (timeline_year != inferred_year):
            timeline_stage = "outside requested timeframe"
        else:
            timeline_stage = "announcement" if timeline_year else "not specified in snippet"

        projects.append(
            {
                # Core identity
                "project_name": title or "Untitled project",
                "company": None,  # we let the chat layer say "not specified in snippet"
                "project_type": project_type,

                # Geography
                "scope": scope,  # "local" or "statewide"
                "location_label": location_label,
                "original_area_label": original_area_label,

                # Scale / economics – kept None (no guessing)
                "sqft": None,
                "jobs": None,
                "investment": None,

                # Timeline
                "timeline_stage": timeline_stage,
                "timeline_year": timeline_year,
                "raw_date": dt,  # keep raw datetime for recency sorting

                # Source
                "url": url,
                "provider": provider,
                "snippet": snippet,
            }
        )

    return projects

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_indiana_developments(
    user_q: str,
    days: int = 365,
    max_items: int = 30,
) -> List[Dict[str, Any]]:
    """
    Main entrypoint.

    - Extracts simple city/county hints from the user question.
    - Builds a Google CSE query anchored on Indiana industrial/commercial projects.
    - First attempts a county/city-biased search.
    - If no results at all, falls back to a statewide search (no city/county constraints)
      so the chat layer can still talk about major Indiana projects.
    - Applies a recency preference: it tries to keep only items whose date is within
      `days` (but not more than about 4 years). If that would yield nothing, it
      falls back to "whatever we have" instead of returning empty.
    - Returns a list of normalized project dicts:

      {
        "project_name": str,
        "company": Optional[str],
        "project_type": str,
        "scope": "local" | "statewide",
        "location_label": str,
        "original_area_label": str,
        "sqft": Optional[str],
        "jobs": Optional[str],
        "investment": Optional[str],
        "timeline_stage": str,
        "timeline_year": Optional[int],
        "url": str,
        "provider": str,
        "snippet": str,
        "raw_date": Optional[datetime],
      }
    """
    city, county = _extract_geo_hint(user_q)
    original_area_label = county or city or "Indiana"
    _ = original_area_label  # reserved for future use

    # 1) County/city-biased search
    query_local = _build_query(user_q, city, county)
    raw_local = _google_cse_search(query_local)

    # 2) If nothing at all came back, fall back to a statewide search
    if not raw_local:
        log.info("No local results; trying statewide fallback query")
        query_statewide = _build_query(user_q, city=None, county=None)
        raw_statewide = _google_cse_search(query_statewide)
        if not raw_statewide:
            return []

        projects = _normalize_projects(raw_statewide, city=None, county=None, user_q=user_q)
    else:
        # We have some local-biased hits; normalize them.
        projects = _normalize_projects(raw_local, city=city, county=county, user_q=user_q)

    if not projects:
        return []

    # -------------------------------------------------------------------
    # Recency preference
    # -------------------------------------------------------------------
    # Interpret `days` as "prefer results within this many days, capped at ~4 years".
    # Your /api/chat currently passes something like 365*10; this keeps it from
    # dragging in very old stuff if newer projects exist.
    max_recent_days = min(days, 365 * 4)  # ~4 years max window
    now = datetime.utcnow()

    recent: List[Dict[str, Any]] = []
    older: List[Dict[str, Any]] = []

    for p in projects:
        dt = p.get("raw_date")
        if isinstance(dt, datetime):
            age_days = (now - dt).days
            if age_days <= max_recent_days:
                recent.append(p)
            else:
                older.append(p)
        else:
            # No date info; treat as older / unknown
            older.append(p)

    if recent:
        candidates = recent
    else:
        # Nothing recent – fall back so we give *something* instead of "nothing found"
        candidates = older

    # Sort newest → oldest using raw_date
    def _sort_key(p: Dict[str, Any]) -> datetime:
        dt = p.get("raw_date")
        if isinstance(dt, datetime):
            return dt
        # Push unknown dates to the back
        return datetime(1900, 1, 1)

    candidates.sort(key=_sort_key, reverse=True)

    return candidates[:max_items]


def render_developments_markdown(items: List[Dict[str, Any]]) -> str:
    """
    Simple markdown-ish debug formatter. This is what your /api/chat mode
    passes into the model as context, so we make sure it includes recency
    and forklift relevance.
    """
    if not items:
        return (
            "No web results were found for that location and timeframe. "
            "Try adjusting the date range or phrasing."
        )

    lines: List[str] = []
    lines.append("Recent forklift-relevant Indiana projects (web search hits):")

    for i, item in enumerate(items[:15], start=1):
        title = item.get("project_name") or item.get("title") or "Untitled"
        snippet = (item.get("snippet") or "").strip()
        url = item.get("url") or ""
        provider = item.get("provider") or ""
        year = item.get("timeline_year")
        stage = item.get("timeline_stage") or ""
        loc = item.get("location_label") or item.get("original_area_label") or "Indiana"
        ptype = item.get("project_type") or "Industrial / commercial project"

        lines.append(f"{i}. {title} — {loc}")
        meta_bits = [ptype]
        if provider:
            meta_bits.append(provider)
        if stage:
            if year:
                meta_bits.append(f"{stage} ({year})")
            else:
                meta_bits.append(stage)
        if meta_bits:
            lines.append("   " + " • ".join(meta_bits))
        if snippet:
            lines.append(f"   Snippet: {snippet}")
        if url:
            lines.append(f"   URL: {url}")

    return "\n".join(lines)


__all__ = [
    "search_indiana_developments",
    "render_developments_markdown",
]
