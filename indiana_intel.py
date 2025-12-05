"""
indiana_intel.py

Indiana developments / lead finder for Tynan / Heli AI.

Responsibilities:
- Take a natural-language question (e.g. "new warehouses in Boone County in the last 90 days")
- Hit Google Custom Search JSON API (Programmable Search Engine) that is Indiana-biased
- Pull back industrial / logistics / manufacturing / commercial facilities
- Normalize results into simple dicts the chat layer can format
- Provide a plain-text summary for debugging

Environment variables required:
- GOOGLE_CSE_KEY : Google API key for Custom Search JSON API
- GOOGLE_CSE_CX  : Programmable Search Engine ID (cx) configured for Indiana
"""

from __future__ import annotations

import os
import re
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger("indiana_intel")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

# ---------------------------------------------------------------------------
# Config: Google CSE
# ---------------------------------------------------------------------------

GOOGLE_CSE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"
GOOGLE_CSE_KEY = os.environ.get("GOOGLE_CSE_KEY")
GOOGLE_CSE_CX = os.environ.get("GOOGLE_CSE_CX")

# Base industrial / logistics / commercial keywords
BASE_KEYWORDS = (
    '(warehouse OR "distribution center" OR "distribution facility" OR '
    '"distribution hub" OR logistics OR "logistics center" OR '
    '"logistics facility" OR "logistics hub" OR "fulfillment center" OR '
    '"industrial park" OR "business park" OR "industrial complex" OR '
    '"manufacturing plant" OR "manufacturing facility" OR plant OR factory '
    'OR "production plant" OR "assembly plant" OR "cold storage" OR facility)'
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
    - Tries to grab a city name after 'in ' / 'near ' / 'around ' if it looks like 'Plainfield' or 'Plainfield, IN'
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
        # Strip "IN" or "Indiana" if included
        raw = re.sub(r"\b(Indiana|IN)\b\.?", "", raw, flags=re.I).strip()
        if raw:
            city = raw

    return (city, county)


_STOPWORDS = {
    "what",
    "are",
    "there",
    "any",
    "new",
    "or",
    "in",
    "the",
    "last",
    "month",
    "months",
    "recent",
    "recently",
    "project",
    "projects",
    "have",
    "has",
    "been",
    "announced",
    "announcement",
    "for",
    "about",
    "on",
    "of",
    "a",
    "an",
    "county",
    "indiana",
    "logistics",
    "warehouse",
    "warehouses",
    "distribution",
    "center",
    "centers",
    "companies",
    "coming",
    "to",
    "area",
    "city",
    "kind",
    "sort",
    "type",
    "planned",
    "plan",
    "announce",
    "announced",
    "expanded",
    "expansion",
    "hiring",
    "jobs",
}


def _build_query(user_q: str, city: Optional[str], county: Optional[str]) -> str:
    """
    Build a Google CSE query anchored on industrial/commercial keywords,
    with optional city / county bias, plus a small keyword tail from the question.
    Always includes 'Indiana' to bias the search.
    """
    parts: List[str] = ["Indiana", BASE_KEYWORDS]

    if county:
        parts.append(f'"{county}"')
    if city:
        parts.append(f'"{city}"')

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
                try:
                    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                except Exception:
                    dt = None
                    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
                        try:
                            dt = datetime.strptime(raw[:10], fmt)
                            break
                        except Exception:
                            dt = None
                    if not dt:
                        continue
                if dt.tzinfo is not None:
                    dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
                return dt

    return None


def _days_to_date_restrict(days: Optional[int]) -> Optional[str]:
    """
    Map a days window into Google CSE dateRestrict syntax.
    - dN = last N days
    - mN = last N months
    """
    if not days or days <= 0:
        return None
    if days <= 31:
        return f"d{days}"
    months = max(1, int(round(days / 30)))
    if months <= 24:
        return f"m{months}"
    return None


def _google_cse_search(
    query: str,
    max_results: int = 30,
    days: Optional[int] = None,
) -> List[Dict[str, Any]]:
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

    date_restrict = _days_to_date_restrict(days)

    base_params = {
        "key": GOOGLE_CSE_KEY,
        "cx": GOOGLE_CSE_CX,
        "q": query,
        "num": 10,
        "sort": "date",
    }
    if date_restrict:
        base_params["dateRestrict"] = date_restrict

    out: List[Dict[str, Any]] = []
    start = 1

    while len(out) < max_results and start <= 91:
        params = dict(base_params)
        params["start"] = start

        try:
            resp = requests.get(GOOGLE_CSE_ENDPOINT, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.warning("Google CSE request failed (start=%s): %s", start, e)
            break

        items = data.get("items", []) or []
        log.info("Google CSE returned %s items at start=%s", len(items), start)
        if not items:
            break

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
                    "date": dt,
                }
            )

            if len(out) >= max_results:
                break

        if len(out) >= max_results:
            break

        start += 10

    return out

# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

_FORKLIFT_POSITIVE = [
    "warehouse",
    "distribution center",
    "distribution facility",
    "distribution hub",
    "fulfillment center",
    "fulfillment facility",
    "logistics center",
    "logistics facility",
    "logistics hub",
    "logistics park",
    "industrial park",
    "business park",
    "industrial complex",
    "manufacturing plant",
    "manufacturing facility",
    "production plant",
    "assembly plant",
    "factory",
    "cold storage",
    "3pl",
    "third-party logistics",
    "third party logistics",
]

# Obvious non-prospect noise
_PROJECT_NEGATIVE_TEXT = [
    "visit hendricks county",
    "visit indiana",
    "visit our town",
    "tourism",
    "visitors bureau",
    "shopping center",
    "outlet mall",
    "premium outlets",
    "outlets",
    "mall",
    "hotel",
    "resort",
    "casino",
    "water park",
    "amusement park",
    "museum",
    "library",
    "stadium",
    "arena",
    "sports complex",
    "golf",
    "park and recreation",
    "parks and recreation",
    "trailhead",
    "apartments",
    "apartment complex",
    "housing development",
    "subdivision",
    "condominiums",
    "condo",
    "senior living",
    "assisted living",
    "retirement community",
    "elementary school",
    "middle school",
    "high school",
    "university",
    "college",
    "campus",
    "hospital",
    "medical center",
    "clinic",
    "behavioral health",
    "mental health",
    "church",
    "ministry",
]

# Obvious junk domains
_PROJECT_NEGATIVE_URL = [
    "facebook.com",
    "instagram.com",
    "twitter.com",
    "x.com",
    "youtube.com",
    "tripadvisor.com",
]


def _looks_like_facility_hit(title: str, snippet: str, url: str) -> bool:
    """
    Decide whether a search hit is an industrial / forklift-relevant facility
    vs pure junk.

    Rules:
    - Must contain at least one forklift-positive keyword
      (warehouse, DC, logistics center, plant, industrial park, etc.).
    - Must NOT obviously be tourism/housing/hospital/social content.
    """
    text = _lower(f"{title} {snippet}")
    url_l = _lower(url or "")

    # Filter obvious junk by URL
    for bad in _PROJECT_NEGATIVE_URL:
        if bad in url_l:
            return False

    # Filter obvious junk by content
    for neg in _PROJECT_NEGATIVE_TEXT:
        if neg in text:
            return False

    # Must mention forklift-relevant facility type
    has_positive = any(pos in text for pos in _FORKLIFT_POSITIVE)
    if not has_positive:
        return False

    return True


def _compute_forklift_score(title: str, snippet: str, url: str) -> Tuple[int, str]:
    """
    Rough forklift relevance score 1–5, used for ranking only.
    """
    text = _lower(f"{title} {snippet}")
    score = 0

    # Count positive facility keywords
    for pos in _FORKLIFT_POSITIVE:
        if pos in text:
            score += 2

    # Size hints
    if re.search(r"\b\d{2,4}[,\d]{0,4}\s*(square[-\s]?feet|sq\.?\s*ft|sf)\b", text):
        score += 2

    # Jobs hints
    if re.search(r"\b\d{2,5}\s+(new\s+)?jobs\b", text):
        score += 1

    if score <= 1:
        numeric = 2
    elif score <= 3:
        numeric = 3
    elif score <= 5:
        numeric = 4
    else:
        numeric = 5

    label_map = {
        2: "Possible forklift-using facility",
        3: "Likely forklift-using facility",
        4: "Strong forklift-using facility",
        5: "Very strong forklift-using facility",
    }
    return numeric, label_map.get(numeric, "Likely forklift-using facility")


def _geo_match_scores(
    title: str,
    snippet: str,
    city: Optional[str],
    county: Optional[str],
) -> Tuple[int, bool, bool]:
    """
    Compute a simple geographic match score:
      - geo_match_score: 0..2
      - match_city: bool
      - match_county: bool
    """
    text = _lower(title + " " + snippet)
    match_city = False
    match_county = False

    if county:
        base = county.split()[0].lower()
        if base and base in text:
            match_county = True
        elif county.lower() in text:
            match_county = True

    if city:
        c = city.lower()
        if c and c in text:
            match_city = True

    if match_city and match_county:
        geo_score = 2
    elif match_city or match_county:
        geo_score = 1
    else:
        geo_score = 0

    return geo_score, match_city, match_county


def _infer_project_type(title: str, snippet: str) -> str:
    text = _lower(title + " " + snippet)
    if any(w in text for w in ("warehouse", "distribution center", "distribution facility", "fulfillment center")):
        return "warehouse / logistics facility"
    if any(w in text for w in ("logistics hub", "logistics park", "logistics center", "logistics facility")):
        return "warehouse / logistics facility"
    if any(w in text for w in ("manufacturing plant", "manufacturing facility", "production plant", "assembly plant", "factory")):
        return "manufacturing plant"
    if any(w in text for w in ("industrial park", "business park", "industrial complex")):
        return "business / industrial park"
    return "Industrial / commercial project"


def _normalize_projects(
    raw_items: List[Dict[str, Any]],
    city: Optional[str],
    county: Optional[str],
    user_q: str,
    source_tier: str,
) -> List[Dict[str, Any]]:
    """
    Convert raw CSE results into the structured dicts used by the chat layer.

    We try to be loose enough to not come back totally empty:
    - Reject obvious tourism/housing/hospitals/social junk
    - Require at least one industrial / warehouse keyword
    - Keep everything that passes that, but rank by forklift_score + geo match + recency
    """
    original_area_label = county or city or "Indiana"

    projects: List[Dict[str, Any]] = []
    for it in raw_items:
        title = it.get("title") or ""
        snippet = it.get("snippet") or ""
        url = it.get("url") or ""
        provider = it.get("provider") or ""
        dt = it.get("date")

        if not _looks_like_facility_hit(title, snippet, url):
            continue

        forklift_score, forklift_label = _compute_forklift_score(title, snippet, url)
        geo_score, match_city, match_county = _geo_match_scores(title, snippet, city, county)
        project_type = _infer_project_type(title, snippet)

        if match_county:
            location_label = county or "Indiana"
        elif match_city:
            location_label = city or "Indiana"
        else:
            location_label = "Indiana"

        timeline_year: Optional[int] = dt.year if isinstance(dt, datetime) else None
        timeline_stage = "announcement" if timeline_year else "not specified in snippet"

        projects.append(
            {
                "project_name": title or "Untitled project",
                "company": None,
                "project_type": project_type,
                "scope": "local" if geo_score > 0 else "statewide",
                "location_label": location_label,
                "original_area_label": original_area_label,
                "forklift_score": forklift_score,
                "forklift_label": forklift_label,
                "geo_match_score": geo_score,
                "match_city": match_city,
                "match_county": match_county,
                "sqft": None,
                "jobs": None,
                "investment": None,
                "timeline_stage": timeline_stage,
                "timeline_year": timeline_year,
                "raw_date": dt,
                "url": url,
                "provider": provider,
                "snippet": snippet,
                "source_tier": source_tier,  # "local", "statewide", "fallback"
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

    Strategy (tiers):
      1) "local":  city/county-aware query based on the user's question.
      2) "statewide": same question, but without city/county constraint.
      3) "fallback": county-aware generic search like
         "new warehouses / DCs / plants in and around Boone County, Indiana".

    Returns a list of normalized project dicts.
    """
    city, county = _extract_geo_hint(user_q)

    projects: List[Dict[str, Any]] = []

    # ── Tier 1: local ─────────────────────────────────────────────────────────
    query_local = _build_query(user_q, city, county)
    raw_local = _google_cse_search(query_local, max_results=max_items, days=days)
    if raw_local:
        projects = _normalize_projects(raw_local, city, county, user_q, source_tier="local")

    # ── Tier 2: statewide ─────────────────────────────────────────────────────
    if not projects:
        log.info("No forklift-relevant local projects; trying statewide tier")
        query_statewide = _build_query(user_q, city=None, county=None)
        raw_statewide = _google_cse_search(query_statewide, max_results=max_items, days=days)
        if raw_statewide:
            projects = _normalize_projects(raw_statewide, None, None, user_q, source_tier="statewide")

    # ── Tier 3: county-aware fallback ─────────────────────────────────────────
    if not projects:
        log.info("No forklift-relevant hits for user query; running county-aware fallback")
        if county:
            generic_q = (
                f"new or expanded warehouses, distribution centers, logistics facilities, "
                f"manufacturing plants, and industrial parks in and around {county}, Indiana "
                f"in the last few years"
            )
            query_fallback = _build_query(generic_q, city=None, county=county)
        else:
            generic_q = (
                "new or expanded warehouses, distribution centers, logistics facilities, "
                "manufacturing plants, and industrial parks in Indiana in the last few years"
            )
            query_fallback = _build_query(generic_q, city=None, county=None)

        raw_fallback = _google_cse_search(query_fallback, max_results=max_items, days=max(days, 730))
        if raw_fallback:
            projects = _normalize_projects(raw_fallback, None, county, generic_q, source_tier="fallback")

    if not projects:
        return []

    # ── Rank by geo match, forklift relevance, and recency ────────────────────
    now = datetime.utcnow()

    def _sort_key(p: Dict[str, Any]) -> Tuple[int, int, int]:
        geo = p.get("geo_match_score") or 0
        score = p.get("forklift_score") or 0
        dt = p.get("raw_date")
        age_days = 9999
        if isinstance(dt, datetime):
            age_days = (now - dt).days
        # Sort: higher geo, higher forklift_score, newer (smaller age_days)
        return (-geo, -score, age_days)

    projects.sort(key=_sort_key)
    return projects[:max_items]


def render_developments_markdown(items: List[Dict[str, Any]]) -> str:
    """
    Simple markdown-ish debug formatter.
    """
    if not items:
        return (
            "No web results were found for that location and timeframe. "
            "Try adjusting the date range or phrasing."
        )

    lines: List[str] = []
    lines.append("Industrial / logistics projects (web search hits):")

    for i, item in enumerate(items[:15], start=1):
        title = item.get("project_name") or "Untitled"
        snippet = (item.get("snippet") or "").strip()
        url = item.get("url") or ""
        provider = item.get("provider") or ""
        year = item.get("timeline_year")
        stage = item.get("timeline_stage") or ""
        loc = item.get("location_label") or item.get("original_area_label") or "Indiana"
        ptype = item.get("project_type") or "Industrial / commercial project"
        forklift_score = item.get("forklift_score")
        forklift_label = item.get("forklift_label") or ""
        geo_score = item.get("geo_match_score") or 0
        tier = item.get("source_tier") or "unknown"

        lines.append(f"{i}. {title} — {loc}")

        meta_bits = [ptype]
        if provider:
            meta_bits.append(provider)
        if forklift_score:
            meta_bits.append(f"Forklift relevance {forklift_score}/5")
        if geo_score:
            meta_bits.append(f"Geo match {geo_score}/2")
        if tier:
            meta_bits.append(f"Source: {tier}")
        if stage:
            if year:
                meta_bits.append(f"{stage} ({year})")
            else:
                meta_bits.append(stage)
        if forklift_label:
            meta_bits.append(forklift_label)

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
