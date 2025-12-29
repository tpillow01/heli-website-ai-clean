"""
indiana_intel.py

Indiana developments / lead finder for Tynan / Heli AI.

What this module does well:
- Works for BOTH "facility/news" searches (new DCs, expansions, hiring) and
  "planning/zoning" searches (plan commission / BZA / hearing examiner / agenda packets).
- Uses Google Programmable Search Engine (Custom Search JSON API) for retrieval.
- Normalizes results into structured dicts for your chat layer.

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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

# ---------------------------------------------------------------------------
# Config: Google CSE
# ---------------------------------------------------------------------------

GOOGLE_CSE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"
GOOGLE_CSE_KEY = os.environ.get("GOOGLE_CSE_KEY")
GOOGLE_CSE_CX = os.environ.get("GOOGLE_CSE_CX")

# Facility / industrial keywords (good for news + prospecting)
BASE_FACILITY_KEYWORDS = (
    '(warehouse OR "distribution center" OR "distribution facility" OR '
    '"distribution hub" OR logistics OR "logistics center" OR '
    '"logistics facility" OR "logistics hub" OR "fulfillment center" OR '
    '"fulfillment facility" OR "industrial park" OR "business park" OR '
    '"industrial complex" OR "manufacturing plant" OR "manufacturing facility" OR '
    'plant OR factory OR "production plant" OR "assembly plant" OR '
    '"cold storage" OR "spec building" OR "truck terminal" OR facility OR 3PL)'
)

# Planning / zoning keywords (good for agendas, packets, staff reports)
PLANNING_KEYWORDS = (
    '("plan commission" OR "area plan commission" OR "planning commission" OR '
    '"board of zoning appeals" OR BZA OR "hearing examiner" OR "hearing officer" OR '
    '"metropolitan development commission" OR MDC OR agenda OR minutes OR docket OR '
    'petition OR rezoning OR rezone OR PUD OR "development plan" OR "site plan" OR '
    '"staff report" OR "public hearing" OR ordinance OR variance OR '
    '"special exception" OR "zoning map" OR "primary plat" OR "secondary plat" OR '
    '"concept plan" OR "zoning case")'
)

# Industrial signals inside planning docs (we score these)
INDUSTRIAL_SIGNALS = [
    "industrial",
    "warehouse",
    "distribution",
    "logistics",
    "cold storage",
    "spec building",
    "truck terminal",
    "manufacturing",
    "plant",
    "3pl",
    "fulfillment",
    # common zoning hints
    "i-1",
    "i-2",
    "i-3",
    "i-4",
    "i1",
    "i2",
    "i3",
    "i4",
    "industrial district",
    "light industrial",
    "heavy industrial",
]

# Common platform/page markers for agendas/packets
AGENDA_PLATFORMS = [
    "agendacenter",
    "viewfile",
    "documentcenter",
    "legistar",
    "municode",
    "granicus",
    "minutes",
    "agenda",
    "packet",
    "meeting",
    "hearing",
]

_STOPWORDS = {
    "what", "are", "there", "any", "new", "or", "in", "the", "last", "month", "months",
    "recent", "recently", "project", "projects", "have", "has", "been", "announced",
    "announcement", "for", "about", "on", "of", "a", "an", "county", "indiana",
    "coming", "to", "area", "city", "kind", "sort", "type", "planned", "plan",
    "announce", "expanded", "expansion", "hiring", "jobs",
}

# Obvious non-prospect noise
_PROJECT_NEGATIVE_TEXT = [
    "visit", "tourism", "visitors bureau", "parks and recreation", "park and recreation",
    "shopping center", "outlet", "mall", "hotel", "resort", "casino", "water park",
    "museum", "library", "stadium", "arena", "sports complex", "golf",
    "apartments", "apartment", "housing development", "subdivision", "condo",
    "senior living", "assisted living", "retirement community",
    "elementary school", "middle school", "high school", "university", "college",
    "hospital", "medical center", "clinic", "behavioral health", "mental health",
    "church", "ministry",
]

_PROJECT_NEGATIVE_URL = [
    "facebook.com", "instagram.com", "twitter.com", "x.com", "youtube.com",
    "tripadvisor.com",
]

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _lower(s: Any) -> str:
    return str(s or "").lower()

def _slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9\s-]", "", _lower(s))
    s = re.sub(r"\s+", "-", s).strip("-")
    return s

def _is_planning_query(q: str) -> bool:
    """Detect if user is asking for agendas/minutes/rezoning/site plan types of things."""
    t = _lower(q)
    triggers = [
        "plan commission", "area plan", "planning commission", "bza", "board of zoning",
        "agenda", "minutes", "packet", "staff report", "rezoning", "rezone", "pud",
        "petition", "site plan", "development plan", "ordinance", "variance",
        "hearing examiner", "hearing officer", "metropolitan development commission", "mdc",
    ]
    return any(w in t for w in triggers)

def _extract_geo_hint(q: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract (city, county) from a natural-language question.

    - Finds 'X County' → county='X County'
    - Grabs a city after 'in / near / around' phrases.
    - Handles "Indianapolis (Marion County)" style.
    """
    if not q:
        return (None, None)

    text = q.strip()

    m_county = re.search(r"\b([A-Za-z]+)\s+County\b", text)
    county = f"{m_county.group(1).strip()} County" if m_county else None

    # "Indianapolis (Marion County)" or "Indianapolis - Marion County"
    m_paren = re.search(r"\b([A-Za-z][A-Za-z\s]+?)\s*\(\s*([A-Za-z]+)\s+County\s*\)", text)
    if m_paren:
        city = m_paren.group(1).strip()
        county = f"{m_paren.group(2).strip()} County"
        return (city, county)

    # City: look for "in Plainfield", "around Whitestown, IN", etc.
    m_city = re.search(r"\b(?:in|around|near)\s+([A-Za-z\s]+?)(?:,|\?|\.|$)", text)
    city = None
    if m_city:
        raw = m_city.group(1).strip()
        raw = re.sub(r"\b(Indiana|IN)\b\.?", "", raw, flags=re.I).strip()
        if raw:
            city = raw

    return (city, county)

def _parse_date_from_pagemap(it: Dict[str, Any]) -> Optional[datetime]:
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
    if not days or days <= 0:
        return None
    if days <= 31:
        return f"d{days}"
    months = max(1, int(round(days / 30)))
    if months <= 24:
        return f"m{months}"
    return None

def _google_cse_search(query: str, max_results: int = 30, days: Optional[int] = None) -> List[Dict[str, Any]]:
    if not GOOGLE_CSE_KEY or not GOOGLE_CSE_CX:
        log.warning(
            "GOOGLE_CSE_KEY or GOOGLE_CSE_CX not set; returning empty result list. "
            f"GOOGLE_CSE_KEY present={bool(GOOGLE_CSE_KEY)}, GOOGLE_CSE_CX present={bool(GOOGLE_CSE_CX)}"
        )
        return []

    date_restrict = _days_to_date_restrict(days)
    base_params = {
        "key": GOOGLE_CSE_KEY,
        "cx": GOOGLE_CSE_CX,
        "q": query,
        "num": 10,
        # NOTE: CSE "sort=date" depends on engine config; we keep it but still rank ourselves.
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
            out.append({"title": title, "snippet": snippet, "url": url, "provider": provider, "date": dt})
            if len(out) >= max_results:
                break

        start += 10

    return out

def _dedupe_by_url(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for it in items:
        u = (it.get("url") or "").strip()
        if not u:
            continue
        key = u.split("#")[0]
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out

# ---------------------------------------------------------------------------
# Filters / scoring
# ---------------------------------------------------------------------------

_FACILITY_POSITIVE = [
    "warehouse",
    "distribution center",
    "distribution facility",
    "distribution hub",
    "fulfillment center",
    "fulfillment facility",
    "logistics center",
    "logistics facility",
    "logistics hub",
    "industrial park",
    "business park",
    "industrial complex",
    "manufacturing plant",
    "manufacturing facility",
    "production plant",
    "assembly plant",
    "factory",
    "cold storage",
    "spec building",
    "truck terminal",
    "3pl",
    "third-party logistics",
    "third party logistics",
]

def _looks_like_facility_hit(title: str, snippet: str, url: str) -> bool:
    text = _lower(f"{title} {snippet}")
    url_l = _lower(url or "")

    for bad in _PROJECT_NEGATIVE_URL:
        if bad in url_l:
            return False

    for neg in _PROJECT_NEGATIVE_TEXT:
        if neg in text:
            return False

    has_positive = any(pos in text for pos in _FACILITY_POSITIVE)
    return bool(has_positive)

def _looks_like_planning_doc(title: str, snippet: str, url: str) -> bool:
    """
    Accept agendas/packets/minutes/staff reports even if they don't say 'warehouse' in snippet.
    Then we score for industrial signals.
    """
    text = _lower(f"{title} {snippet}")
    url_l = _lower(url or "")

    for bad in _PROJECT_NEGATIVE_URL:
        if bad in url_l:
            return False

    for neg in _PROJECT_NEGATIVE_TEXT:
        if neg in text:
            return False

    # Must look like a meeting/agenda doc OR contain planning terms
    looks_platform = any(m in url_l for m in AGENDA_PLATFORMS) or any(m in text for m in AGENDA_PLATFORMS)
    has_planning = any(k in text for k in ["agenda", "minutes", "packet", "staff report", "petition", "rezon", "hearing", "commission", "bza", "plan commission", "mdc"])
    if not (looks_platform or has_planning):
        return False

    # We don't require industrial signals to *accept*; we use them to rank.
    return True

def _geo_match_scores(title: str, snippet: str, city: Optional[str], county: Optional[str]) -> Tuple[int, bool, bool]:
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

def _score_planning_industrial(title: str, snippet: str, url: str) -> int:
    text = _lower(f"{title} {snippet} {url}")
    score = 0
    for s in INDUSTRIAL_SIGNALS:
        if s in text:
            score += 2
    # PDFs/packets are usually the good stuff
    if ".pdf" in text:
        score += 1
    # bonus for staff report mention
    if "staff report" in text:
        score += 1
    return score

def _compute_facility_score(title: str, snippet: str) -> int:
    text = _lower(f"{title} {snippet}")
    score = 0
    for pos in _FACILITY_POSITIVE:
        if pos in text:
            score += 2
    if re.search(r"\b\d{2,4}[,\d]{0,4}\s*(square[-\s]?feet|sq\.?\s*ft|sf)\b", text):
        score += 2
    if re.search(r"\b\d{2,5}\s+(new\s+)?jobs\b", text):
        score += 1
    return score

def _infer_project_type(title: str, snippet: str, is_planning: bool) -> str:
    text = _lower(title + " " + snippet)
    if is_planning:
        return "planning / zoning filing"
    if any(w in text for w in ("warehouse", "distribution center", "distribution facility", "fulfillment center", "cold storage")):
        return "warehouse / logistics facility"
    if any(w in text for w in ("manufacturing plant", "manufacturing facility", "production plant", "assembly plant", "factory")):
        return "manufacturing plant"
    if any(w in text for w in ("industrial park", "business park", "industrial complex")):
        return "business / industrial park"
    return "Industrial / commercial project"

# ---------------------------------------------------------------------------
# Query builders
# ---------------------------------------------------------------------------

def _tail_from_user_q(user_q: str, limit: int = 7) -> str:
    cleaned = re.sub(r"[“”\"']", " ", user_q or "")
    tokens = re.findall(r"[A-Za-z0-9]+", cleaned)
    extra: List[str] = []
    for tok in tokens:
        tl = tok.lower()
        if tl in _STOPWORDS:
            continue
        extra.append(tok)
    return " ".join(extra[:limit])

def _build_facility_query(user_q: str, city: Optional[str], county: Optional[str]) -> str:
    parts: List[str] = ["Indiana", BASE_FACILITY_KEYWORDS]
    if county:
        parts.append(f'"{county}"')
    if city:
        parts.append(f'"{city}"')
    tail = _tail_from_user_q(user_q, limit=6)
    if tail:
        parts.append(tail)
    q = " ".join(parts)
    log.info("Google CSE facility query: %s", q)
    return q

def _build_planning_query(user_q: str, city: Optional[str], county: Optional[str]) -> str:
    parts: List[str] = ["Indiana", PLANNING_KEYWORDS]

    # Keep geo anchors tight
    if county:
        parts.append(f'"{county}"')
    if city:
        parts.append(f'"{city}"')

    # Industrial relevance helps the agenda search
    parts.append("(industrial OR warehouse OR distribution OR logistics OR manufacturing OR \"cold storage\" OR \"industrial park\")")

    # Force common hosting patterns
    parts.append('("AgendaCenter" OR "ViewFile" OR "DocumentCenter" OR "Legistar" OR municode OR granicus OR packet OR agenda OR minutes)')

    tail = _tail_from_user_q(user_q, limit=5)
    if tail:
        parts.append(tail)

    q = " ".join(parts)
    log.info("Google CSE planning query: %s", q)
    return q

def _build_planning_site_restrict(city: Optional[str], county: Optional[str]) -> str:
    """
    A light site restrict (not perfect) to prioritize government/municipal sources.
    We do NOT hard-restrict to one domain because Indiana counties vary wildly.
    """
    sites = [
        "site:*.in.gov",
        "site:in.gov",
        "site:*.in.us",
        "site:in.us",
        "site:*.us",
        # municode is common for agendas/minutes
        "site:meetings.municode.com",
        "site:*municodemeetings.com",
        # Indy/Marion often comes from these
        "site:indy.gov",
        "site:indianapolis.granicus.com",
        "site:calendar.indy.gov",
    ]
    # If county is provided, add some common county-domain guesses (not required)
    if county:
        cslug = _slug(county.replace(" county", ""))
        sites.extend([
            f"site:co.{cslug}.in.us",
            f"site:{cslug}county.in.gov",
            f"site:{cslug}county.in.us",
        ])
    return "(" + " OR ".join(sites) + ")"

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize(
    raw_items: List[Dict[str, Any]],
    city: Optional[str],
    county: Optional[str],
    user_q: str,
    source_tier: str,
    is_planning: bool,
) -> List[Dict[str, Any]]:
    original_area_label = county or city or "Indiana"
    projects: List[Dict[str, Any]] = []

    for it in raw_items:
        title = it.get("title") or ""
        snippet = it.get("snippet") or ""
        url = it.get("url") or ""
        provider = it.get("provider") or ""
        dt = it.get("date")

        if is_planning:
            if not _looks_like_planning_doc(title, snippet, url):
                continue
            geo_score, match_city, match_county = _geo_match_scores(title, snippet, city, county)
            industrial_score = _score_planning_industrial(title, snippet, url)
            project_type = _infer_project_type(title, snippet, is_planning=True)

            projects.append({
                "project_name": title or "Untitled document",
                "company": None,
                "project_type": project_type,
                "scope": "local" if geo_score > 0 else "statewide",
                "location_label": (county or city or "Indiana") if geo_score > 0 else "Indiana",
                "original_area_label": original_area_label,
                "forklift_score": max(1, min(5, 1 + industrial_score // 3)),  # map industrial score to 1..5
                "forklift_label": "Planning doc (ranked for industrial relevance)",
                "geo_match_score": geo_score,
                "match_city": match_city,
                "match_county": match_county,
                "sqft": None,
                "jobs": None,
                "investment": None,
                "timeline_stage": "agenda/minutes" if url.lower().endswith(".pdf") or "agenda" in _lower(title) else "planning filing",
                "timeline_year": dt.year if isinstance(dt, datetime) else None,
                "raw_date": dt,
                "url": url,
                "provider": provider,
                "snippet": snippet,
                "source_tier": source_tier,
                "result_mode": "planning",
                "industrial_signal_score": industrial_score,
            })
        else:
            if not _looks_like_facility_hit(title, snippet, url):
                continue
            geo_score, match_city, match_county = _geo_match_scores(title, snippet, city, county)
            facility_score = _compute_facility_score(title, snippet)
            project_type = _infer_project_type(title, snippet, is_planning=False)

            projects.append({
                "project_name": title or "Untitled project",
                "company": None,
                "project_type": project_type,
                "scope": "local" if geo_score > 0 else "statewide",
                "location_label": (county or city or "Indiana") if geo_score > 0 else "Indiana",
                "original_area_label": original_area_label,
                "forklift_score": max(1, min(5, 1 + facility_score // 3)),
                "forklift_label": "Facility/news hit",
                "geo_match_score": geo_score,
                "match_city": match_city,
                "match_county": match_county,
                "sqft": None,
                "jobs": None,
                "investment": None,
                "timeline_stage": "announcement" if isinstance(dt, datetime) else "not specified in snippet",
                "timeline_year": dt.year if isinstance(dt, datetime) else None,
                "raw_date": dt,
                "url": url,
                "provider": provider,
                "snippet": snippet,
                "source_tier": source_tier,
                "result_mode": "facility",
                "industrial_signal_score": None,
            })

    return projects

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_indiana_developments(user_q: str, days: int = 365, max_items: int = 30) -> List[Dict[str, Any]]:
    """
    Main entrypoint.

    If the user question looks like a planning/zoning query:
      - Prioritize planning-doc retrieval (agenda packets, staff reports, minutes).
    Otherwise:
      - Prioritize facility/news retrieval.

    Returns normalized dicts.
    """
    city, county = _extract_geo_hint(user_q)
    is_planning = _is_planning_query(user_q)

    log.info("Geo hint: city=%s county=%s planning=%s", city, county, is_planning)

    projects: List[Dict[str, Any]] = []

    # ---- Tier 1: local (planning or facility) ----
    if is_planning:
        site_bias = _build_planning_site_restrict(city, county)
        q1 = _build_planning_query(user_q, city, county)
        query_local = f"{site_bias} {q1}"
        raw_local = _google_cse_search(query_local, max_results=max_items, days=days)
        raw_local = _dedupe_by_url(raw_local)
        if raw_local:
            projects = _normalize(raw_local, city, county, user_q, source_tier="local", is_planning=True)
    else:
        query_local = _build_facility_query(user_q, city, county)
        raw_local = _google_cse_search(query_local, max_results=max_items, days=days)
        raw_local = _dedupe_by_url(raw_local)
        if raw_local:
            projects = _normalize(raw_local, city, county, user_q, source_tier="local", is_planning=False)

    # ---- Tier 2: statewide fallback (same mode, looser geo) ----
    if not projects:
        log.info("No local results; trying statewide tier")
        if is_planning:
            site_bias = _build_planning_site_restrict(city=None, county=None)
            q2 = _build_planning_query(user_q, city=None, county=None)
            query_statewide = f"{site_bias} {q2}"
            raw_statewide = _google_cse_search(query_statewide, max_results=max_items, days=days)
            raw_statewide = _dedupe_by_url(raw_statewide)
            if raw_statewide:
                projects = _normalize(raw_statewide, None, None, user_q, source_tier="statewide", is_planning=True)
        else:
            query_statewide = _build_facility_query(user_q, city=None, county=None)
            raw_statewide = _google_cse_search(query_statewide, max_results=max_items, days=days)
            raw_statewide = _dedupe_by_url(raw_statewide)
            if raw_statewide:
                projects = _normalize(raw_statewide, None, None, user_q, source_tier="statewide", is_planning=False)

    # ---- Tier 3: mode-specific generic fallback ----
    if not projects:
        log.info("No results; running generic fallback")
        if is_planning:
            generic = f'{(county or "Indiana")} {PLANNING_KEYWORDS} (industrial OR warehouse OR logistics OR manufacturing) ("AgendaCenter" OR "ViewFile" OR municode OR agenda OR minutes OR packet)'
            site_bias = _build_planning_site_restrict(city=None, county=county)
            query_fallback = f"{site_bias} {generic}"
            raw = _google_cse_search(query_fallback, max_results=max_items, days=max(days, 730))
            raw = _dedupe_by_url(raw)
            if raw:
                projects = _normalize(raw, None, county, generic, source_tier="fallback", is_planning=True)
        else:
            generic = f'Indiana {BASE_FACILITY_KEYWORDS} (announced OR expansion OR groundbreaking OR "now hiring")'
            query_fallback = generic
            raw = _google_cse_search(query_fallback, max_results=max_items, days=max(days, 730))
            raw = _dedupe_by_url(raw)
            if raw:
                projects = _normalize(raw, None, None, generic, source_tier="fallback", is_planning=False)

    if not projects:
        return []

    # ---- Ranking ----
    now = datetime.utcnow()

    def _sort_key(p: Dict[str, Any]) -> Tuple[int, int, int]:
        geo = int(p.get("geo_match_score") or 0)
        score = int(p.get("forklift_score") or 0)
        # planning mode: boost industrial_signal_score
        bonus = int(p.get("industrial_signal_score") or 0)
        dt = p.get("raw_date")
        age_days = 9999
        if isinstance(dt, datetime):
            age_days = (now - dt).days
        # Sort: higher geo, higher score, higher bonus, newer
        return (-geo, -(score * 10 + bonus), age_days)

    projects.sort(key=_sort_key)
    return projects[:max_items]

def render_developments_markdown(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "No web results were found for that location and timeframe. Try adjusting the date range or phrasing."

    lines: List[str] = []
    lines.append("Industrial / logistics results (web search hits):")

    for i, item in enumerate(items[:15], start=1):
        title = item.get("project_name") or "Untitled"
        snippet = (item.get("snippet") or "").strip()
        url = item.get("url") or ""
        provider = item.get("provider") or ""
        year = item.get("timeline_year")
        stage = item.get("timeline_stage") or ""
        loc = item.get("location_label") or item.get("original_area_label") or "Indiana"
        ptype = item.get("project_type") or "Industrial / commercial project"
        score = item.get("forklift_score")
        geo_score = item.get("geo_match_score") or 0
        tier = item.get("source_tier") or "unknown"
        mode = item.get("result_mode") or "unknown"

        lines.append(f"{i}. {title} — {loc}")
        meta_bits = [ptype, f"Mode: {mode}"]
        if provider:
            meta_bits.append(provider)
        if score:
            meta_bits.append(f"Relevance {score}/5")
        if geo_score:
            meta_bits.append(f"Geo match {geo_score}/2")
        if tier:
            meta_bits.append(f"Source: {tier}")
        if stage:
            meta_bits.append(f"{stage}{f' ({year})' if year else ''}")

        lines.append("   " + " • ".join(meta_bits))
        if snippet:
            lines.append(f"   Snippet: {snippet}")
        if url:
            lines.append(f"   URL: {url}")

    return "\n".join(lines)

__all__ = ["search_indiana_developments", "render_developments_markdown"]
