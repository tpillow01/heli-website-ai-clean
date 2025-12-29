"""
indiana_intel.py

Indiana developments / lead finder for Tynan / Heli AI.

Key improvement:
- Adds a dedicated "local government" tier (plan commission / zoning / DP-SP / PUD / minutes / agendas)
  because county-specific development info often lives in agendas/minutes PDFs and case lists that do
  NOT include warehouse/manufacturing keywords in Google snippets.

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

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    )
}

# Base industrial / logistics / commercial keywords (news-y / business-y)
BASE_KEYWORDS = (
    '(warehouse OR "distribution center" OR "distribution facility" OR '
    '"distribution hub" OR logistics OR "logistics center" OR '
    '"logistics facility" OR "logistics hub" OR "fulfillment center" OR '
    '"industrial park" OR "business park" OR "industrial complex" OR '
    '"manufacturing plant" OR "manufacturing facility" OR plant OR factory '
    'OR "production plant" OR "assembly plant" OR "cold storage" OR facility)'
)

# Local-government vocabulary that often appears in agendas/minutes/case lists
LOCAL_GOV_KEYWORDS = (
    '("plan commission" OR "area plan commission" OR "board of zoning appeals" OR BZA OR '
    'agenda OR minutes OR docket OR petition OR rezoning OR rezone OR PUD OR '
    '"development plan" OR "site plan" OR "primary plat" OR "secondary plat" OR '
    '"staff report" OR "public hearing")'
)

# Some counties/towns host the actionable documents; expand this list over time.
# (These are not required to be perfect; they're a strong starting point.)
COUNTY_GOV_SITES: Dict[str, List[str]] = {
    # Boone County + key towns
    "boone": [
        "boonecounty.in.gov",
        "whitestown.in.gov",
        "lebanon.in.gov",
        "zionsville-in.gov",
    ],
    # Hendricks County + key towns
    "hendricks": [
        "co.hendricks.in.us",
        "townofplainfield.com",
        "brownsburg.org",
        "avonindiana.gov",
        "danvilleindiana.org",
    ],
}

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _lower(s: Any) -> str:
    return str(s or "").lower()


def _extract_geo_hint(q: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract (city, county) from question text.
    Fix: prevents capturing "Boone County in the last 180 days" as a city.
    """
    if not q:
        return (None, None)
    text = q.strip()

    m_county = re.search(r"\b([A-Za-z]+)\s+County\b", text)
    county = f"{m_county.group(1).strip()} County" if m_county else None

    m_city = re.search(
        r"\b(?:in|around|near)\s+([A-Za-z][A-Za-z\s]{1,40}?)(?:,|\?|\.|$)",
        text,
        flags=re.I,
    )
    city = None
    if m_city:
        raw = m_city.group(1).strip()
        raw = re.sub(r"\b(Indiana|IN)\b\.?", "", raw, flags=re.I).strip()
        # Reject timeframe captures
        if re.search(r"\b(last|past|days|months|weeks|years|since|recent)\b", raw, flags=re.I):
            raw = ""
        # Reject "X County" as a city
        if re.search(r"\bCounty\b", raw, flags=re.I):
            raw = ""
        if raw:
            city = raw

    return (city, county)


_STOPWORDS = {
    "what", "are", "there", "any", "new", "or", "in", "the", "last", "month", "months",
    "recent", "recently", "project", "projects", "have", "has", "been", "announced",
    "announcement", "for", "about", "on", "of", "a", "an", "county", "indiana", "logistics",
    "warehouse", "warehouses", "distribution", "center", "centers", "companies", "coming",
    "to", "area", "city", "kind", "sort", "type", "planned", "plan", "announce", "expanded",
    "expansion", "hiring", "jobs",
}


def _build_query(user_q: str, city: Optional[str], county: Optional[str]) -> str:
    """
    Standard (news/business) query for industrial facilities.
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
        if tok.lower() in _STOPWORDS:
            continue
        extra_tokens.append(tok)
    if extra_tokens:
        parts.append(" ".join(extra_tokens[:6]))

    query = " ".join(parts)
    log.info("Google CSE query: %s", query)
    return query


def _build_local_gov_query(county: Optional[str], city: Optional[str]) -> Optional[str]:
    """
    County/city-local government query:
    - Forces results to official county/town sites (where agendas/minutes/case lists live)
    - Uses zoning/plan-commission vocabulary rather than warehouse vocabulary
    """
    sites: List[str] = []
    if county:
        key = county.split()[0].lower()
        sites = COUNTY_GOV_SITES.get(key, [])

    # If we don't know the county or we don't have sites for it, still try a general .in.gov/.in.us approach.
    # (This is weaker but better than nothing.)
    if not sites:
        site_block = '(site:in.gov OR site:in.us OR site:*.in.gov OR site:*.in.us)'
    else:
        site_block = "(" + " OR ".join([f"site:{s}" for s in sites]) + ")"

    geo_bits: List[str] = []
    if county:
        geo_bits.append(f'"{county}"')
    if city:
        geo_bits.append(f'"{city}"')

    geo_clause = ""
    if geo_bits:
        geo_clause = "(" + " OR ".join(geo_bits) + ")"

    # We include “industrial/warehouse/manufacturing” softly (OR block) but we do NOT require it.
    industrial_soft = '(industrial OR warehouse OR "distribution" OR logistics OR manufacturing OR "cold storage" OR "spec building")'

    q = f"{site_block} {LOCAL_GOV_KEYWORDS} {geo_clause} {industrial_soft}".strip()
    log.info("Google CSE local-gov query: %s", q)
    return q


def _parse_date_from_pagemap(it: Dict[str, Any]) -> Optional[datetime]:
    """
    Sniff publish/update date from CSE pagemap metatags.
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
    Google CSE supports:
      dN (days), wN (weeks), mN (months), yN (years)
    """
    if not days or days <= 0:
        return None
    if days <= 31:
        return f"d{days}"
    if days <= 180:
        weeks = max(1, int(round(days / 7)))
        return f"w{weeks}"
    months = max(1, int(round(days / 30)))
    if months <= 24:
        return f"m{months}"
    years = max(1, int(round(days / 365)))
    return f"y{years}"


def _google_cse_search(
    query: str,
    max_results: int = 30,
    days: Optional[int] = None,
    file_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Wrapper around Google Custom Search JSON API.
    """
    if not GOOGLE_CSE_KEY or not GOOGLE_CSE_CX:
        log.warning(
            "GOOGLE_CSE_KEY or GOOGLE_CSE_CX not set; returning empty list. "
            f"key_present={bool(GOOGLE_CSE_KEY)} cx_present={bool(GOOGLE_CSE_CX)}"
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
    if file_type:
        base_params["fileType"] = file_type

    out: List[Dict[str, Any]] = []
    start = 1

    while len(out) < max_results and start <= 91:
        params = dict(base_params)
        params["start"] = start

        try:
            resp = requests.get(
                GOOGLE_CSE_ENDPOINT,
                params=params,
                headers=REQUEST_HEADERS,
                timeout=12,
            )
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

# Local government terms (for allowing plan commission docs through)
_LOCAL_GOV_POSITIVE = [
    "plan commission",
    "area plan commission",
    "board of zoning appeals",
    "bza",
    "agenda",
    "minutes",
    "docket",
    "petition",
    "rezoning",
    "rezone",
    "pud",
    "development plan",
    "site plan",
    "primary plat",
    "secondary plat",
    "staff report",
    "public hearing",
]

_PROJECT_NEGATIVE_TEXT = [
    "visit ",
    "tourism",
    "visitors bureau",
    "shopping center",
    "outlet",
    "mall",
    "hotel",
    "resort",
    "casino",
    "museum",
    "library",
    "stadium",
    "arena",
    "sports complex",
    "apartments",
    "housing development",
    "subdivision",
    "condominiums",
    "senior living",
    "assisted living",
    "retirement community",
    "elementary school",
    "middle school",
    "high school",
    "university",
    "college",
    "hospital",
    "medical center",
    "clinic",
    "church",
    "ministry",
]

_PROJECT_NEGATIVE_URL = [
    "facebook.com",
    "instagram.com",
    "twitter.com",
    "x.com",
    "youtube.com",
    "tripadvisor.com",
]


def _looks_like_facility_hit(title: str, snippet: str, url: str, allow_local_gov: bool = False) -> bool:
    """
    Standard filter:
    - blocks obvious junk
    - requires a facility keyword

    Local-gov override:
    - allow plan commission docs even if "warehouse" isn't in snippet
    """
    text = _lower(f"{title} {snippet}")
    url_l = _lower(url or "")

    for bad in _PROJECT_NEGATIVE_URL:
        if bad in url_l:
            return False

    for neg in _PROJECT_NEGATIVE_TEXT:
        if neg in text:
            return False

    if allow_local_gov and any(k in text for k in _LOCAL_GOV_POSITIVE):
        return True

    return any(pos in text for pos in _FORKLIFT_POSITIVE)


def _compute_forklift_score(title: str, snippet: str) -> Tuple[int, str]:
    text = _lower(f"{title} {snippet}")
    score = 0

    for pos in _FORKLIFT_POSITIVE:
        if pos in text:
            score += 2

    if re.search(r"\b\d{2,4}[,\d]{0,4}\s*(square[-\s]?feet|sq\.?\s*ft|sf)\b", text):
        score += 2

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
    original_area_label = county or city or "Indiana"

    allow_local_gov = source_tier == "local_gov"

    projects: List[Dict[str, Any]] = []
    for it in raw_items:
        title = it.get("title") or ""
        snippet = it.get("snippet") or ""
        url = it.get("url") or ""
        provider = it.get("provider") or ""
        dt = it.get("date")

        if not _looks_like_facility_hit(title, snippet, url, allow_local_gov=allow_local_gov):
            continue

        forklift_score, forklift_label = _compute_forklift_score(title, snippet)
        geo_score, match_city, match_county = _geo_match_scores(title, snippet, city, county)
        project_type = _infer_project_type(title, snippet)

        if match_county:
            location_label = county or "Indiana"
        elif match_city:
            location_label = city or "Indiana"
        else:
            location_label = "Indiana"

        timeline_year: Optional[int] = dt.year if isinstance(dt, datetime) else None
        timeline_stage = "agenda/minutes" if allow_local_gov else ("announcement" if timeline_year else "not specified in snippet")

        projects.append(
            {
                "project_name": title or "Untitled project",
                "company": None,
                "project_type": project_type if not allow_local_gov else "planning / zoning filing",
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
                "source_tier": source_tier,
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
      1) local: city/county-aware industrial query
      2) local_gov: county/city site-targeted plan commission / zoning docs (PDF-heavy)
      3) statewide: broad Indiana industrial query (only if needed)
      4) fallback: generic Indiana fallback
    """
    city, county = _extract_geo_hint(user_q)

    projects: List[Dict[str, Any]] = []

    # ── Tier 1: local (news/business) ─────────────────────────────────────────
    query_local = _build_query(user_q, city, county)
    raw_local = _google_cse_search(query_local, max_results=max_items, days=days)
    if raw_local:
        projects = _normalize_projects(raw_local, city, county, user_q, source_tier="local")

    # ── Tier 2: local government docs (agendas/minutes/cases) ─────────────────
    if not projects:
        log.info("No forklift-relevant local projects; trying local government tier")
        q_gov = _build_local_gov_query(county=county, city=city)
        if q_gov:
            # PDFs are common; if your PSE indexes them, this is gold.
            raw_gov = _google_cse_search(q_gov, max_results=max_items, days=days, file_type="pdf")
            if raw_gov:
                projects = _normalize_projects(raw_gov, city, county, user_q, source_tier="local_gov")

            # If pdf-only returns nothing, try without fileType restriction
            if not projects:
                raw_gov2 = _google_cse_search(q_gov, max_results=max_items, days=days, file_type=None)
                if raw_gov2:
                    projects = _normalize_projects(raw_gov2, city, county, user_q, source_tier="local_gov")

    # ── Tier 3: statewide ─────────────────────────────────────────────────────
    if not projects:
        log.info("No hits yet; trying statewide tier")
        query_statewide = _build_query(user_q, city=None, county=None)
        raw_statewide = _google_cse_search(query_statewide, max_results=max_items, days=days)
        if raw_statewide:
            projects = _normalize_projects(raw_statewide, None, None, user_q, source_tier="statewide")

    # ── Tier 4: fallback ──────────────────────────────────────────────────────
    if not projects:
        log.info("No hits; running fallback")
        generic_q = (
            "new or expanded warehouses, distribution centers, logistics facilities, "
            "manufacturing plants, and industrial parks in Indiana in the last few years"
        )
        query_fallback = _build_query(generic_q, city=None, county=county)
        raw_fallback = _google_cse_search(query_fallback, max_results=max_items, days=max(days, 730))
        if raw_fallback:
            projects = _normalize_projects(raw_fallback, None, county, generic_q, source_tier="fallback")

    if not projects:
        return []

    # Rank by geo match, forklift relevance, and recency
    now = datetime.utcnow()

    def _sort_key(p: Dict[str, Any]) -> Tuple[int, int, int]:
        geo = p.get("geo_match_score") or 0
        score = p.get("forklift_score") or 0
        dt = p.get("raw_date")
        age_days = 9999
        if isinstance(dt, datetime):
            age_days = (now - dt).days
        return (-geo, -score, age_days)

    projects.sort(key=_sort_key)
    return projects[:max_items]


def render_developments_markdown(items: List[Dict[str, Any]]) -> str:
    if not items:
        return (
            "No web results were found for that location and timeframe.\n"
            "Tip: plan commission documents are often PDFs and may require site-targeted searching."
        )

    lines: List[str] = []
    lines.append("Industrial / logistics projects (web search hits):")

    for i, item in enumerate(items[:15], start=1):
        title = item.get("project_name") or "Untitled"
        snippet = (item.get("snippet") or "").strip()
        url = item.get("url") or ""
        provider = item.get("provider") or ""
        loc = item.get("location_label") or item.get("original_area_label") or "Indiana"
        ptype = item.get("project_type") or "Industrial / commercial project"
        forklift_score = item.get("forklift_score")
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
