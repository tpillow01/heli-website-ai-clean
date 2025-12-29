"""
indiana_intel.py

Indiana developments / lead finder for Tynan / Heli AI.

What this version fixes:
- Town-only queries (e.g., "Whitestown plan commission...") will NOT fall back to statewide filler.
- City queries are "hard-locked" like county queries (returns only city/county matches or empty).
- Infers county from common Indiana towns (Whitestown→Boone, Plainfield→Hendricks, etc.).
- Adds a dedicated Local Government tier for plan commission / zoning agendas/minutes/PDFs.
- Better dedupe + safer URL cleanup.
- Preserves the output schema used by your chat layer.

Environment variables required:
- GOOGLE_CSE_KEY : Google API key for Custom Search JSON API
- GOOGLE_CSE_CX  : Programmable Search Engine ID (cx) configured for Indiana
"""

from __future__ import annotations

import os
import re
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

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

# ---------------------------------------------------------------------------
# Keywords
# ---------------------------------------------------------------------------

# Business/news-style industrial keywords
BASE_KEYWORDS = (
    '(warehouse OR "distribution center" OR "distribution facility" OR '
    '"distribution hub" OR logistics OR "logistics center" OR '
    '"logistics facility" OR "logistics hub" OR "fulfillment center" OR '
    '"industrial park" OR "business park" OR "industrial complex" OR '
    '"manufacturing plant" OR "manufacturing facility" OR plant OR factory '
    'OR "production plant" OR "assembly plant" OR "cold storage" OR facility)'
)

# Local government / filings vocabulary (agendas/minutes/petitions/case lists)
LOCAL_GOV_KEYWORDS = (
    '("plan commission" OR "area plan commission" OR '
    '"board of zoning appeals" OR BZA OR agenda OR minutes OR docket OR '
    "petition OR rezoning OR rezone OR PUD OR "
    '"development plan" OR "site plan" OR '
    '"primary plat" OR "secondary plat" OR '
    '"staff report" OR "public hearing" OR ordinance OR variance)'
)

# Soft industrial terms for local-gov searches (NOT required)
INDUSTRIAL_SOFT = (
    '(industrial OR warehouse OR "distribution" OR logistics OR manufacturing OR '
    '"cold storage" OR "spec building" OR "truck terminal")'
)

# ---------------------------------------------------------------------------
# Geo intelligence
# ---------------------------------------------------------------------------

# City -> County inference (expand as you like)
CITY_TO_COUNTY: Dict[str, str] = {
    # Boone
    "whitestown": "Boone County",
    "lebanon": "Boone County",
    "zionsville": "Boone County",
    "jamestown": "Boone County",
    "thorntown": "Boone County",
    # Hendricks
    "plainfield": "Hendricks County",
    "brownsburg": "Hendricks County",
    "avon": "Hendricks County",
    "danville": "Hendricks County",
    "clayton": "Hendricks County",
    "pittsboro": "Hendricks County",
    # Hamilton
    "fishers": "Hamilton County",
    "carmel": "Hamilton County",
    "noblesville": "Hamilton County",
    "westfield": "Hamilton County",
    # Marion
    "indianapolis": "Marion County",
    "speedway": "Marion County",
    "lawrence": "Marion County",
    "beech grove": "Marion County",
}

# County -> likely government sites (for local-gov tier).
# If you don’t have a county here, the code falls back to broad in.gov/in.us searching.
COUNTY_GOV_SITES: Dict[str, List[str]] = {
    "boone": [
        "boonecounty.in.gov",
        "whitestown.in.gov",
        "lebanon.in.gov",
        "zionsville-in.gov",
    ],
    "hendricks": [
        "co.hendricks.in.us",
        "townofplainfield.com",
        "brownsburg.org",
        "avonindiana.gov",
        "danvilleindiana.org",
    ],
    "hamilton": [
        "hamiltoncounty.in.gov",
        "carmel.in.gov",
        "fishers.in.us",
        "noblesville.in.us",
        "westfield.in.gov",
    ],
    "marion": [
        "indy.gov",
        "indianapolis.in.gov",
        "speedwayin.gov",
        "cityoflawrence.org",
        "beechgrove.com",
    ],
}

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "what", "are", "there", "any", "new", "or", "in", "the", "last", "month", "months",
    "recent", "recently", "project", "projects", "have", "has", "been", "announced",
    "announcement", "for", "about", "on", "of", "a", "an", "county", "indiana", "logistics",
    "warehouse", "warehouses", "distribution", "center", "centers", "companies", "coming",
    "to", "area", "city", "kind", "sort", "type", "planned", "plan", "announce", "expanded",
    "expansion", "hiring", "jobs",
}

_FORKLIFT_POSITIVE = [
    "warehouse", "distribution center", "distribution facility", "distribution hub",
    "fulfillment center", "fulfillment facility",
    "logistics center", "logistics facility", "logistics hub", "logistics park",
    "industrial park", "business park", "industrial complex",
    "manufacturing plant", "manufacturing facility",
    "production plant", "assembly plant", "factory",
    "cold storage", "3pl", "third-party logistics", "third party logistics",
]

_LOCAL_GOV_POSITIVE = [
    "plan commission", "area plan commission", "board of zoning appeals", "bza",
    "agenda", "minutes", "docket", "petition", "rezoning", "rezone", "pud",
    "development plan", "site plan", "primary plat", "secondary plat",
    "staff report", "public hearing", "variance", "ordinance",
]

_PROJECT_NEGATIVE_TEXT = [
    "visit ", "tourism", "visitors bureau",
    "shopping center", "outlet", "mall",
    "hotel", "resort", "casino",
    "museum", "library", "stadium", "arena", "sports complex",
    "apartments", "housing development", "subdivision", "condominiums",
    "senior living", "assisted living", "retirement community",
    "elementary school", "middle school", "high school", "university", "college",
    "hospital", "medical center", "clinic",
    "church", "ministry",

    # common “not a development lead” pages:
    "our locations", "warehouse locations", "locations",
    "careers", "jobs", "job openings",
    "contact us",
]

_PROJECT_NEGATIVE_URL = [
    "facebook.com", "instagram.com", "twitter.com", "x.com", "youtube.com", "tripadvisor.com"
]

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _lower(s: Any) -> str:
    return str(s or "").lower()


def _canonicalize_url(url: str) -> str:
    """
    Remove common tracking params and normalize URL for dedupe.
    """
    try:
        u = urlparse(url)
        qs = [
            (k, v)
            for (k, v) in parse_qsl(u.query, keep_blank_values=True)
            if k.lower()
            not in {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "gclid", "fbclid"}
        ]
        new_query = urlencode(qs, doseq=True)
        clean = urlunparse((u.scheme, u.netloc, u.path, u.params, new_query, ""))
        return clean.rstrip("/")
    except Exception:
        return (url or "").rstrip("/")


def _infer_county_from_city(city: Optional[str]) -> Optional[str]:
    if not city:
        return None
    key = city.strip().lower()
    # allow matching "Beech Grove" and similar
    return CITY_TO_COUNTY.get(key)


def _extract_geo_hint(q: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract (city, county) from question text.
    Also infers county from city when possible.
    """
    if not q:
        return (None, None)
    text = q.strip()

    # County like "Boone County"
    m_county = re.search(r"\b([A-Za-z]+)\s+County\b", text)
    county = f"{m_county.group(1).strip()} County" if m_county else None

    # City: "in Whitestown" / "near Plainfield, IN"
    m_city = re.search(
        r"\b(?:in|around|near)\s+([A-Za-z][A-Za-z\s]{1,40}?)(?:,?\s*(?:IN|Indiana)\b|,|\?|\.|$)",
        text,
        flags=re.I,
    )
    city = None
    if m_city:
        raw = m_city.group(1).strip()
        raw = re.sub(r"\b(Indiana|IN)\b\.?", "", raw, flags=re.I).strip()
        if re.search(r"\b(last|past|days|months|weeks|years|since|recent)\b", raw, flags=re.I):
            raw = ""
        if re.search(r"\bCounty\b", raw, flags=re.I):
            raw = ""
        if raw:
            city = " ".join(raw.split()[:3]).strip()

    # County inference from city if missing
    if not county:
        inferred = _infer_county_from_city(city)
        if inferred:
            county = inferred

    log.info("Geo hint: city=%s county=%s", city, county)
    return (city, county)


def _geo_lock_required(city: Optional[str], county: Optional[str]) -> bool:
    """
    HARD RULE: if user specifies ANY location (city or county), do NOT return statewide substitutes.
    """
    return bool(city or county)


def _days_to_date_restrict(days: Optional[int]) -> Optional[str]:
    """
    Google CSE dateRestrict supports dN, wN, mN, yN.
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


def _parse_date_from_pagemap(it: Dict[str, Any]) -> Optional[datetime]:
    """
    Try to sniff out publication/update date from Google CSE pagemap metatags.
    """
    pagemap = it.get("pagemap") or {}
    meta_list = pagemap.get("metatags") or []
    if not isinstance(meta_list, list):
        return None

    keys = (
        "article:published_time",
        "article:modified_time",
        "og:published_time",
        "og:updated_time",
        "date",
        "dc.date",
        "dc.date.issued",
        "pubdate",
        "publishdate",
        "datepublished",
    )

    for m in meta_list:
        if not isinstance(m, dict):
            continue
        for key in keys:
            raw = m.get(key)
            if not raw:
                continue
            raw_s = str(raw).strip()
            try:
                dt = datetime.fromisoformat(raw_s.replace("Z", "+00:00"))
            except Exception:
                dt = None
                for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
                    try:
                        dt = datetime.strptime(raw_s[:10], fmt)
                        break
                    except Exception:
                        dt = None
            if not dt:
                continue
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt

    return None


def _google_cse_search(
    query: str,
    max_results: int = 30,
    days: Optional[int] = None,
    file_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Wrapper around Google Custom Search JSON API.
    Returns list: { title, snippet, url, provider, date }
    """
    if not GOOGLE_CSE_KEY or not GOOGLE_CSE_CX:
        log.warning(
            "GOOGLE_CSE_KEY or GOOGLE_CSE_CX not set; returning empty result list. "
            f"key_present={bool(GOOGLE_CSE_KEY)} cx_present={bool(GOOGLE_CSE_CX)}"
        )
        return []

    date_restrict = _days_to_date_restrict(days)
    cutoff = (datetime.utcnow() - timedelta(days=days)) if days and days > 0 else None

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
    seen: set[str] = set()
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
            if not url:
                continue

            canon = _canonicalize_url(url)
            if canon in seen:
                continue
            seen.add(canon)

            dt = _parse_date_from_pagemap(it)

            # If we have a date and a cutoff, enforce it
            if cutoff and isinstance(dt, datetime) and dt < cutoff:
                continue

            out.append(
                {"title": title, "snippet": snippet, "url": canon, "provider": provider, "date": dt}
            )
            if len(out) >= max_results:
                break

        start += 10

    return out


def _build_query(user_q: str, city: Optional[str], county: Optional[str]) -> str:
    """
    Business/news-style industrial query.
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

    q = " ".join(parts)
    log.info("Google CSE query: %s", q)
    return q


def _build_local_gov_query(county: Optional[str], city: Optional[str]) -> str:
    """
    Local-government query pinned to official sites where possible.
    """
    sites: List[str] = []
    if county:
        ck = county.split()[0].strip().lower()
        sites = COUNTY_GOV_SITES.get(ck, [])

    if sites:
        site_block = "(" + " OR ".join([f"site:{s}" for s in sites]) + ")"
    else:
        # Broad fallback if we don't have known sites for that county yet
        site_block = "(site:in.gov OR site:in.us OR site:*.in.gov OR site:*.in.us)"

    geo_bits: List[str] = []
    if county:
        geo_bits.append(f'"{county}"')
    if city:
        geo_bits.append(f'"{city}"')
        geo_bits.append(f'"{city}, IN"')

    geo_clause = "(" + " OR ".join(geo_bits) + ")" if geo_bits else ""

    q = f"{site_block} {LOCAL_GOV_KEYWORDS} {geo_clause} {INDUSTRIAL_SOFT}".strip()
    log.info("Google CSE local-gov query: %s", q)
    return q


def _looks_like_hit(title: str, snippet: str, url: str, allow_local_gov: bool) -> bool:
    """
    Filters obvious junk, then:
    - For normal tier: require facility keyword
    - For local-gov tier: allow agenda/minutes/petition docs even if no "warehouse" in snippet
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

    # size hints
    if re.search(r"\b\d{2,4}[,\d]{0,4}\s*(square[-\s]?feet|sq\.?\s*ft|sf)\b", text):
        score += 2

    # jobs hints
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
    Geo match based on title/snippet mention.
    NOTE: This is intentionally simple; local-gov tier uses site targeting to do the heavy lifting.
    """
    text = _lower(title + " " + snippet)
    match_city = False
    match_county = False

    if county:
        base = county.split()[0].lower()
        if base and re.search(rf"\b{re.escape(base)}\b", text):
            match_county = True
        elif county.lower() in text:
            match_county = True

    if city:
        c = city.lower()
        if c and re.search(rf"\b{re.escape(c)}\b", text):
            match_city = True

    if match_city and match_county:
        geo_score = 2
    elif match_city or match_county:
        geo_score = 1
    else:
        geo_score = 0

    return geo_score, match_city, match_county


def _infer_project_type(title: str, snippet: str, allow_local_gov: bool) -> str:
    if allow_local_gov:
        return "planning / zoning filing"
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
    Convert raw CSE results into dicts used by your chat layer.
    """
    allow_local_gov = (source_tier == "local_gov")
    original_area_label = county or city or "Indiana"

    projects: List[Dict[str, Any]] = []
    for it in raw_items:
        title = it.get("title") or ""
        snippet = it.get("snippet") or ""
        url = it.get("url") or ""
        provider = it.get("provider") or ""
        dt = it.get("date")

        if not _looks_like_hit(title, snippet, url, allow_local_gov=allow_local_gov):
            continue

        forklift_score, forklift_label = _compute_forklift_score(title, snippet)
        geo_score, match_city, match_county = _geo_match_scores(title, snippet, city, county)
        project_type = _infer_project_type(title, snippet, allow_local_gov=allow_local_gov)

        # HARD FILTER: if user provided city/county, require at least one geo match
        # (This prevents statewide filler from sneaking in via weak snippets.)
        if _geo_lock_required(city, county):
            if not (match_city or match_county):
                # For local-gov tier, some docs won't mention the city in snippet; site targeting helps,
                # but we keep this guard to avoid unrelated IN results when city is specified.
                continue

        location_label = county or city or "Indiana"

        timeline_year: Optional[int] = dt.year if isinstance(dt, datetime) else None
        timeline_stage = "agenda/minutes" if allow_local_gov else ("announcement" if timeline_year else "not specified in snippet")

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
                "source_tier": source_tier,
            }
        )

    return projects


def _rank_projects(projects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    now = datetime.utcnow()

    def _sort_key(p: Dict[str, Any]) -> Tuple[int, int, int]:
        geo = p.get("geo_match_score") or 0
        score = p.get("forklift_score") or 0
        dt = p.get("raw_date")
        age_days = 9999
        if isinstance(dt, datetime):
            age_days = (now - dt).days
        # Higher geo, higher forklift score, more recent
        return (-geo, -score, age_days)

    projects.sort(key=_sort_key)
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

    Strategy:
      1) local (industrial/news)
      2) local_gov (plan commission/zoning, PDF heavy)
      3) statewide/fallback ONLY if no location was requested

    HARD RULE:
      If user asks about a specific city OR county, we DO NOT return statewide substitutes.
    """
    city, county = _extract_geo_hint(user_q)
    location_locked = _geo_lock_required(city, county)

    projects: List[Dict[str, Any]] = []

    # ── Tier 1: local industrial/news ────────────────────────────────────────
    query_local = _build_query(user_q, city, county)
    raw_local = _google_cse_search(query_local, max_results=max_items, days=days)
    if raw_local:
        projects.extend(_normalize_projects(raw_local, city, county, user_q, source_tier="local"))

    # ── Tier 2: local government docs (agendas/minutes/petitions) ────────────
    q_gov = _build_local_gov_query(county=county, city=city)
    if q_gov:
        # Try PDFs first (often where agendas/minutes live)
        raw_gov_pdf = _google_cse_search(q_gov, max_results=max_items, days=days, file_type="pdf")
        if raw_gov_pdf:
            projects.extend(_normalize_projects(raw_gov_pdf, city, county, user_q, source_tier="local_gov"))

        # Then try HTML pages
        if not projects:
            raw_gov = _google_cse_search(q_gov, max_results=max_items, days=days, file_type=None)
            if raw_gov:
                projects.extend(_normalize_projects(raw_gov, city, county, user_q, source_tier="local_gov"))

    # If location-locked, STOP here (no statewide filler)
    if location_locked:
        projects = _rank_projects(projects)
        return projects[:max_items]

    # ── Tier 3: statewide ────────────────────────────────────────────────────
    if not projects:
        log.info("No local hits; trying statewide tier")
        query_statewide = _build_query(user_q, city=None, county=None)
        raw_statewide = _google_cse_search(query_statewide, max_results=max_items, days=days)
        if raw_statewide:
            projects = _normalize_projects(raw_statewide, None, None, user_q, source_tier="statewide")

    # ── Tier 4: fallback ─────────────────────────────────────────────────────
    if not projects:
        log.info("No hits; trying fallback tier")
        generic_q = (
            "new or expanded warehouses, distribution centers, logistics facilities, "
            "manufacturing plants, and industrial parks in Indiana in the last few years"
        )
        query_fallback = _build_query(generic_q, city=None, county=None)
        raw_fallback = _google_cse_search(query_fallback, max_results=max_items, days=max(days, 730))
        if raw_fallback:
            projects = _normalize_projects(raw_fallback, None, None, generic_q, source_tier="fallback")

    projects = _rank_projects(projects)
    return projects[:max_items]


def render_developments_markdown(items: List[Dict[str, Any]]) -> str:
    """
    Simple debug formatter.
    """
    if not items:
        return (
            "No location-specific results were found for that timeframe.\n"
            "Tip: try using 'agenda minutes pdf' or 'petition rezoning' phrasing, or expand the day range."
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
        raw_date = item.get("raw_date")

        lines.append(f"{i}. {title} — {loc}")

        meta_bits = [ptype]
        if provider:
            meta_bits.append(provider)
        if forklift_score:
            meta_bits.append(f"Forklift relevance {forklift_score}/5")
        if geo_score:
            meta_bits.append(f"Geo match {geo_score}/2")
        if isinstance(raw_date, datetime):
            meta_bits.append(f"Date: {raw_date.date().isoformat()}")
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
