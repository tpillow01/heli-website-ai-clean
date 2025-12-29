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
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

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

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    )
}

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


def _canonicalize_url(url: str) -> str:
    """
    Normalize URLs for dedupe:
    - remove common tracking params
    - remove fragments
    - strip trailing slash
    """
    try:
        u = urlparse(url)
        qs = [
            (k, v)
            for (k, v) in parse_qsl(u.query, keep_blank_values=True)
            if k.lower()
            not in {
                "utm_source",
                "utm_medium",
                "utm_campaign",
                "utm_term",
                "utm_content",
                "gclid",
                "fbclid",
            }
        ]
        new_query = urlencode(qs, doseq=True)
        clean = urlunparse((u.scheme, u.netloc, u.path, u.params, new_query, ""))
        return clean.rstrip("/")
    except Exception:
        return (url or "").rstrip("/")


def _extract_geo_hint(q: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract (city, county) from question text.

    Fix vs your current version:
    - Avoids capturing the entire remainder of the question as "city"
    - Avoids treating "<County> County" as a city
    """
    if not q:
        return (None, None)

    text = q.strip()

    # County: e.g. "Boone County", "in Boone County"
    m_county = re.search(r"\b([A-Za-z]+)\s+County\b", text)
    county = None
    if m_county:
        county = f"{m_county.group(1).strip()} County"

    # City: ONLY capture a reasonable city phrase, and stop before common timeframe words
    # Examples captured:
    #   "in Plainfield"
    #   "near Brownsburg, IN"
    #   "around Lebanon Indiana"
    #
    # Examples NOT captured as city:
    #   "in Boone County in the last 180 days"
    #   "in Hendricks County last month"
    m_city = re.search(
        r"\b(?:in|around|near)\s+([A-Za-z][A-Za-z\s]{1,40}?)(?:,?\s*(?:IN|Indiana)\b|,|\?|\.|$)",
        text,
        flags=re.I,
    )
    city = None
    if m_city:
        raw = m_city.group(1).strip()

        # If the "city" phrase contains obvious timeframe words, discard it
        if re.search(r"\b(last|past|days|months|weeks|years|since|recent)\b", raw, flags=re.I):
            raw = ""

        # Strip "IN" or "Indiana" if present
        raw = re.sub(r"\b(Indiana|IN)\b\.?", "", raw, flags=re.I).strip()

        # Don't treat "Boone County" as a city
        if re.search(r"\bCounty\b", raw, flags=re.I):
            raw = ""

        if raw:
            # Keep it short and sane
            city = " ".join(raw.split()[:3]).strip()

    log.info("Geo hint extracted: city=%s county=%s", city, county)
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
    Build a Google CSE query anchored on industrial/commercial keywords,
    with optional city / county bias, plus a small keyword tail from the question.

    Fix vs your current version:
    - If both city+county exist, uses (county OR city) instead of ANDing both.
    """
    parts: List[str] = ["Indiana", BASE_KEYWORDS]

    geo_terms: List[str] = []
    if county:
        geo_terms.append(f'"{county}"')
    if city:
        geo_terms.append(f'"{city}"')

    if geo_terms:
        if len(geo_terms) == 1:
            parts.append(geo_terms[0])
        else:
            parts.append("(" + " OR ".join(geo_terms) + ")")

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

    keys_to_try = (
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
        "sailthru.date",
        "parsely-pub-date",
    )

    for m in meta_list:
        if not isinstance(m, dict):
            continue
        for key in keys_to_try:
            if key in m and m[key]:
                raw = str(m[key]).strip()
                dt = _parse_any_date(raw)
                if dt:
                    return dt
    return None


def _parse_any_date(raw: str) -> Optional[datetime]:
    """
    Parse common date strings into UTC-naive datetime.
    """
    if not raw:
        return None

    raw = raw.strip()

    # ISO-ish first
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except Exception:
        pass

    # Common formats
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%b %d, %Y", "%B %d, %Y"):
        try:
            dt = datetime.strptime(raw[:25], fmt)
            return dt
        except Exception:
            continue

    # Last resort: try to find YYYY-MM-DD inside
    m = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", raw)
    if m:
        try:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return datetime(y, mo, d)
        except Exception:
            return None

    return None


def _days_to_date_restrict(days: Optional[int]) -> Optional[str]:
    """
    Map a days window into Google CSE dateRestrict syntax.
    - dN = last N days
    - wN = last N weeks
    - mN = last N months
    - yN = last N years
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


def _try_extract_date_from_html(url: str) -> Optional[datetime]:
    """
    Lightweight publish-date extraction by fetching the page and checking:
    - meta article:published_time / og:published_time
    - meta name=pubdate/date/datePublished
    - <time datetime="...">

    Only called for a limited number of top results to avoid being slow.
    """
    if not url:
        return None

    try:
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=10, allow_redirects=True)
        if resp.status_code >= 400:
            return None
        html = resp.text or ""
    except Exception:
        return None

    # Meta tags
    meta_patterns = [
        r'property=["\']article:published_time["\']\s+content=["\']([^"\']+)["\']',
        r'property=["\']og:published_time["\']\s+content=["\']([^"\']+)["\']',
        r'name=["\']pubdate["\']\s+content=["\']([^"\']+)["\']',
        r'name=["\']publishdate["\']\s+content=["\']([^"\']+)["\']',
        r'name=["\']date["\']\s+content=["\']([^"\']+)["\']',
        r'itemprop=["\']datePublished["\']\s+content=["\']([^"\']+)["\']',
    ]
    for pat in meta_patterns:
        m = re.search(pat, html, flags=re.I)
        if m:
            dt = _parse_any_date(m.group(1))
            if dt:
                return dt

    # <time datetime="...">
    m_time = re.search(r'<time[^>]+datetime=["\']([^"\']+)["\']', html, flags=re.I)
    if m_time:
        dt = _parse_any_date(m_time.group(1))
        if dt:
            return dt

    return None


def _google_cse_search(
    query: str,
    max_results: int = 30,
    days: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Wrapper around Google Custom Search JSON API.

    Improvements:
    - dateRestrict applied
    - strict cutoff applied when dates are available
    - optional page fetch for missing dates (top N only)
    - dedupe by canonical URL
    """
    if not GOOGLE_CSE_KEY or not GOOGLE_CSE_CX:
        log.warning(
            "GOOGLE_CSE_KEY or GOOGLE_CSE_CX not set or empty; returning empty result list. "
            f"GOOGLE_CSE_KEY present={bool(GOOGLE_CSE_KEY)}, GOOGLE_CSE_CX present={bool(GOOGLE_CSE_CX)}"
        )
        return []

    date_restrict = _days_to_date_restrict(days)
    cutoff = None
    if days and days > 0:
        cutoff = datetime.utcnow() - timedelta(days=days)

    base_params = {
        "key": GOOGLE_CSE_KEY,
        "cx": GOOGLE_CSE_CX,
        "q": query,
        "num": 10,
        # NOTE: sort behavior depends on your PSE settings; leaving it enabled,
        # but dateRestrict + our strict cutoff does the real work.
        "sort": "date",
    }
    if date_restrict:
        base_params["dateRestrict"] = date_restrict

    out: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()
    start = 1

    while len(out) < max_results and start <= 91:
        params = dict(base_params)
        params["start"] = start

        try:
            resp = requests.get(
                GOOGLE_CSE_ENDPOINT,
                params=params,
                headers=REQUEST_HEADERS,
                timeout=10,
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
            if canon in seen_urls:
                continue
            seen_urls.add(canon)

            dt = _parse_date_from_pagemap(it)

            # Strict cutoff if we got a date
            if cutoff and isinstance(dt, datetime) and dt < cutoff:
                continue

            out.append(
                {
                    "title": title,
                    "snippet": snippet,
                    "url": canon,
                    "provider": provider,
                    "date": dt,
                }
            )

            if len(out) >= max_results:
                break

        if len(out) >= max_results:
            break

        start += 10

    # If we have a cutoff and many items have no date, try to fill missing dates for top items
    if cutoff:
        missing = [i for i in out if not isinstance(i.get("date"), datetime)]
        # Only fetch a small number to avoid slowing down the request
        for i in missing[:8]:
            fetched_dt = _try_extract_date_from_html(i.get("url") or "")
            if fetched_dt:
                i["date"] = fetched_dt

        # Apply strict cutoff again after filling dates
        filtered: List[Dict[str, Any]] = []
        for i in out:
            dt = i.get("date")
            if isinstance(dt, datetime) and dt < cutoff:
                continue
            filtered.append(i)
        out = filtered

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

_PROJECT_NEGATIVE_URL = [
    "facebook.com",
    "instagram.com",
    "twitter.com",
    "x.com",
    "youtube.com",
    "tripadvisor.com",
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

    has_positive = any(pos in text for pos in _FORKLIFT_POSITIVE)
    if not has_positive:
        return False

    return True


def _compute_forklift_score(title: str, snippet: str, url: str) -> Tuple[int, str]:
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
      1) "local":  city/county-aware query based on the user's question.
      2) "statewide": same question, but without city/county constraint.
      3) "fallback": county-aware generic search.
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
                f"in the last {days} days"
            )
            query_fallback = _build_query(generic_q, city=None, county=county)
        else:
            generic_q = (
                f"new or expanded warehouses, distribution centers, logistics facilities, "
                f"manufacturing plants, and industrial parks in Indiana in the last {days} days"
            )
            query_fallback = _build_query(generic_q, city=None, county=None)

        raw_fallback = _google_cse_search(query_fallback, max_results=max_items, days=days)
        if raw_fallback:
            projects = _normalize_projects(raw_fallback, None, county, generic_q, source_tier="fallback")

    if not projects:
        return []

    # ── If user asked for a county/city and we have any geo-matching hits, drop statewide noise ──
    if county or city:
        geo_hits = [p for p in projects if (p.get("geo_match_score") or 0) > 0]
        if geo_hits:
            projects = geo_hits

    # ── Rank by geo match, forklift relevance, and recency ────────────────────
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
        raw_date = item.get("raw_date")

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
        if isinstance(raw_date, datetime):
            meta_bits.append(f"Date: {raw_date.date().isoformat()}")
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
