"""
indiana_intel.py

Indiana developments / lead finder for Tynan / Heli AI.

Fixes vs prior version:
- Better geo extraction (prevents "Boone County in the last 180 days" from becoming a city)
- County requests are enforced: results must mention county OR a major town in that county
- Adds "new development" intent gate (filters generic "locations" / "careers" pages)
- Stronger geo matching using county town aliases
- Optional publish-date fill for missing dates (top few results only)
- URL canonicalization + dedupe

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

# Base industrial / logistics / commercial keywords
BASE_KEYWORDS = (
    '(warehouse OR "distribution center" OR "distribution facility" OR '
    '"distribution hub" OR logistics OR "logistics center" OR '
    '"logistics facility" OR "logistics hub" OR "fulfillment center" OR '
    '"industrial park" OR "business park" OR "industrial complex" OR '
    '"manufacturing plant" OR "manufacturing facility" OR plant OR factory '
    'OR "production plant" OR "assembly plant" OR "cold storage" OR facility)'
)

# County → major towns (used as acceptable geo matches when county requested)
COUNTY_TOWNS: Dict[str, List[str]] = {
    "boone": ["Lebanon", "Whitestown", "Zionsville", "Jamestown", "Thorntown"],
    "hendricks": ["Plainfield", "Avon", "Brownsburg", "Danville", "Clayton", "Pittsboro"],
    "hamilton": ["Fishers", "Carmel", "Noblesville", "Westfield"],
    "marion": ["Indianapolis", "Speedway", "Lawrence", "Beech Grove"],
    # add more as you like
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lower(s: Any) -> str:
    return str(s or "").lower()


def _canonicalize_url(url: str) -> str:
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

    Avoids capturing extra trailing phrases as city.
    """
    if not q:
        return (None, None)

    text = q.strip()

    # County: "Boone County"
    m_county = re.search(r"\b([A-Za-z]+)\s+County\b", text)
    county = None
    if m_county:
        county = f"{m_county.group(1).strip()} County"

    # City: capture up to punctuation / end, but reject timeframe-y captures
    m_city = re.search(
        r"\b(?:in|around|near)\s+([A-Za-z][A-Za-z\s]{1,40}?)(?:,?\s*(?:IN|Indiana)\b|,|\?|\.|$)",
        text,
        flags=re.I,
    )
    city = None
    if m_city:
        raw = m_city.group(1).strip()
        if re.search(r"\b(last|past|days|months|weeks|years|since|recent)\b", raw, flags=re.I):
            raw = ""
        raw = re.sub(r"\b(Indiana|IN)\b\.?", "", raw, flags=re.I).strip()
        if re.search(r"\bCounty\b", raw, flags=re.I):
            raw = ""
        if raw:
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


def _user_wants_new_devs(user_q: str) -> bool:
    """
    Determine if the question implies "new/announced/expanded" developments.
    If yes, we enforce NEW_DEV_TERMS in the hit text to avoid generic location pages.
    """
    t = _lower(user_q)
    triggers = [
        "new", "announced", "announcement", "breaking ground", "groundbreaking",
        "expansion", "expanded", "to build", "plans", "planning", "permit", "rezoning",
        "spec building", "construction", "development",
    ]
    return any(x in t for x in triggers)


def _county_key(county: Optional[str]) -> Optional[str]:
    if not county:
        return None
    return county.split()[0].strip().lower()


def _geo_terms(city: Optional[str], county: Optional[str]) -> List[str]:
    """
    Geo terms accepted for matching. If county is specified, include its major towns.
    """
    terms: List[str] = []
    if county:
        terms.append(county)
        ck = _county_key(county)
        if ck and ck in COUNTY_TOWNS:
            terms.extend([f"{town}, IN" for town in COUNTY_TOWNS[ck]])
            terms.extend(COUNTY_TOWNS[ck])
    if city:
        terms.append(city)
        terms.append(f"{city}, IN")
    # unique, preserve order
    seen = set()
    out = []
    for x in terms:
        x2 = x.strip()
        if not x2:
            continue
        k = x2.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(x2)
    return out


def _build_query(user_q: str, city: Optional[str], county: Optional[str]) -> str:
    """
    Build query with county/city geo terms.
    If county is provided, include major towns as OR terms.
    """
    parts: List[str] = ["Indiana", BASE_KEYWORDS]

    geo = _geo_terms(city, county)
    if geo:
        geo_q = " OR ".join([f'"{g}"' for g in geo[:10]])  # cap to keep query sane
        parts.append(f"({geo_q})")

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


def _parse_any_date(raw: str) -> Optional[datetime]:
    if not raw:
        return None
    raw = raw.strip()
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except Exception:
        pass

    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(raw[:25], fmt)
        except Exception:
            continue

    m = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", raw)
    if m:
        try:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return datetime(y, mo, d)
        except Exception:
            return None
    return None


def _parse_date_from_pagemap(it: Dict[str, Any]) -> Optional[datetime]:
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
                dt = _parse_any_date(str(m[key]).strip())
                if dt:
                    return dt
    return None


def _days_to_date_restrict(days: Optional[int]) -> Optional[str]:
    """
    Google CSE dateRestrict accepts: dN, wN, mN, yN.
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
    if not url:
        return None
    try:
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=10, allow_redirects=True)
        if resp.status_code >= 400:
            return None
        html = resp.text or ""
    except Exception:
        return None

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

    m_time = re.search(r'<time[^>]+datetime=["\']([^"\']+)["\']', html, flags=re.I)
    if m_time:
        dt = _parse_any_date(m_time.group(1))
        if dt:
            return dt

    return None


def _google_cse_search(query: str, max_results: int = 30, days: Optional[int] = None) -> List[Dict[str, Any]]:
    if not GOOGLE_CSE_KEY or not GOOGLE_CSE_CX:
        log.warning(
            "GOOGLE_CSE_KEY or GOOGLE_CSE_CX not set or empty; returning empty result list. "
            f"GOOGLE_CSE_KEY present={bool(GOOGLE_CSE_KEY)}, GOOGLE_CSE_CX present={bool(GOOGLE_CSE_CX)}"
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

    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    start = 1

    while len(out) < max_results and start <= 91:
        params = dict(base_params)
        params["start"] = start

        try:
            resp = requests.get(GOOGLE_CSE_ENDPOINT, params=params, headers=REQUEST_HEADERS, timeout=10)
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

            # strict cutoff if we have a date
            if cutoff and isinstance(dt, datetime) and dt < cutoff:
                continue

            out.append({"title": title, "snippet": snippet, "url": canon, "provider": provider, "date": dt})
            if len(out) >= max_results:
                break

        if len(out) >= max_results:
            break
        start += 10

    # Fill missing dates for a few results, then apply cutoff again
    if cutoff:
        missing = [i for i in out if not isinstance(i.get("date"), datetime)]
        for i in missing[:8]:
            fetched_dt = _try_extract_date_from_html(i.get("url") or "")
            if fetched_dt:
                i["date"] = fetched_dt

        out2: List[Dict[str, Any]] = []
        for i in out:
            dt = i.get("date")
            if isinstance(dt, datetime) and dt < cutoff:
                continue
            out2.append(i)
        out = out2

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

# These terms indicate "actual development news", not generic service/location pages
NEW_DEV_TERMS = [
    "announced",
    "announcement",
    "breaking ground",
    "groundbreaking",
    "to build",
    "plans to build",
    "construction",
    "expands",
    "expansion",
    "new facility",
    "new plant",
    "new distribution",
    "new warehouse",
    "site plan",
    "rezoning",
    "plan commission",
    "permit",
    "invest",
    "investment",
    "square feet",
    "sq ft",
    "jobs",
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
    # common “not-new-dev” pages:
    "our locations",
    "locations",
    "warehouse locations",
    "service locations",
    "contact us",
]

_PROJECT_NEGATIVE_URL = [
    "facebook.com",
    "instagram.com",
    "twitter.com",
    "x.com",
    "youtube.com",
    "tripadvisor.com",
]


def _looks_like_facility_hit(title: str, snippet: str, url: str, require_new_dev: bool) -> bool:
    text = _lower(f"{title} {snippet}")
    url_l = _lower(url or "")

    for bad in _PROJECT_NEGATIVE_URL:
        if bad in url_l:
            return False

    for neg in _PROJECT_NEGATIVE_TEXT:
        if neg in text:
            return False

    if not any(pos in text for pos in _FORKLIFT_POSITIVE):
        return False

    if require_new_dev:
        if not any(t in text for t in NEW_DEV_TERMS):
            return False

    return True


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


def _geo_match_scores(title: str, snippet: str, geo_terms: List[str]) -> Tuple[int, bool]:
    """
    Score geo match based on whether ANY acceptable geo term appears.
    """
    text = _lower(title + " " + snippet)
    for term in geo_terms:
        t = term.lower()
        # word-ish boundary match for single words, substring for multiword phrases
        if " " in t or "," in t:
            if t in text:
                return 2, True
        else:
            if re.search(rf"\b{re.escape(t)}\b", text):
                return 2, True
    return 0, False


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
    user_q: str,
    city: Optional[str],
    county: Optional[str],
    source_tier: str,
) -> List[Dict[str, Any]]:
    original_area_label = county or city or "Indiana"
    geo_terms = _geo_terms(city, county)
    require_new_dev = _user_wants_new_devs(user_q)

    projects: List[Dict[str, Any]] = []
    for it in raw_items:
        title = it.get("title") or ""
        snippet = it.get("snippet") or ""
        url = it.get("url") or ""
        provider = it.get("provider") or ""
        dt = it.get("date")

        if not _looks_like_facility_hit(title, snippet, url, require_new_dev=require_new_dev):
            continue

        forklift_score, forklift_label = _compute_forklift_score(title, snippet)
        geo_score, geo_ok = _geo_match_scores(title, snippet, geo_terms)

        # If county was requested, enforce geo_ok (county OR town in county must be in title/snippet)
        if county and not geo_ok:
            continue

        project_type = _infer_project_type(title, snippet)
        location_label = county or city or "Indiana"

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

def search_indiana_developments(user_q: str, days: int = 365, max_items: int = 30) -> List[Dict[str, Any]]:
    """
    Main entrypoint.

    Important behavior change:
    - If a COUNTY is requested, we do NOT return statewide substitutes.
      We either return county-matching hits, or return [].
    """
    city, county = _extract_geo_hint(user_q)

    # Tier 1: local
    query_local = _build_query(user_q, city, county)
    raw_local = _google_cse_search(query_local, max_results=max_items, days=days)
    projects = _normalize_projects(raw_local, user_q, city, county, source_tier="local") if raw_local else []

    # If user asked for a county, do NOT “helpfully” replace with statewide noise
    if county:
        return _rank_projects(projects)[:max_items]

    # Tier 2: statewide (only if no county specified)
    if not projects:
        log.info("No forklift-relevant local projects; trying statewide tier")
        query_statewide = _build_query(user_q, city=None, county=None)
        raw_statewide = _google_cse_search(query_statewide, max_results=max_items, days=days)
        if raw_statewide:
            projects = _normalize_projects(raw_statewide, user_q, None, None, source_tier="statewide")

    # Tier 3: fallback (only if no county specified)
    if not projects:
        log.info("No forklift-relevant hits for user query; running generic Indiana fallback")
        generic_q = (
            f"new or expanded warehouses, distribution centers, logistics facilities, "
            f"manufacturing plants, and industrial parks in Indiana in the last {days} days"
        )
        query_fallback = _build_query(generic_q, city=None, county=None)
        raw_fallback = _google_cse_search(query_fallback, max_results=max_items, days=days)
        if raw_fallback:
            projects = _normalize_projects(raw_fallback, generic_q, None, None, source_tier="fallback")

    return _rank_projects(projects)[:max_items]


def _rank_projects(projects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
    return projects


def render_developments_markdown(items: List[Dict[str, Any]]) -> str:
    if not items:
        return (
            "No county-specific web hits were found for that location/timeframe.\n"
            "Tip: try a slightly broader phrasing like:\n"
            "- \"industrial park rezoning Boone County\" \n"
            "- \"plan commission warehouse Lebanon IN\" \n"
            "- \"spec building Whitestown\""
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


__all__ = ["search_indiana_developments", "render_developments_markdown"]
