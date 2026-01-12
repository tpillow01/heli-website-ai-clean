"""
indiana_intel.py

Indiana developments / lead finder for Tynan / Heli AI.

This version is built to solve the "same results regardless of county/city" problem by:
1) Stronger geo extraction (county/city detection is resilient to casing and phrasing)
2) "Site discovery" for planning docs:
   - First find the local municipal/agenda domains for the area
   - Then search WITHIN those sites for industrial keywords
3) Better junk filtering (job boards, licensing/forms pages, tourism, etc.)
4) Optional post-filtering by recency when dates are detectable
"""

from __future__ import annotations

import os
import re
import time
import random
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

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

CSE_TIMEOUT_SECONDS = float(os.environ.get("CSE_TIMEOUT_SECONDS", "10"))
CSE_MAX_RETRIES = int(os.environ.get("CSE_MAX_RETRIES", "4"))
CSE_BACKOFF_BASE_SECONDS = float(os.environ.get("CSE_BACKOFF_BASE_SECONDS", "1.2"))
CSE_BACKOFF_MAX_SECONDS = float(os.environ.get("CSE_BACKOFF_MAX_SECONDS", "20"))
CSE_MIN_INTERVAL_SECONDS = float(os.environ.get("CSE_MIN_INTERVAL_SECONDS", "0.5"))

# Pagination controls: each page is 10 results
CSE_MAX_PAGES = int(os.environ.get("CSE_MAX_PAGES", "1"))
CSE_MAX_RESULTS_HARD_CAP = 50

# Cache controls
CACHE_TTL_SECONDS = int(os.environ.get("CSE_CACHE_TTL_SECONDS", "900"))  # 15 min

# Enable discovery pass (highly recommended)
ENABLE_SITE_DISCOVERY = os.environ.get("CSE_ENABLE_SITE_DISCOVERY", "1").strip() not in {"0", "false", "False"}

# ---------------------------------------------------------------------------
# Keywords
# ---------------------------------------------------------------------------

FACILITY_KEYWORDS = (
    '(warehouse OR "distribution center" OR "distribution facility" OR "distribution hub" OR '
    'logistics OR "logistics center" OR "logistics facility" OR "fulfillment center" OR '
    '"fulfillment facility" OR "industrial park" OR "business park" OR "industrial complex" OR '
    '"manufacturing plant" OR "manufacturing facility" OR plant OR factory OR "production plant" OR '
    '"assembly plant" OR "cold storage" OR "spec building" OR "truck terminal" OR facility OR 3PL OR "data center")'
)

PLANNING_KEYWORDS = (
    '("plan commission" OR "area plan commission" OR "planning commission" OR '
    '"board of zoning appeals" OR BZA OR "metropolitan development commission" OR MDC OR '
    'agenda OR minutes OR packet OR "staff report" OR docket OR petition OR rezoning OR rezone OR '
    'PUD OR "development plan" OR "site plan" OR ordinance OR variance OR "special exception" OR '
    '"primary plat" OR "secondary plat" OR "concept plan")'
)

AGENDA_PLATFORM_HINTS = (
    '("AgendaCenter" OR "DocumentCenter" OR "ViewFile" OR Legistar OR municode OR granicus OR "Meeting Minutes")'
)

INDUSTRIAL_HINTS = (
    '(industrial OR warehouse OR distribution OR logistics OR manufacturing OR "cold storage" OR "data center" OR 3PL)'
)

# Strong junk blockers (domain + keyword)
BLOCK_DOMAINS = {
    "indeed.com", "linkedin.com", "glassdoor.com", "simplyhired.com", "ziprecruiter.com",
    "facebook.com", "instagram.com", "twitter.com", "x.com", "youtube.com", "tiktok.com",
}

BLOCK_TEXT_CONTAINS = [
    # tourism / community
    "visit", "tourism", "visitor", "parks and recreation", "park and recreation", "museum", "library",
    # residential / non-industrial
    "apartments", "subdivision", "condo", "senior living", "assisted living",
    "elementary school", "middle school", "high school", "university", "college",
    "hospital", "medical center", "clinic",
    # licensing / forms pages that match “warehouse” but are not facilities
    "grain buyers", "grain licensing", "warehouse licensing", "licensee listing", "statute and rule", "licensing forms",
]

POSITIVE_FACILITY_TERMS = [
    "warehouse", "distribution center", "distribution facility", "fulfillment center",
    "logistics", "industrial park", "manufacturing", "plant", "factory", "cold storage",
    "spec building", "truck terminal", "data center", "3pl",
]

# ---------------------------------------------------------------------------
# TTL cache (in-process)
# ---------------------------------------------------------------------------

_CACHE: Dict[str, Tuple[float, Any]] = {}
_LAST_REQUEST_TS: float = 0.0


def _cache_get(key: str) -> Optional[Any]:
    now = time.time()
    hit = _CACHE.get(key)
    if not hit:
        return None
    ts, val = hit
    if now - ts > CACHE_TTL_SECONDS:
        _CACHE.pop(key, None)
        return None
    return val


def _cache_set(key: str, val: Any) -> None:
    _CACHE[key] = (time.time(), val)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class CSEQuotaError(RuntimeError):
    """Raised when Google CSE rate-limits (429) and we can't recover quickly."""


def _lower(s: Any) -> str:
    return str(s or "").lower()


def _throttle() -> None:
    global _LAST_REQUEST_TS
    now = time.time()
    elapsed = now - _LAST_REQUEST_TS
    if elapsed < CSE_MIN_INTERVAL_SECONDS:
        time.sleep(CSE_MIN_INTERVAL_SECONDS - elapsed)
    _LAST_REQUEST_TS = time.time()


def _safe_http_error_log(resp: requests.Response, prefix: str) -> None:
    try:
        body = (resp.text or "")[:200].replace("\n", " ")
    except Exception:
        body = ""
    log.warning("%s status=%s body=%s", prefix, resp.status_code, body)


def _days_to_date_restrict(days: Optional[int]) -> Optional[str]:
    """
    CSE dateRestrict is rough (dN or mN). We still use it to bias recency,
    but we also do our own post-filtering when we can detect dates.
    """
    if not days or days <= 0:
        return None
    if days <= 31:
        return f"d{days}"
    months = max(1, int(round(days / 30)))
    if months <= 24:
        return f"m{months}"
    return "y2"  # best-effort fallback; some CSE configs ignore this


def _parse_date_from_pagemap(it: Dict[str, Any]) -> Optional[datetime]:
    pagemap = it.get("pagemap") or {}
    meta_list = pagemap.get("metatags") or []
    if not isinstance(meta_list, list):
        return None

    for m in meta_list:
        if not isinstance(m, dict):
            continue
        for key in ("article:published_time", "article:modified_time", "og:updated_time", "date", "dc.date", "pubdate"):
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


def _parse_date_from_snippet(snippet: str) -> Optional[datetime]:
    """
    Try to catch dates like 'Jan 5, 2026' or 'January 5, 2026' in snippets.
    """
    s = snippet or ""
    m = re.search(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+(20\d{2})\b", s, re.I)
    if m:
        try:
            return datetime.strptime(m.group(0), "%b %d, %Y")
        except Exception:
            try:
                return datetime.strptime(m.group(0), "%B %d, %Y")
            except Exception:
                return None
    return None


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def _is_blocked(title: str, snippet: str, url: str) -> bool:
    txt = _lower(f"{title} {snippet}")
    dom = _domain(url)
    if any(bad in dom for bad in BLOCK_DOMAINS):
        return True
    for bad in BLOCK_TEXT_CONTAINS:
        if bad in txt:
            return True
    return False


def _looks_industrial(title: str, snippet: str) -> bool:
    txt = _lower(f"{title} {snippet}")
    return any(t in txt for t in POSITIVE_FACILITY_TERMS)


def _extract_geo_hint(q: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (city, county).
    Handles:
    - "Boone County" (any casing)
    - "Whitestown, IN" / "Whitestown Indiana"
    - "in Whitestown" / "near Whitestown"
    - bare county token "Hendricks"
    """
    if not q:
        return (None, None)

    text = q.strip()
    tl = text.lower()

    IN_COUNTIES = {
        "adams","allen","bartholomew","benton","blackford","boone","brown","carroll","cass","clark","clay",
        "clinton","crawford","daviess","dearborn","decatur","dekalb","delaware","duboise","elkhart","fayette",
        "floyd","fountain","franklin","fulton","gibson","grant","greene","hamilton","hancock","harrison",
        "hendricks","henry","howard","huntington","jackson","jasper","jay","jennings","johnson","knox",
        "kosciusko","lagrange","lake","laporte","lawrence","madison","marion","marshall","martin","miami",
        "monroe","montgomery","morgan","newton","noble","ohio","orange","owen","parke","perry","pike","porter",
        "posey","pulaski","putnam","randolph","ripley","rush","scott","shelby","spencer","starke","steuben",
        "sullivan","switzerland","tippecanoe","tipton","union","vanderburgh","vermillion","vigo","wabash",
        "warren","warrick","washington","wayne","wells","white","whitley"
    }

    city: Optional[str] = None
    county: Optional[str] = None

    # explicit county phrase
    m = re.search(r"\b([a-z]+)\s+county\b", tl, flags=re.I)
    if m:
        c = m.group(1).strip().lower()
        if c in IN_COUNTIES:
            county = f"{c.title()} County"

    # city, IN or city Indiana
    m = re.search(r"\b([A-Za-z][A-Za-z\s\.\-']{2,})\s*,?\s*(Indiana|IN)\b", text, flags=re.I)
    if m:
        city = m.group(1).strip()

    # in/near/around City
    if not city:
        m = re.search(r"\b(?:in|near|around)\s+([A-Za-z][A-Za-z\s\.\-']+)", text, flags=re.I)
        if m:
            raw = m.group(1).strip()
            raw = re.split(r"\b(20\d{2}|this year|next year|last year|today|now|last|past|recent)\b", raw, flags=re.I)[0].strip()
            raw = re.sub(r"\b(Indiana|IN)\b\.?", "", raw, flags=re.I).strip()
            if raw:
                city = raw

    # bare county token fallback
    if not county:
        tokens = re.findall(r"[A-Za-z]+", text)
        for tok in tokens:
            t = tok.lower()
            if t in IN_COUNTIES:
                county = f"{t.title()} County"
                break

    return (city, county)


def _is_planning_query(q: str) -> bool:
    t = _lower(q)
    triggers = [
        "plan commission", "area plan", "planning commission",
        "bza", "board of zoning", "metropolitan development commission", "mdc",
        "agenda", "minutes", "packet", "staff report", "docket", "petition",
        "rezoning", "rezone", "pud", "variance", "special exception", "site plan",
        "primary plat", "secondary plat",
        "being built", "under construction", "constructed", "construction",
        "breaking ground", "breaks ground",
    ]
    return any(w in t for w in triggers)


# ---------------------------------------------------------------------------
# Google CSE
# ---------------------------------------------------------------------------

def _google_cse_search(query: str, max_results: int = 20, days: Optional[int] = None) -> List[Dict[str, Any]]:
    if not GOOGLE_CSE_KEY or not GOOGLE_CSE_CX:
        log.warning("GOOGLE_CSE_KEY or GOOGLE_CSE_CX not set; returning empty list.")
        return []

    max_results = max(1, min(int(max_results), CSE_MAX_RESULTS_HARD_CAP))
    pages_needed = (max_results + 9) // 10
    pages = max(1, min(pages_needed, max(1, CSE_MAX_PAGES)))

    date_restrict = _days_to_date_restrict(days)

    cache_key = f"q={query}|days={days}|pages={pages}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    sess = requests.Session()
    base_params = {"key": GOOGLE_CSE_KEY, "cx": GOOGLE_CSE_CX, "q": query, "num": 10}
    if date_restrict:
        base_params["dateRestrict"] = date_restrict

    out: List[Dict[str, Any]] = []
    start = 1

    for _ in range(pages):
        _throttle()
        params = dict(base_params)
        params["start"] = start

        attempt = 0
        while True:
            attempt += 1
            try:
                resp = sess.get(GOOGLE_CSE_ENDPOINT, params=params, timeout=CSE_TIMEOUT_SECONDS)
            except Exception as e:
                log.warning("Google CSE request failed start=%s err=%s", start, e)
                break

            if resp.status_code == 200:
                try:
                    data = resp.json()
                except Exception as e:
                    log.warning("Google CSE JSON parse failed start=%s err=%s", start, e)
                    break

                items = data.get("items", []) or []
                log.info("Google CSE returned %s items at start=%s", len(items), start)
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    title = it.get("title") or ""
                    snippet = it.get("snippet") or it.get("htmlSnippet") or ""
                    url = it.get("link") or ""
                    provider = it.get("displayLink") or _domain(url)
                    dt = _parse_date_from_pagemap(it) or _parse_date_from_snippet(snippet)
                    out.append({"title": title, "snippet": snippet, "url": url, "provider": provider, "date": dt})
                    if len(out) >= max_results:
                        break
                break

            if resp.status_code in (429, 500, 502, 503, 504):
                _safe_http_error_log(resp, prefix="Google CSE transient")
                if attempt >= CSE_MAX_RETRIES:
                    if resp.status_code == 429:
                        raise CSEQuotaError("Google CSE rate-limited (429).")
                    break
                wait = min(CSE_BACKOFF_MAX_SECONDS, CSE_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))
                wait += random.uniform(0.0, 0.6)
                time.sleep(wait)
                continue

            _safe_http_error_log(resp, prefix="Google CSE non-retryable")
            break

        start += 10
        if len(out) >= max_results:
            break

    _cache_set(cache_key, out)
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
# Site discovery (the big upgrade)
# ---------------------------------------------------------------------------

def _discover_local_sites(city: Optional[str], county: Optional[str], days: int) -> List[str]:
    """
    Use CSE to discover relevant municipal/agenda domains for this area.
    Returns list of domains (no site: prefix yet).
    """
    if not ENABLE_SITE_DISCOVERY:
        return []

    area = city or county or "Indiana"
    discovery_q = f'{area} Indiana {AGENDA_PLATFORM_HINTS} (agenda OR minutes OR packet OR docket)'
    log.info("Discovery query: %s", discovery_q)

    raw = _google_cse_search(discovery_q, max_results=10, days=days)
    raw = _dedupe_by_url(raw)

    domains: List[str] = []
    for it in raw:
        u = it.get("url") or ""
        d = _domain(u)
        if not d:
            continue
        if any(bad in d for bad in BLOCK_DOMAINS):
            continue
        # Bias toward government-ish domains
        if d.endswith(".gov") or d.endswith(".in.us") or ".in.gov" in d or "municode" in d or "granicus" in d:
            if d not in domains:
                domains.append(d)

    return domains[:8]


def _site_or_clause(domains: List[str]) -> str:
    if not domains:
        return ""
    bits = [f"site:{d}" for d in domains]
    return "(" + " OR ".join(bits) + ")"


# ---------------------------------------------------------------------------
# Normalization / ranking
# ---------------------------------------------------------------------------

def _geo_score(title: str, snippet: str, city: Optional[str], county: Optional[str]) -> int:
    txt = _lower(f"{title} {snippet}")
    score = 0
    if city and city.lower() in txt:
        score += 2
    if county:
        c = county.split()[0].lower()
        if c and c in txt:
            score += 1
    return score


def _recency_ok(dt: Optional[datetime], days: int) -> bool:
    if not dt:
        return True  # unknown date -> keep, but rank lower
    cutoff = datetime.utcnow() - timedelta(days=days)
    return dt >= cutoff


def _compute_score(title: str, snippet: str, url: str, city: Optional[str], county: Optional[str], is_planning: bool) -> int:
    txt = _lower(f"{title} {snippet} {url}")
    score = 0

    score += _geo_score(title, snippet, city, county)

    if _looks_industrial(title, snippet):
        score += 4

    if is_planning:
        if any(k in txt for k in ["docket", "dp-", "pp-", "rezone", "rezon", "variance", "petition", "staff report"]):
            score += 3
        if ".pdf" in txt:
            score += 1

    if re.search(r"\b\d{2,4}[,\d]{0,4}\s*(square[-\s]?feet|sq\.?\s*ft|sf)\b", txt):
        score += 2

    if any(k in txt for k in ["groundbreaking", "breaks ground", "expansion", "announce", "ribbon cutting"]):
        score += 1

    return score


def _rank(items: List[Dict[str, Any]], city: Optional[str], county: Optional[str], is_planning: bool, days: int) -> List[Dict[str, Any]]:
    now = datetime.utcnow()

    scored: List[Dict[str, Any]] = []
    for it in items:
        title = it.get("title") or ""
        snippet = it.get("snippet") or ""
        url = it.get("url") or ""
        dt = it.get("date")

        if _is_blocked(title, snippet, url):
            continue

        # In facility mode, require at least some industrial term
        if not is_planning and not _looks_industrial(title, snippet):
            continue

        # Post-filter recency when date is known
        if not _recency_ok(dt, days):
            continue

        score = _compute_score(title, snippet, url, city, county, is_planning)

        scored.append({
            "project_name": title or "Untitled",
            "company": None,
            "project_type": "planning / zoning filing" if is_planning else "facility/news hit",
            "location_label": county or city or "Indiana",
            "geo_match_score": _geo_score(title, snippet, city, county),
            "forklift_score": max(1, min(5, 1 + score // 4)),
            "timeline_year": dt.year if isinstance(dt, datetime) else None,
            "timeline_stage": "planning doc" if is_planning else "announcement/news",
            "raw_date": dt,
            "url": url,
            "provider": it.get("provider") or _domain(url),
            "snippet": snippet,
            "source_tier": "local",
            "result_mode": "planning" if is_planning else "facility",
            "_score": score,
            "_age_days": (now - dt).days if isinstance(dt, datetime) else 9999,
        })

    scored.sort(key=lambda x: (-int(x.get("_score", 0)), int(x.get("_age_days", 9999))))
    return scored


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_indiana_developments(user_q: str, days: int = 365, max_items: int = 25) -> List[Dict[str, Any]]:
    """
    Main entrypoint.
    """
    city, county = _extract_geo_hint(user_q)
    is_planning = _is_planning_query(user_q)

    log.info("Geo hint: city=%s county=%s planning=%s", city, county, is_planning)

    max_items = max(1, min(int(max_items), CSE_MAX_RESULTS_HARD_CAP))

    # 1) If planning-style question, do site discovery and search within those sites
    if is_planning:
        domains = _discover_local_sites(city, county, days=min(days, 730))
        site_clause = _site_or_clause(domains)

        area = city or county or "Indiana"
        q = f'{site_clause} {area} Indiana {PLANNING_KEYWORDS} {INDUSTRIAL_HINTS}'
        log.info("Planning query: %s", q)

        try:
            raw = _google_cse_search(q, max_results=max_items, days=min(days, 730))
        except CSEQuotaError:
            return []

        raw = _dedupe_by_url(raw)
        ranked = _rank(raw, city, county, is_planning=True, days=min(days, 730))
        return ranked[:15]

    # 2) Facility/news mode (announcements/expansions)
    area = city or county or "Indiana"
    # Add “announce/expansion” bias, and exclude job listings
    q = f'{area} Indiana {FACILITY_KEYWORDS} (announced OR expansion OR groundbreaking OR "now open" OR "ribbon cutting") -jobs -hiring -indeed -linkedin'
    log.info("Facility query: %s", q)

    try:
        raw = _google_cse_search(q, max_results=max_items, days=days)
    except CSEQuotaError:
        return []

    raw = _dedupe_by_url(raw)
    ranked = _rank(raw, city, county, is_planning=False, days=days)
    return ranked[:15]


def render_developments_markdown(items: List[Dict[str, Any]]) -> str:
    if not items:
        return (
            "No results returned for that specific area/time window.\n"
            "If your logs show city=None/county=None, your query isn't being localized.\n"
            "Try a planning-style question (agenda/packet) or increase days to 730."
        )

    lines: List[str] = []
    lines.append("Industrial / logistics results (web search hits):")

    for i, item in enumerate(items[:15], start=1):
        title = item.get("project_name") or "Untitled"
        snippet = (item.get("snippet") or "").strip()
        url = item.get("url") or ""
        provider = item.get("provider") or ""
        year = item.get("timeline_year")
        stage = item.get("timeline_stage") or ""
        loc = item.get("location_label") or "Indiana"
        ptype = item.get("project_type") or "Industrial / commercial project"
        score = item.get("forklift_score") or ""
        geo_score = item.get("geo_match_score") or 0
        mode = item.get("result_mode") or "unknown"

        stage_year = stage
        if year:
            stage_year = f"{stage} ({year})"

        lines.append(f"{i}. {title} — {loc}")
        meta_bits = [ptype, f"Mode: {mode}"]
        if provider:
            meta_bits.append(provider)
        if score:
            meta_bits.append(f"Relevance {score}/5")
        if geo_score:
            meta_bits.append(f"Geo match {geo_score}/3")
        if stage_year:
            meta_bits.append(stage_year)

        lines.append("   " + " • ".join(meta_bits))
        if snippet:
            lines.append(f"   Snippet: {snippet}")
        if url:
            lines.append(f"   URL: {url}")

    return "\n".join(lines)


__all__ = ["search_indiana_developments", "render_developments_markdown"]
