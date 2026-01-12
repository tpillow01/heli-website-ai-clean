"""
indiana_intel.py

Indiana developments / lead finder for Tynan / Heli AI.

GOAL (what “works the way you hope” means in practice):
- When you ask about a CITY or COUNTY in Indiana, it should actually detect that geo.
- It should avoid garbage like job boards and generic statewide “jobs” pages.
- It should return the strongest “industrial development signals” first:
    1) Facility/news announcements (warehouses, DCs, plants, expansions)
    2) Planning/zoning/agenda packets (often the earliest proof a project is real)
- It should still return something useful even when “news” is thin by leaning into planning docs.
- It should be stable on Render: cached, throttled, resilient to 429s, and never logs your API key.

Exports:
- search_indiana_developments(user_q: str, days: int = 365, max_items: int = 30) -> List[Dict]
- render_developments_markdown(items: List[Dict]) -> str
"""

from __future__ import annotations

import os
import re
import time
import random
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
log = logging.getLogger("indiana_intel")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


# -----------------------------------------------------------------------------
# Config: Google CSE
# -----------------------------------------------------------------------------
GOOGLE_CSE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"
GOOGLE_CSE_KEY = os.environ.get("GOOGLE_CSE_KEY")
GOOGLE_CSE_CX = os.environ.get("GOOGLE_CSE_CX")

CSE_TIMEOUT_SECONDS = float(os.environ.get("CSE_TIMEOUT_SECONDS", "10"))
CSE_MAX_RETRIES = int(os.environ.get("CSE_MAX_RETRIES", "4"))
CSE_BACKOFF_BASE_SECONDS = float(os.environ.get("CSE_BACKOFF_BASE_SECONDS", "1.2"))
CSE_BACKOFF_MAX_SECONDS = float(os.environ.get("CSE_BACKOFF_MAX_SECONDS", "20"))
CSE_MIN_INTERVAL_SECONDS = float(os.environ.get("CSE_MIN_INTERVAL_SECONDS", "0.5"))

# Pagination: keep small to avoid quota burn (10 results per page)
CSE_MAX_PAGES = int(os.environ.get("CSE_MAX_PAGES", "1"))  # 1 page=10 results
CSE_MAX_RESULTS_HARD_CAP = 50

# Cache TTL (seconds)
CACHE_TTL_SECONDS = int(os.environ.get("CSE_CACHE_TTL_SECONDS", "900"))  # 15 minutes

# Optional: Google CSE sort parameter. Many CSE setups don't support sort.
# If your CSE supports it, set CSE_SORT="date" or a custom sort expression.
CSE_SORT = os.environ.get("CSE_SORT", "").strip()  # default "" (disabled)


# -----------------------------------------------------------------------------
# Keyword sets
# -----------------------------------------------------------------------------
BASE_FACILITY_KEYWORDS = (
    '(warehouse OR "distribution center" OR "distribution facility" OR "distribution hub" OR '
    'logistics OR "logistics center" OR "logistics facility" OR "logistics hub" OR '
    '"fulfillment center" OR "fulfillment facility" OR "industrial park" OR "business park" OR '
    '"industrial complex" OR "manufacturing plant" OR "manufacturing facility" OR plant OR factory OR '
    '"production plant" OR "assembly plant" OR "cold storage" OR "spec building" OR '
    '"truck terminal" OR 3PL OR "third-party logistics")'
)

# Planning docs often contain the earliest verifiable signal
PLANNING_KEYWORDS = (
    '("plan commission" OR "area plan commission" OR "planning commission" OR '
    '"board of zoning appeals" OR BZA OR "metropolitan development commission" OR MDC OR '
    "agenda OR minutes OR docket OR petition OR rezoning OR rezone OR PUD OR "
    '"development plan" OR "site plan" OR "staff report" OR "public hearing" OR ordinance OR '
    "variance OR "
    '"primary plat" OR "secondary plat" OR "concept plan" OR "zoning case" OR '
    '"special exception" OR "improvement location permit" OR "building permit")'
)

INDUSTRIAL_SIGNALS = [
    "industrial", "warehouse", "distribution", "logistics", "cold storage", "fulfillment",
    "spec building", "truck terminal", "manufacturing", "plant", "factory", "3pl",
    "i-1", "i-2", "i-3", "i-4", "i1", "i2", "i3", "i4",
    "industrial district", "light industrial", "heavy industrial",
    "groundbreaking", "breaks ground", "under construction",
    "square feet", "sq ft", "sf",
]

AGENDA_PLATFORMS = [
    "agendacenter", "viewfile", "documentcenter", "legistar", "municode",
    "granicus", "minutes", "agenda", "packet", "meeting", "hearing",
]

# Strong negatives to reduce junk
NEGATIVE_TEXT = [
    "tourism", "visitors bureau", "parks and recreation", "park and recreation",
    "shopping center", "outlet", "mall", "hotel", "resort", "casino", "water park",
    "museum", "library", "stadium", "arena", "sports complex", "golf",
    "apartments", "apartment", "subdivision", "condo", "senior living",
    "assisted living", "retirement community",
    "elementary school", "middle school", "high school", "university", "college",
    "hospital", "medical center", "clinic",
    "church", "ministry",
    "wikipedia",  # tends to be broad + not “new”
]

NEGATIVE_URL = [
    "facebook.com", "instagram.com", "twitter.com", "x.com", "youtube.com",
    "tripadvisor.com",
    # job boards / career pages
    "indeed.com", "linkedin.com", "ziprecruiter.com", "glassdoor.com",
    "simplyhired.com", "monster.com", "careerbuilder.com", "jobs.", "/jobs",
]

FACILITY_POSITIVE = [
    "warehouse", "distribution center", "distribution facility", "distribution hub",
    "fulfillment center", "fulfillment facility",
    "logistics center", "logistics facility", "logistics hub",
    "industrial park", "business park", "industrial complex",
    "manufacturing plant", "manufacturing facility", "production plant", "assembly plant",
    "factory", "cold storage", "spec building", "truck terminal",
    "3pl", "third-party logistics", "third party logistics",
]

STOPWORDS = {
    "what", "are", "there", "any", "new", "or", "in", "the", "last", "month", "months",
    "recent", "recently", "project", "projects", "have", "has", "been", "announced",
    "announcement", "for", "about", "on", "of", "a", "an", "county", "indiana",
    "coming", "to", "area", "city", "kind", "sort", "type", "planned", "plan",
    "announce", "expanded", "expansion", "hiring", "jobs",
    "happening", "occur", "occurring", "constructed", "construction",
    "being", "built", "build", "opening", "open",
}


# -----------------------------------------------------------------------------
# Simple TTL cache (in-process)
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
class CSEQuotaError(RuntimeError):
    """Raised when Google CSE rate-limits (429) and we can't recover quickly."""


def _lower(s: Any) -> str:
    return str(s or "").lower()


def _slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9\s-]", "", _lower(s))
    s = re.sub(r"\s+", "-", s).strip("-")
    return s


# Full-ish county list (spelling corrected: dubois)
IN_COUNTIES = {
    "adams","allen","bartholomew","benton","blackford","boone","brown","carroll","cass","clark","clay",
    "clinton","crawford","daviess","dearborn","decatur","dekalb","delaware","dubois","elkhart","fayette",
    "floyd","fountain","franklin","fulton","gibson","grant","greene","hamilton","hancock","harrison",
    "hendricks","henry","howard","huntington","jackson","jasper","jay","jennings",
    "johnson","knox","kosciusko","lagrange","lake","laporte","lawrence","madison","marion","marshall",
    "martin","miami","monroe","montgomery","morgan","newton","noble","ohio","orange","owen","parke",
    "perry","pike","porter","posey","pulaski","putnam","randolph","ripley","rush","scott","shelby",
    "spencer","starke","steuben","sullivan","switzerland","tippecanoe","tipton","union","vanderburgh",
    "vermillion","vigo","wabash","warren","warrick","washington","wayne","wells","white","whitley"
}


def _extract_geo_hint(q: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (city, county) where county is "X County" or None.
    Handles:
      - "Boone County"
      - "Whitestown (Boone County)"
      - "Whitestown, IN" / "Whitestown Indiana"
      - "in Whitestown"
      - "Boone developments" / "Hendricks 2026"
    """
    if not q:
        return (None, None)

    text = q.strip()
    tl = text.lower()

    city: Optional[str] = None
    county: Optional[str] = None

    # 1) Explicit "X County"
    m = re.search(r"\b([a-z]+)\s+county\b", tl, flags=re.I)
    if m:
        c = m.group(1).strip().lower()
        if c in IN_COUNTIES:
            county = f"{c.title()} County"

    # 2) "City (X County)"
    m = re.search(r"\b([A-Za-z][A-Za-z\s\.\-']+?)\s*\(\s*([A-Za-z]+)\s+County\s*\)", text, flags=re.I)
    if m:
        city = m.group(1).strip()
        c = m.group(2).strip().lower()
        if c in IN_COUNTIES:
            county = f"{c.title()} County"
        return (city, county)

    # 3) "City, IN" or "City Indiana"
    m = re.search(r"\b([A-Za-z][A-Za-z\s\.\-']{2,})\s*,?\s*(Indiana|IN)\b", text, flags=re.I)
    if m:
        city = m.group(1).strip()
    else:
        # 4) "in/near/around City"
        m = re.search(r"\b(?:in|near|around)\s+([A-Za-z][A-Za-z\s\.\-']+)", text, flags=re.I)
        if m:
            raw = m.group(1).strip()
            raw = re.split(r"\b(20\d{2}|this year|next year|last year|today|now|last|past|recent)\b", raw, flags=re.I)[0].strip()
            raw = re.sub(r"\b(Indiana|IN)\b\.?", "", raw, flags=re.I).strip()
            if raw:
                city = raw

    # 5) Bare county token anywhere
    if county is None:
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
        "agenda", "minutes", "packet", "staff report",
        "rezoning", "rezone", "pud", "petition", "variance", "special exception",
        "site plan", "development plan", "primary plat", "secondary plat",
        "permit", "building permit", "zoning", "zoning case",
        "being built", "under construction", "breaking ground", "breaks ground",
    ]
    return any(w in t for w in triggers)


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


def _parse_date_from_snippet(snippet: str) -> Optional[datetime]:
    """
    Very light fallback: tries to catch things like 'Oct 15, 2025' in snippet.
    """
    s = (snippet or "").strip()
    if not s:
        return None

    # Month name date
    m = re.search(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+(\d{1,2}),\s+(20\d{2})\b", s, re.I)
    if m:
        mon = m.group(1)[:3].title()
        day = int(m.group(2))
        year = int(m.group(3))
        try:
            dt = datetime.strptime(f"{mon} {day} {year}", "%b %d %Y")
            return dt
        except Exception:
            return None
    return None


def _days_to_date_restrict(days: Optional[int]) -> Optional[str]:
    # Google CSE dateRestrict supports d[number] or m[number]
    if not days or days <= 0:
        return None
    if days <= 31:
        return f"d{days}"
    months = max(1, int(round(days / 30)))
    if months <= 24:
        return f"m{months}"
    return None


def _throttle() -> None:
    global _LAST_REQUEST_TS
    now = time.time()
    elapsed = now - _LAST_REQUEST_TS
    if elapsed < CSE_MIN_INTERVAL_SECONDS:
        time.sleep(CSE_MIN_INTERVAL_SECONDS - elapsed)
    _LAST_REQUEST_TS = time.time()


def _safe_http_error_log(resp: requests.Response, prefix: str) -> None:
    # Never log full URL (contains key). Log only status + a short excerpt.
    try:
        body = (resp.text or "")[:200].replace("\n", " ")
    except Exception:
        body = ""
    log.warning("%s status=%s body=%s", prefix, resp.status_code, body)


def _google_cse_search(query: str, max_results: int = 30, days: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Google CSE wrapper with:
    - TTL cache
    - retry/backoff on 429/5xx
    - capped pages to prevent quota burn
    """
    if not GOOGLE_CSE_KEY or not GOOGLE_CSE_CX:
        log.warning(
            "GOOGLE_CSE_KEY or GOOGLE_CSE_CX not set; returning empty list. "
            f"GOOGLE_CSE_KEY present={bool(GOOGLE_CSE_KEY)}, GOOGLE_CSE_CX present={bool(GOOGLE_CSE_CX)}"
        )
        return []

    max_results = max(1, min(int(max_results), CSE_MAX_RESULTS_HARD_CAP))
    pages_needed = (max_results + 9) // 10
    pages = max(1, min(pages_needed, max(1, CSE_MAX_PAGES)))

    date_restrict = _days_to_date_restrict(days)

    cache_key = f"q={query}|days={days}|pages={pages}|sort={CSE_SORT}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    sess = requests.Session()
    base_params = {
        "key": GOOGLE_CSE_KEY,
        "cx": GOOGLE_CSE_CX,
        "q": query,
        "num": 10,
    }
    if CSE_SORT:
        base_params["sort"] = CSE_SORT
    if date_restrict:
        base_params["dateRestrict"] = date_restrict

    out: List[Dict[str, Any]] = []
    start = 1

    for _page_idx in range(pages):
        _throttle()

        params = dict(base_params)
        params["start"] = start

        attempt = 0
        while True:
            attempt += 1
            try:
                resp = sess.get(GOOGLE_CSE_ENDPOINT, params=params, timeout=CSE_TIMEOUT_SECONDS)
            except Exception as e:
                log.warning("Google CSE request failed (network) start=%s err=%s", start, e)
                break

            if resp.status_code == 200:
                try:
                    data = resp.json()
                except Exception as e:
                    log.warning("Google CSE JSON parse failed start=%s err=%s", start, e)
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
                    dt = _parse_date_from_pagemap(it) or _parse_date_from_snippet(snippet)
                    out.append({"title": title, "snippet": snippet, "url": url, "provider": provider, "date": dt})
                    if len(out) >= max_results:
                        break

                break  # success

            if resp.status_code in (429, 500, 502, 503, 504):
                _safe_http_error_log(resp, prefix="Google CSE transient error")
                if attempt >= CSE_MAX_RETRIES:
                    if resp.status_code == 429:
                        raise CSEQuotaError("Google CSE rate-limited (429). Reduce pages/queries or increase quota.")
                    break

                retry_after = resp.headers.get("Retry-After")
                wait: Optional[float] = None
                if retry_after:
                    try:
                        wait = float(retry_after)
                    except Exception:
                        wait = None

                if wait is None:
                    wait = min(CSE_BACKOFF_MAX_SECONDS, CSE_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))
                    wait += random.uniform(0.0, 0.6)

                time.sleep(wait)
                continue

            _safe_http_error_log(resp, prefix="Google CSE non-retryable error")
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


def _looks_junky(title: str, snippet: str, url: str) -> bool:
    text = _lower(f"{title} {snippet}")
    url_l = _lower(url)

    for bad in NEGATIVE_URL:
        if bad in url_l:
            return True
    for neg in NEGATIVE_TEXT:
        if neg in text:
            return True

    # strong job-listing heuristic
    if any(w in text for w in ["job", "jobs", "career", "careers", "apply now"]):
        if any(w in url_l for w in ["job", "jobs", "career", "careers"]):
            return True

    return False


def _looks_like_facility_hit(title: str, snippet: str, url: str) -> bool:
    if _looks_junky(title, snippet, url):
        return False

    text = _lower(f"{title} {snippet}")
    # require at least one strong positive phrase OR multiple industrial signals
    pos = sum(1 for p in FACILITY_POSITIVE if p in text)
    sig = sum(1 for s in INDUSTRIAL_SIGNALS if s in text)

    return (pos >= 1) or (sig >= 3)


def _looks_like_planning_doc(title: str, snippet: str, url: str) -> bool:
    if _looks_junky(title, snippet, url):
        return False

    text = _lower(f"{title} {snippet}")
    url_l = _lower(url)

    looks_platform = any(m in url_l for m in AGENDA_PLATFORMS) or any(m in text for m in AGENDA_PLATFORMS)
    has_planning_terms = any(k in text for k in ["agenda", "minutes", "packet", "staff report", "petition", "rezon", "hearing", "commission", "bza", "plan commission", "mdc"])
    pdfish = (url_l.endswith(".pdf") or ".pdf" in url_l)

    # at least one of: platform OR planning terms OR pdf, and some industrial signal
    industrial_signal = any(s in text for s in ["industrial", "warehouse", "distribution", "logistics", "manufacturing", "cold storage", "spec building"])

    return bool((looks_platform or has_planning_terms or pdfish) and industrial_signal)


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


def _score_text_industrial(title: str, snippet: str, url: str) -> int:
    text = _lower(f"{title} {snippet} {url}")
    score = 0

    # industrial signals
    for s in INDUSTRIAL_SIGNALS:
        if s in text:
            score += 2

    # size/jobs/investment pattern boosts
    if re.search(r"\b\d{2,4}[,\d]{0,4}\s*(square[-\s]?feet|sq\.?\s*ft|sf)\b", text):
        score += 4
    if re.search(r"\b\d{2,5}\s+(new\s+)?jobs\b", text):
        score += 3
    if re.search(r"\$\s*\d+(\.\d+)?\s*(million|billion)\b", text) or re.search(r"\b\d+\s*(million|billion)\s*(investment|invest)\b", text):
        score += 3

    # source quality hints
    if any(host in text for host in ["insideindianabusiness", "ibj.com", "in.gov", "co.", "in.us", "municode", "granicus"]):
        score += 2

    # pdf gets a tiny bump for planning docs
    if ".pdf" in text:
        score += 1

    return score


def _infer_project_type(title: str, snippet: str, is_planning: bool) -> str:
    text = _lower(title + " " + snippet)
    if is_planning:
        return "planning / zoning filing"
    if any(w in text for w in ("warehouse", "distribution center", "distribution facility", "fulfillment center", "cold storage", "logistics")):
        return "warehouse / logistics facility"
    if any(w in text for w in ("manufacturing plant", "manufacturing facility", "production plant", "assembly plant", "factory")):
        return "manufacturing plant"
    if any(w in text for w in ("industrial park", "business park", "industrial complex")):
        return "business / industrial park"
    return "industrial / commercial project"


def _extract_company_guess(title: str) -> Optional[str]:
    """
    Light heuristic: pull 'Company' from 'Company – Something...' or 'Company to ...'
    """
    t = (title or "").strip()
    if not t:
        return None

    # before dash
    m = re.match(r"^(.{2,80}?)\s*[–-]\s+.+$", t)
    if m:
        cand = m.group(1).strip()
        # avoid generic headers
        if len(cand.split()) <= 8 and not any(w in cand.lower() for w in ["indiana", "county", "agenda", "minutes", "packet"]):
            return cand

    # "X to build/expand/opens"
    m = re.match(r"^(.{2,80}?)\s+(to\s+(build|expand|open)|breaks\s+ground|announces)\b", t, flags=re.I)
    if m:
        cand = m.group(1).strip()
        if len(cand.split()) <= 8:
            return cand

    return None


def _tail_from_user_q(user_q: str, limit: int = 6) -> str:
    """
    Extracts a few non-stopword tokens from the user's question WITHOUT poisoning the query
    with years like 2026 or filler like 'happening'.
    """
    cleaned = re.sub(r"[“”\"']", " ", user_q or "")
    tokens = re.findall(r"[A-Za-z0-9]+", cleaned)
    extra: List[str] = []

    for tok in tokens:
        tl = tok.lower()

        # drop 4-digit years
        if re.fullmatch(r"(19|20)\d{2}", tl):
            continue

        if tl in STOPWORDS:
            continue

        # avoid very short tokens
        if len(tl) <= 2:
            continue

        extra.append(tok)

    return " ".join(extra[:limit])


def _build_facility_site_bias(city: Optional[str], county: Optional[str]) -> str:
    """
    Facility/news hits are higher quality when biased to real news + gov sources.
    Keep this light to avoid missing results.
    """
    sites = [
        "site:insideindianabusiness.com",
        "site:ibj.com",
        "site:indystar.com",
        "site:*.in.gov",
        "site:in.gov",
        "site:*.in.us",
        "site:in.us",
    ]

    # county gov sites (often host press releases + RFPs + zoning notices)
    if county:
        base = (county.split()[0] if county.split() else county).strip().lower()
        cslug = _slug(base)
        sites.extend([
            f"site:co.{cslug}.in.us",
            f"site:{cslug}county.in.gov",
            f"site:{cslug}county.in.us",
        ])

    # city gov sites sometimes have plan commission / news too
    if city:
        c = _slug(city)
        if c:
            sites.append(f"site:{c}.in.gov")

    return "(" + " OR ".join(sites) + ")"


def _build_planning_site_bias(county: Optional[str]) -> str:
    sites = [
        "site:*.in.gov", "site:in.gov", "site:*.in.us", "site:in.us",
        "site:meetings.municode.com", "site:*municodemeetings.com",
        "site:*.granicus.com", "site:granicus.com",
    ]
    if county:
        base = (county.split()[0] if county.split() else county).strip().lower()
        cslug = _slug(base)
        sites.extend([
            f"site:co.{cslug}.in.us",
            f"site:{cslug}county.in.gov",
            f"site:{cslug}county.in.us",
        ])
    return "(" + " OR ".join(sites) + ")"


def _build_facility_query(user_q: str, city: Optional[str], county: Optional[str]) -> str:
    parts: List[str] = ["Indiana", BASE_FACILITY_KEYWORDS]

    if county:
        parts.append(f'"{county}"')
    if city:
        parts.append(f'"{city}"')

    # “new” signals without overfitting to a year
    parts.append('(announces OR announced OR expansion OR expands OR expanding OR "breaks ground" OR groundbreaking OR "plans to" OR "to build" OR "under construction")')

    # avoid job board junk in-query
    parts.append('-(indeed OR linkedin OR "job openings" OR "apply now" OR careers OR "now hiring")')

    tail = _tail_from_user_q(user_q, limit=5)
    if tail:
        parts.append(tail)

    q = " ".join(parts)
    log.info("Google CSE facility query: %s", q)
    return q


def _build_planning_query(user_q: str, city: Optional[str], county: Optional[str]) -> str:
    parts: List[str] = ["Indiana", PLANNING_KEYWORDS]

    if county:
        parts.append(f'"{county}"')
    if city:
        parts.append(f'"{city}"')

    # anchor to industrial uses
    parts.append('(industrial OR warehouse OR distribution OR logistics OR manufacturing OR "cold storage" OR "spec building")')
    parts.append('("AgendaCenter" OR "ViewFile" OR "DocumentCenter" OR "Legistar" OR municode OR granicus OR packet OR agenda OR minutes OR "staff report")')

    # avoid job board junk
    parts.append('-(indeed OR linkedin OR "job openings" OR "apply now" OR careers)')

    tail = _tail_from_user_q(user_q, limit=4)
    if tail:
        parts.append(tail)

    q = " ".join(parts)
    log.info("Google CSE planning query: %s", q)
    return q


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
        else:
            if not _looks_like_facility_hit(title, snippet, url):
                continue

        geo_score, match_city, match_county = _geo_match_scores(title, snippet, city, county)
        industrial_score = _score_text_industrial(title, snippet, url)
        project_type = _infer_project_type(title, snippet, is_planning=is_planning)
        company_guess = _extract_company_guess(title)

        # timeline stage guess
        url_l = url.lower()
        stage = "planning filing" if is_planning else "announcement"
        if url_l.endswith(".pdf") or ".pdf" in url_l:
            stage = "agenda/packet (PDF)"
        elif "agenda" in _lower(title) or "minutes" in _lower(title):
            stage = "agenda/minutes"
        elif not isinstance(dt, datetime):
            stage = "not specified in snippet"

        # forklift score 1-5 derived from industrial_score
        forklift_score = max(1, min(5, 1 + (industrial_score // 6)))

        projects.append({
            "project_name": title or ("Untitled document" if is_planning else "Untitled project"),
            "company": company_guess,
            "project_type": project_type,
            "scope": "local" if geo_score > 0 else "statewide",
            "location_label": (county or city or "Indiana") if geo_score > 0 else "Indiana",
            "original_area_label": original_area_label,
            "forklift_score": forklift_score,
            "forklift_label": "Planning doc (industrial signal)" if is_planning else "Facility/news hit",
            "geo_match_score": geo_score,
            "match_city": match_city,
            "match_county": match_county,
            "sqft": None,
            "jobs": None,
            "investment": None,
            "timeline_stage": stage,
            "timeline_year": dt.year if isinstance(dt, datetime) else None,
            "raw_date": dt,
            "url": url,
            "provider": provider,
            "snippet": snippet,
            "source_tier": source_tier,
            "result_mode": "planning" if is_planning else "facility",
            "industrial_signal_score": industrial_score,
        })

    return projects


def _rank_projects(projects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    now = datetime.utcnow()

    def _sort_key(p: Dict[str, Any]) -> Tuple[int, int, int]:
        geo = int(p.get("geo_match_score") or 0)
        score = int(p.get("forklift_score") or 0)
        bonus = int(p.get("industrial_signal_score") or 0)
        dt = p.get("raw_date")
        age_days = 999999
        if isinstance(dt, datetime):
            age_days = max(0, (now - dt).days)
        # higher geo, higher score+bonus, newer first
        return (-geo, -(score * 10 + bonus), age_days)

    projects.sort(key=_sort_key)
    return projects


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def search_indiana_developments(user_q: str, days: int = 365, max_items: int = 30) -> List[Dict[str, Any]]:
    """
    Main entrypoint.

    Strategy:
    1) Try FACILITY/NEWS first for the specified geo (if present)
    2) If facility is thin AND a geo exists, automatically run PLANNING docs too
       (this is where most “real” early projects show up)
    3) If user explicitly asked planning-style, planning runs first
    4) Minimal statewide fallback only if local is empty
    """
    city, county = _extract_geo_hint(user_q)
    wants_planning = _is_planning_query(user_q)

    log.info("INTEL_VERSION=2026-01-12B geo city=%s county=%s planning_query=%s q=%r", city, county, wants_planning, user_q)

    max_items = max(1, min(int(max_items), CSE_MAX_RESULTS_HARD_CAP))
    projects_all: List[Dict[str, Any]] = []

    # --- Tier 1: local (planning-first if asked) ---
    try:
        if wants_planning:
            site_bias = _build_planning_site_bias(county)
            q = _build_planning_query(user_q, city, county)
            raw = _google_cse_search(f"{site_bias} {q}", max_results=max_items, days=days)
            raw = _dedupe_by_url(raw)
            projects_all.extend(_normalize(raw, city, county, user_q, source_tier="local", is_planning=True))
        else:
            site_bias = _build_facility_site_bias(city, county)
            q = _build_facility_query(user_q, city, county)
            raw = _google_cse_search(f"{site_bias} {q}", max_results=max_items, days=days)
            raw = _dedupe_by_url(raw)
            projects_all.extend(_normalize(raw, city, county, user_q, source_tier="local", is_planning=False))
    except CSEQuotaError as e:
        log.warning("CSE quota error on local tier: %s", e)
        return []

    # --- If facility is thin and geo exists, auto-run planning docs ---
    if (not wants_planning) and (city or county):
        facility_count = sum(1 for p in projects_all if p.get("result_mode") == "facility")
        if facility_count < 3:
            try:
                site_bias = _build_planning_site_bias(county)
                q = _build_planning_query(user_q, city, county)
                raw = _google_cse_search(f"{site_bias} {q}", max_results=max_items, days=days)
                raw = _dedupe_by_url(raw)
                projects_all.extend(_normalize(raw, city, county, user_q, source_tier="local", is_planning=True))
            except CSEQuotaError as e:
                log.warning("CSE quota error on planning assist tier: %s", e)

    projects_all = _rank_projects(_dedupe_projects(projects_all))
    if projects_all:
        return projects_all

    # --- Tier 2: statewide fallback (kept minimal) ---
    try:
        # facility fallback
        site_bias = _build_facility_site_bias(None, None)
        q_state = _build_facility_query(user_q, city=None, county=None)
        raw_state = _google_cse_search(f"{site_bias} {q_state}", max_results=max_items, days=days)
        raw_state = _dedupe_by_url(raw_state)
        projects_all.extend(_normalize(raw_state, None, None, user_q, source_tier="statewide", is_planning=False))

        # planning fallback
        site_bias_p = _build_planning_site_bias(None)
        q_state_p = _build_planning_query(user_q, city=None, county=None)
        raw_state_p = _google_cse_search(f"{site_bias_p} {q_state_p}", max_results=max_items, days=days)
        raw_state_p = _dedupe_by_url(raw_state_p)
        projects_all.extend(_normalize(raw_state_p, None, None, user_q, source_tier="statewide", is_planning=True))
    except CSEQuotaError as e:
        log.warning("CSE quota error on statewide tier: %s", e)
        return []

    projects_all = _rank_projects(_dedupe_projects(projects_all))
    return projects_all


def _dedupe_projects(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Dedupe by URL; if URL repeats between facility+planning queries, keep the higher score one.
    """
    best: Dict[str, Dict[str, Any]] = {}
    for it in items:
        u = (it.get("url") or "").split("#")[0].strip()
        if not u:
            continue
        if u not in best:
            best[u] = it
            continue
        # keep whichever has higher industrial score, then geo score
        a = best[u]
        if int(it.get("industrial_signal_score") or 0) > int(a.get("industrial_signal_score") or 0):
            best[u] = it
        elif int(it.get("geo_match_score") or 0) > int(a.get("geo_match_score") or 0):
            best[u] = it
    return list(best.values())


def render_developments_markdown(items: List[Dict[str, Any]]) -> str:
    if not items:
        return (
            "No results returned.\n"
            "- If your logs show 429s, Google CSE is rate-limiting you.\n"
            "- If geo is always None/None, your app is not running the file you think it is (look for INTEL_VERSION in logs).\n"
            "Try a planning-style question (agenda/packet) for faster proof of real projects."
        )

    now = datetime.utcnow()
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
        ptype = item.get("project_type") or "industrial / commercial project"
        score = item.get("forklift_score") or ""
        geo_score = item.get("geo_match_score") or 0
        tier = item.get("source_tier") or "unknown"
        mode = item.get("result_mode") or "unknown"
        company = item.get("company")

        dt = item.get("raw_date")
        age = None
        if isinstance(dt, datetime):
            try:
                age = max(0, (now - dt).days)
            except Exception:
                age = None

        lines.append(f"{i}. {title} — {loc}")
        meta_bits = [ptype, f"Mode: {mode}"]
        if company:
            meta_bits.append(f"Company: {company}")
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
        if age is not None:
            meta_bits.append(f"Age: {age}d")

        lines.append("   " + " • ".join(meta_bits))
        if snippet:
            lines.append(f"   Snippet: {snippet}")
        if url:
            lines.append(f"   URL: {url}")

    return "\n".join(lines)


__all__ = ["search_indiana_developments", "render_developments_markdown"]
