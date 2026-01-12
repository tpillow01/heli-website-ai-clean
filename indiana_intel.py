"""
indiana_intel.py

Indiana developments / lead finder for Tynan / Heli AI.

This version adds "richer notes" by enriching the top results:
- fetches the page text for top N hits (small, controlled)
- extracts: sqft, jobs, investment, dates, docket/case numbers
- grabs 1–3 relevant sentences mentioning warehouse/logistics/manufacturing terms
- injects that into output as Notes/Highlights

Controls (env vars):
- ENRICH_ENABLED=1|0
- ENRICH_FETCH_TOP_N=4
- ENRICH_TIMEOUT_SECONDS=8
- ENRICH_MAX_CHARS=22000
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

CSE_MAX_PAGES = int(os.environ.get("CSE_MAX_PAGES", "1"))  # each page = 10 results
CSE_MAX_RESULTS_HARD_CAP = 50

CACHE_TTL_SECONDS = int(os.environ.get("CSE_CACHE_TTL_SECONDS", "900"))  # 15 min

ENABLE_SITE_DISCOVERY = os.environ.get("CSE_ENABLE_SITE_DISCOVERY", "1").strip() not in {"0", "false", "False"}

# ---------------------------------------------------------------------------
# Enrichment controls (richer notes)
# ---------------------------------------------------------------------------

ENRICH_ENABLED = os.environ.get("ENRICH_ENABLED", "1").strip() not in {"0", "false", "False"}
ENRICH_FETCH_TOP_N = int(os.environ.get("ENRICH_FETCH_TOP_N", "4"))
ENRICH_TIMEOUT_SECONDS = float(os.environ.get("ENRICH_TIMEOUT_SECONDS", "8"))
ENRICH_MAX_CHARS = int(os.environ.get("ENRICH_MAX_CHARS", "22000"))

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

HIGHLIGHT_TERMS = [
    "warehouse", "distribution", "logistics", "manufactur", "plant", "factory",
    "cold storage", "spec building", "fulfillment", "data center", "industrial park",
    "3pl", "groundbreaking", "ribbon cutting", "expansion", "now open",
    "rezon", "site plan", "staff report", "petition", "variance", "docket",
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
    if not days or days <= 0:
        return None
    if days <= 31:
        return f"d{days}"
    months = max(1, int(round(days / 30)))
    if months <= 24:
        return f"m{months}"
    return "y2"


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
    s = snippet or ""
    # Jan 5, 2026 / January 5, 2026
    m = re.search(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+(20\d{2})\b", s, re.I)
    if m:
        raw = m.group(0)
        for fmt in ("%b %d, %Y", "%B %d, %Y"):
            try:
                return datetime.strptime(raw, fmt)
            except Exception:
                continue
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

    m = re.search(r"\b([a-z]+)\s+county\b", tl, flags=re.I)
    if m:
        c = m.group(1).strip().lower()
        if c in IN_COUNTIES:
            county = f"{c.title()} County"

    m = re.search(r"\b([A-Za-z][A-Za-z\s\.\-']{2,})\s*,?\s*(Indiana|IN)\b", text, flags=re.I)
    if m:
        city = m.group(1).strip()

    if not city:
        m = re.search(r"\b(?:in|near|around)\s+([A-Za-z][A-Za-z\s\.\-']+)", text, flags=re.I)
        if m:
            raw = m.group(1).strip()
            raw = re.split(r"\b(20\d{2}|this year|next year|last year|today|now|last|past|recent)\b", raw, flags=re.I)[0].strip()
            raw = re.sub(r"\b(Indiana|IN)\b\.?", "", raw, flags=re.I).strip()
            if raw:
                city = raw

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
# Site discovery
# ---------------------------------------------------------------------------

def _discover_local_sites(city: Optional[str], county: Optional[str], days: int) -> List[str]:
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
# Enrichment (richer notes)
# ---------------------------------------------------------------------------

_ENRICH_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}


def _enrich_cache_get(url: str) -> Optional[Dict[str, Any]]:
    hit = _ENRICH_CACHE.get(url)
    if not hit:
        return None
    ts, val = hit
    if time.time() - ts > CACHE_TTL_SECONDS:
        _ENRICH_CACHE.pop(url, None)
        return None
    return val


def _enrich_cache_set(url: str, val: Dict[str, Any]) -> None:
    _ENRICH_CACHE[url] = (time.time(), val)


def _strip_html_to_text(html: str) -> str:
    if not html:
        return ""
    # remove scripts/styles
    html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
    # remove tags
    text = re.sub(r"(?is)<[^>]+>", " ", html)
    # decode basic entities
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&quot;", '"').replace("&#39;", "'")
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_numbers(text: str) -> Dict[str, Optional[str]]:
    """
    Pull common project signals from raw text.
    Returns strings to keep it safe/simple for display.
    """
    t = text or ""

    # sqft patterns: 1,200,000 square feet / 1.2 million sq ft / 500k sf
    sqft = None
    m = re.search(r"\b(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(million|m)?\s*(square feet|sq\.?\s*ft|sqft|sf)\b", t, re.I)
    if m:
        num = m.group(1)
        mult = (m.group(2) or "").lower()
        if mult in {"million", "m"}:
            sqft = f"{num} million sq ft"
        else:
            sqft = f"{num} sq ft"

    # jobs patterns: 250 jobs / 1,000 new jobs
    jobs = None
    m = re.search(r"\b(\d{2,5}(?:,\d{3})?)\s+(new\s+)?jobs\b", t, re.I)
    if m:
        jobs = f"{m.group(1)} jobs"

    # investment: $120 million / $1.2B
    investment = None
    m = re.search(r"\$\s?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(billion|bn|b|million|m)?\b", t, re.I)
    if m:
        amt = m.group(1)
        unit = (m.group(2) or "").lower()
        if unit in {"billion", "bn", "b"}:
            investment = f"${amt}B"
        elif unit in {"million", "m"}:
            investment = f"${amt}M"
        else:
            investment = f"${amt}"

    return {"sqft": sqft, "jobs": jobs, "investment": investment}


def _extract_case_numbers(text: str) -> List[str]:
    """
    Planning/permit style identifiers vary by city/county.
    We just try to catch common patterns.
    """
    t = text or ""
    pats = [
        r"\b(?:DP|PZ|PC|BZA|ZA|Z|REZ|PUD|DEV|DEVPLAN)[-\s]?\d{1,5}(?:-\d{1,5})?\b",
        r"\b\d{2}-\d{3,5}\b",
        r"\b\d{4}-\d{3,6}\b",
    ]
    found: List[str] = []
    for p in pats:
        for m in re.findall(p, t, flags=re.I):
            s = m.strip()
            if s and s not in found:
                found.append(s)
    return found[:6]


def _extract_dates(text: str) -> List[str]:
    """
    Extract a few readable dates from page text.
    """
    t = text or ""
    dates: List[str] = []
    # Month Day, Year
    for m in re.findall(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+20\d{2}\b", t, flags=re.I):
        # m is month name only due to group; re-find full match in a safer way:
        pass

    for m in re.finditer(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+20\d{2}\b", t, flags=re.I):
        s = m.group(0)
        if s not in dates:
            dates.append(s)

    for m in re.finditer(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", t):
        s = m.group(0)
        if s not in dates:
            dates.append(s)

    return dates[:6]


def _extract_highlight_sentences(text: str, max_sentences: int = 3) -> List[str]:
    """
    Pull up to N sentences that mention high-signal terms.
    """
    if not text:
        return []
    # crude sentence split
    parts = re.split(r"(?<=[\.\!\?])\s+", text)
    hits: List[str] = []
    for s in parts:
        sl = s.lower()
        if any(term in sl for term in HIGHLIGHT_TERMS):
            s2 = s.strip()
            if len(s2) < 25:
                continue
            if s2 not in hits:
                hits.append(s2)
        if len(hits) >= max_sentences:
            break
    return hits


def _fetch_url_text(url: str) -> str:
    """
    Lightweight fetch (HTML only). PDFs often return binary; we skip heavy PDF parsing here.
    """
    if not url:
        return ""

    cached = _enrich_cache_get(url)
    if cached and "text" in cached:
        return str(cached.get("text") or "")

    dom = _domain(url)
    if any(bad in dom for bad in BLOCK_DOMAINS):
        return ""

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; IndianaIntelBot/1.0; +https://example.com)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=ENRICH_TIMEOUT_SECONDS)
    except Exception as e:
        log.info("Enrich fetch failed url=%s err=%s", url, e)
        return ""

    if resp.status_code != 200:
        return ""

    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "pdf" in ctype or url.lower().endswith(".pdf"):
        # skip heavy PDF parsing (still can show case numbers from URL/snippet elsewhere)
        return ""

    html = resp.text or ""
    text = _strip_html_to_text(html)
    if len(text) > ENRICH_MAX_CHARS:
        text = text[:ENRICH_MAX_CHARS]

    _enrich_cache_set(url, {"text": text})
    return text


def _build_notes_from_enrichment(title: str, snippet: str, url: str, is_planning: bool) -> Dict[str, Any]:
    """
    Returns a dict with richer fields you can display as Notes.
    """
    base = {
        "highlights": [],
        "sqft": None,
        "jobs": None,
        "investment": None,
        "dates": [],
        "case_numbers": [],
    }

    if not ENRICH_ENABLED:
        return base

    text = _fetch_url_text(url)
    if not text:
        # still try to extract basics from snippet/title
        nums = _extract_numbers(f"{title} {snippet}")
        base.update(nums)
        base["case_numbers"] = _extract_case_numbers(f"{title} {snippet} {url}") if is_planning else []
        base["dates"] = _extract_dates(snippet)
        base["highlights"] = _extract_highlight_sentences(f"{title}. {snippet}", max_sentences=2)
        return base

    nums = _extract_numbers(text)
    base.update(nums)

    if is_planning:
        base["case_numbers"] = _extract_case_numbers(text) or _extract_case_numbers(f"{title} {snippet} {url}")

    base["dates"] = _extract_dates(text) or _extract_dates(snippet)

    # highlights: prefer page text; fallback to snippet
    base["highlights"] = _extract_highlight_sentences(text, max_sentences=3)
    if not base["highlights"]:
        base["highlights"] = _extract_highlight_sentences(f"{title}. {snippet}", max_sentences=2)

    return base


# ---------------------------------------------------------------------------
# Ranking
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
        return True
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

        if not is_planning and not _looks_industrial(title, snippet):
            continue

        if not _recency_ok(dt, days):
            continue

        score = _compute_score(title, snippet, url, city, county, is_planning)

        scored.append({
            "title": title,
            "snippet": snippet,
            "url": url,
            "provider": it.get("provider") or _domain(url),
            "date": dt,
            "_score": score,
            "_age_days": (now - dt).days if isinstance(dt, datetime) else 9999,
        })

    scored.sort(key=lambda x: (-int(x.get("_score", 0)), int(x.get("_age_days", 9999))))
    return scored


def _enrich_top_results(ranked_raw: List[Dict[str, Any]], is_planning: bool) -> List[Dict[str, Any]]:
    """
    Enrich top N results for richer notes.
    """
    if not ranked_raw:
        return ranked_raw
    if not ENRICH_ENABLED or ENRICH_FETCH_TOP_N <= 0:
        return ranked_raw

    n = min(ENRICH_FETCH_TOP_N, len(ranked_raw))
    for i in range(n):
        r = ranked_raw[i]
        try:
            notes = _build_notes_from_enrichment(r.get("title", ""), r.get("snippet", ""), r.get("url", ""), is_planning=is_planning)
        except Exception as e:
            log.info("Enrichment failed url=%s err=%s", r.get("url"), e)
            notes = {"highlights": [], "sqft": None, "jobs": None, "investment": None, "dates": [], "case_numbers": []}
        r["notes"] = notes
    return ranked_raw


def _to_project_objects(ranked_raw: List[Dict[str, Any]], city: Optional[str], county: Optional[str], is_planning: bool) -> List[Dict[str, Any]]:
    """
    Convert ranked raw hits into your project dict format.
    """
    projects: List[Dict[str, Any]] = []
    loc = county or city or "Indiana"

    for r in ranked_raw:
        title = r.get("title") or "Untitled"
        snippet = r.get("snippet") or ""
        url = r.get("url") or ""
        provider = r.get("provider") or _domain(url)
        dt = r.get("date")
        notes = r.get("notes") or {}

        # forklift score (1-5)
        score = int(r.get("_score", 0))
        forklift_score = max(1, min(5, 1 + score // 4))

        # timeline
        stage = "planning doc" if is_planning else "announcement/news"
        year = dt.year if isinstance(dt, datetime) else None

        projects.append({
            "project_name": title,
            "company": None,
            "project_type": "planning / zoning filing" if is_planning else "warehouse / industrial facility",
            "location_label": loc,
            "geo_match_score": _geo_score(title, snippet, city, county),
            "forklift_score": forklift_score,
            "timeline_year": year,
            "timeline_stage": stage,
            "raw_date": dt,
            "url": url,
            "provider": provider,
            "snippet": snippet,
            "source_tier": "local",
            "result_mode": "planning" if is_planning else "facility",

            # richer note fields
            "notes_highlights": notes.get("highlights") or [],
            "notes_sqft": notes.get("sqft"),
            "notes_jobs": notes.get("jobs"),
            "notes_investment": notes.get("investment"),
            "notes_dates": notes.get("dates") or [],
            "notes_case_numbers": notes.get("case_numbers") or [],
        })

    return projects


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_indiana_developments(user_q: str, days: int = 365, max_items: int = 25) -> List[Dict[str, Any]]:
    city, county = _extract_geo_hint(user_q)
    is_planning = _is_planning_query(user_q)

    log.info("Geo hint: city=%s county=%s planning=%s", city, county, is_planning)

    max_items = max(1, min(int(max_items), CSE_MAX_RESULTS_HARD_CAP))

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
        ranked = _enrich_top_results(ranked, is_planning=True)
        projects = _to_project_objects(ranked, city, county, is_planning=True)
        return projects[:15]

    # facility/news mode
    area = city or county or "Indiana"
    q = (
        f'{area} Indiana {FACILITY_KEYWORDS} '
        f'(announced OR expansion OR groundbreaking OR "now open" OR "ribbon cutting") '
        f'-jobs -hiring -indeed -linkedin'
    )
    log.info("Facility query: %s", q)

    try:
        raw = _google_cse_search(q, max_results=max_items, days=days)
    except CSEQuotaError:
        return []

    raw = _dedupe_by_url(raw)
    ranked = _rank(raw, city, county, is_planning=False, days=days)
    ranked = _enrich_top_results(ranked, is_planning=False)
    projects = _to_project_objects(ranked, city, county, is_planning=False)
    return projects[:15]

def _make_notes_line(item: Dict[str, Any]) -> str:
    """
    Produce a single, rich 'Notes:' line like:
    'Jul 25, 2024 ... US Cold Storage to expand Lebanon warehouse; welcomes Gorton's processing ... Size: 120,000 sq ft • Jobs: 80'
    """
    # Pick best "source sentence"
    highlights = item.get("notes_highlights") or []
    snippet = (item.get("snippet") or "").strip()

    base_text = ""
    if highlights:
        base_text = highlights[0].strip()
    elif snippet:
        base_text = snippet

    # Ensure it starts with a date-ish prefix if we have it
    dt = item.get("raw_date")
    date_prefix = ""
    if isinstance(dt, datetime):
        date_prefix = dt.strftime("%b %d, %Y")

    # Avoid duplicating date if snippet already begins with it
    if date_prefix and base_text and date_prefix.lower() not in base_text[:40].lower():
        base_text = f"{date_prefix} ... {base_text}"
    elif base_text and not base_text.startswith(("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")) and date_prefix:
        base_text = f"{date_prefix} ... {base_text}"

    # Append concise hard facts
    extras = []
    if item.get("notes_sqft"):
        extras.append(f"Size: {item['notes_sqft']}")
    if item.get("notes_jobs"):
        extras.append(f"Jobs: {item['notes_jobs']}")
    if item.get("notes_investment"):
        extras.append(f"Investment: {item['notes_investment']}")
    if item.get("notes_case_numbers"):
        extras.append(f"Case/Docket: {', '.join(item['notes_case_numbers'][:3])}")

    # Optional: if we extracted dates from page text, include the most relevant one
    # (only if we don't already have raw_date)
    if not isinstance(dt, datetime):
        dates_seen = item.get("notes_dates") or []
        if dates_seen:
            extras.append(f"Date: {dates_seen[0]}")

    notes = base_text.strip()

    # Clean up overly long notes
    notes = re.sub(r"\s+", " ", notes)
    if len(notes) > 240:
        notes = notes[:237].rstrip() + "..."

    if extras:
        notes = f"{notes} • " + " • ".join(extras)

    # If nothing at all, return a fallback
    return notes if notes else "No additional notes available from snippet/page text."


def render_developments_markdown(items: List[Dict[str, Any]]) -> str:
    if not items:
        return (
            "No results returned for that specific area/time window.\n"
            "Tip: try a planning-style question (agenda/packet) or increase days to 730."
        )

    lines: List[str] = []

    for i, item in enumerate(items[:15], start=1):
        title = item.get("project_name") or "Untitled"
        url = item.get("url") or ""
        provider = item.get("provider") or ""
        year = item.get("timeline_year")
        stage = item.get("timeline_stage") or ""
        loc = item.get("location_label") or "Indiana"
        ptype = item.get("project_type") or "Industrial / commercial project"

        # --- Format header like your example ---
        lines.append(f"{title} – {loc}")
        lines.append(f"Type: {ptype}")
        lines.append("Company / Developer: not specified in snippet")
        lines.append("Scope: not specified in snippet")

        timeline = stage
        if year:
            timeline = f"{stage} ({year})"
        lines.append(f"Timeline: {timeline}")

        if url:
            lines.append(f"Source: {url}")
        else:
            lines.append("Source: not specified in snippet")

        # --- The exact thing you want improved ---
        notes_line = _make_notes_line(item)
        lines.append(f"Notes: {notes_line}")

        # Blank line between results
        lines.append("")

    return "\n".join(lines).strip()
