"""
indiana_intel.py (STRICT GEO VERSION)

What this version fixes:
- If the user asks for a specific city/county, results MUST match that geo (no more "same results").
- Filters out common junk sources (job boards) and irrelevant "warehouse licensing" / grain pages.
- Logs a version string AT IMPORT TIME so you can confirm Render is running THIS file.

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
# Version (import-time proof)
# -----------------------------------------------------------------------------
INTEL_VERSION = "2026-01-12_STRICT_GEO_V1"

log = logging.getLogger("indiana_intel")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

log.info("Loaded indiana_intel.py INTEL_VERSION=%s", INTEL_VERSION)


# -----------------------------------------------------------------------------
# Google CSE Config
# -----------------------------------------------------------------------------
GOOGLE_CSE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"
GOOGLE_CSE_KEY = os.environ.get("GOOGLE_CSE_KEY")
GOOGLE_CSE_CX = os.environ.get("GOOGLE_CSE_CX")

CSE_TIMEOUT_SECONDS = float(os.environ.get("CSE_TIMEOUT_SECONDS", "10"))
CSE_MAX_RETRIES = int(os.environ.get("CSE_MAX_RETRIES", "4"))
CSE_BACKOFF_BASE_SECONDS = float(os.environ.get("CSE_BACKOFF_BASE_SECONDS", "1.2"))
CSE_BACKOFF_MAX_SECONDS = float(os.environ.get("CSE_BACKOFF_MAX_SECONDS", "20"))
CSE_MIN_INTERVAL_SECONDS = float(os.environ.get("CSE_MIN_INTERVAL_SECONDS", "0.5"))

# Keep small to avoid quota burn (10 per page)
CSE_MAX_PAGES = int(os.environ.get("CSE_MAX_PAGES", "1"))
CSE_MAX_RESULTS_HARD_CAP = 50

CACHE_TTL_SECONDS = int(os.environ.get("CSE_CACHE_TTL_SECONDS", "900"))


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

PLANNING_KEYWORDS = (
    '("plan commission" OR "area plan commission" OR "planning commission" OR '
    '"board of zoning appeals" OR BZA OR "metropolitan development commission" OR MDC OR '
    "agenda OR minutes OR docket OR petition OR rezoning OR rezone OR PUD OR "
    '"development plan" OR "site plan" OR "staff report" OR "public hearing" OR ordinance OR '
    "variance OR "
    '"primary plat" OR "secondary plat" OR "concept plan" OR "zoning case" OR '
    '"special exception" OR "improvement location permit" OR "building permit")'
)

AGENDA_PLATFORMS = [
    "agendacenter", "viewfile", "documentcenter", "legistar", "municode",
    "granicus", "minutes", "agenda", "packet", "meeting", "hearing",
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

INDUSTRIAL_SIGNALS = [
    "industrial", "warehouse", "distribution", "logistics", "cold storage", "fulfillment",
    "spec building", "truck terminal", "manufacturing", "plant", "factory", "3pl",
    "groundbreaking", "breaks ground", "under construction",
    "square feet", "sq ft", "sf",
]

# Strong junk filters (THIS is why your output was getting “grain warehouse licensing” pages)
NEGATIVE_URL = [
    "facebook.com", "instagram.com", "twitter.com", "x.com", "youtube.com", "tripadvisor.com",
    "indeed.com", "linkedin.com", "ziprecruiter.com", "glassdoor.com", "simplyhired.com",
    "monster.com", "careerbuilder.com", "jobs.", "/jobs", "/careers",
]

NEGATIVE_TEXT = [
    "tourism", "visitors bureau", "parks and recreation",
    "shopping center", "mall", "hotel", "resort",
    "apartments", "subdivision", "condo",
    "school", "university", "college",
    "hospital", "clinic",
    # the repeat offender in your results:
    "grain buyers", "grain licensing", "warehouse licensing", "grain licensee",
    "indiana grain", "igbwla",
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
# Cache + throttle
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


def _throttle() -> None:
    global _LAST_REQUEST_TS
    now = time.time()
    elapsed = now - _LAST_REQUEST_TS
    if elapsed < CSE_MIN_INTERVAL_SECONDS:
        time.sleep(CSE_MIN_INTERVAL_SECONDS - elapsed)
    _LAST_REQUEST_TS = time.time()


class CSEQuotaError(RuntimeError):
    pass


def _lower(s: Any) -> str:
    return str(s or "").lower()


def _days_to_date_restrict(days: Optional[int]) -> Optional[str]:
    if not days or days <= 0:
        return None
    if days <= 31:
        return f"d{days}"
    months = max(1, int(round(days / 30)))
    if months <= 24:
        return f"m{months}"
    return None


def _safe_http_error_log(resp: requests.Response, prefix: str) -> None:
    try:
        body = (resp.text or "")[:200].replace("\n", " ")
    except Exception:
        body = ""
    log.warning("%s status=%s body=%s", prefix, resp.status_code, body)


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


def _google_cse_search(query: str, max_results: int = 30, days: Optional[int] = None) -> List[Dict[str, Any]]:
    if not GOOGLE_CSE_KEY or not GOOGLE_CSE_CX:
        log.warning("Missing GOOGLE_CSE_KEY or GOOGLE_CSE_CX; returning empty.")
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
                    log.warning("CSE JSON parse failed start=%s err=%s", start, e)
                    break

                items = data.get("items", []) or []
                log.info("CSE returned %s items start=%s", len(items), start)
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
                break  # success

            if resp.status_code in (429, 500, 502, 503, 504):
                _safe_http_error_log(resp, "CSE transient error")
                if attempt >= CSE_MAX_RETRIES:
                    if resp.status_code == 429:
                        raise CSEQuotaError("Rate-limited (429).")
                    break
                wait = min(CSE_BACKOFF_MAX_SECONDS, CSE_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))
                wait += random.uniform(0.0, 0.6)
                time.sleep(wait)
                continue

            _safe_http_error_log(resp, "CSE non-retryable error")
            break

        start += 10
        if len(out) >= max_results:
            break

    _cache_set(cache_key, out)
    return out


# -----------------------------------------------------------------------------
# Geo parsing
# -----------------------------------------------------------------------------
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
    if not q:
        return (None, None)

    text = q.strip()
    tl = text.lower()

    city: Optional[str] = None
    county: Optional[str] = None

    m = re.search(r"\b([a-z]+)\s+county\b", tl, flags=re.I)
    if m:
        c = m.group(1).strip().lower()
        if c in IN_COUNTIES:
            county = f"{c.title()} County"

    m = re.search(r"\b([A-Za-z][A-Za-z\s\.\-']+?)\s*\(\s*([A-Za-z]+)\s+County\s*\)", text, flags=re.I)
    if m:
        city = m.group(1).strip()
        c = m.group(2).strip().lower()
        if c in IN_COUNTIES:
            county = f"{c.title()} County"
        return (city, county)

    m = re.search(r"\b([A-Za-z][A-Za-z\s\.\-']{2,})\s*,?\s*(Indiana|IN)\b", text, flags=re.I)
    if m:
        city = m.group(1).strip()
    else:
        m = re.search(r"\b(?:in|near|around)\s+([A-Za-z][A-Za-z\s\.\-']+)", text, flags=re.I)
        if m:
            raw = m.group(1).strip()
            raw = re.split(r"\b(20\d{2}|this year|next year|last year|today|now|last|past|recent)\b", raw, flags=re.I)[0].strip()
            raw = re.sub(r"\b(Indiana|IN)\b\.?", "", raw, flags=re.I).strip()
            if raw:
                city = raw

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
        "plan commission", "planning commission", "bza", "agenda", "minutes", "packet", "staff report",
        "rezoning", "variance", "special exception", "site plan", "development plan", "plat", "permit",
        "under construction", "breaking ground", "being built",
    ]
    return any(w in t for w in triggers)


# -----------------------------------------------------------------------------
# Filtering + scoring
# -----------------------------------------------------------------------------
def _looks_junky(title: str, snippet: str, url: str) -> bool:
    text = _lower(f"{title} {snippet}")
    url_l = _lower(url)

    if any(b in url_l for b in NEGATIVE_URL):
        return True
    if any(n in text for n in NEGATIVE_TEXT):
        return True

    # extra job board heuristic
    if any(w in text for w in ["job", "jobs", "career", "apply now"]) and any(w in url_l for w in ["job", "jobs", "career"]):
        return True

    return False


def _geo_match_scores(title: str, snippet: str, city: Optional[str], county: Optional[str]) -> Tuple[int, bool, bool]:
    text = _lower(title + " " + snippet)
    match_city = False
    match_county = False

    if county:
        base = county.split()[0].lower()
        match_county = (base in text) or (county.lower() in text)

    if city:
        c = city.lower()
        match_city = (c in text)

    if match_city and match_county:
        return (2, True, True)
    if match_city or match_county:
        return (1, match_city, match_county)
    return (0, False, False)


def _score_industrial(title: str, snippet: str, url: str) -> int:
    text = _lower(f"{title} {snippet} {url}")
    score = 0
    for s in INDUSTRIAL_SIGNALS:
        if s in text:
            score += 2
    if re.search(r"\b\d{2,4}[,\d]{0,4}\s*(square[-\s]?feet|sq\.?\s*ft|sf)\b", text):
        score += 4
    if re.search(r"\b\d{2,5}\s+(new\s+)?jobs\b", text):
        score += 3
    if ".pdf" in text:
        score += 1
    return score


def _looks_like_facility_hit(title: str, snippet: str, url: str) -> bool:
    if _looks_junky(title, snippet, url):
        return False
    text = _lower(f"{title} {snippet}")
    pos = sum(1 for p in FACILITY_POSITIVE if p in text)
    sig = sum(1 for s in INDUSTRIAL_SIGNALS if s in text)
    return (pos >= 1) or (sig >= 3)


def _looks_like_planning_doc(title: str, snippet: str, url: str) -> bool:
    if _looks_junky(title, snippet, url):
        return False
    text = _lower(f"{title} {snippet}")
    url_l = _lower(url)
    looks_platform = any(m in url_l for m in AGENDA_PLATFORMS) or any(m in text for m in AGENDA_PLATFORMS)
    has_planning = any(k in text for k in ["agenda", "minutes", "packet", "staff report", "petition", "rezon", "hearing", "commission", "bza", "plan commission", "mdc"])
    industrial_signal = any(s in text for s in ["industrial", "warehouse", "distribution", "logistics", "manufacturing", "cold storage", "spec building"])
    return bool((looks_platform or has_planning or url_l.endswith(".pdf")) and industrial_signal)


def _tail_from_user_q(user_q: str, limit: int = 5) -> str:
    cleaned = re.sub(r"[“”\"']", " ", user_q or "")
    tokens = re.findall(r"[A-Za-z0-9]+", cleaned)
    extra: List[str] = []
    for tok in tokens:
        tl = tok.lower()
        if re.fullmatch(r"(19|20)\d{2}", tl):
            continue
        if tl in STOPWORDS:
            continue
        if len(tl) <= 2:
            continue
        extra.append(tok)
    return " ".join(extra[:limit])


def _build_facility_query(user_q: str, city: Optional[str], county: Optional[str]) -> str:
    parts = ["Indiana", BASE_FACILITY_KEYWORDS]
    if county:
        parts.append(f'"{county}"')
    if city:
        parts.append(f'"{city}"')
    parts.append('(announces OR announced OR expansion OR expands OR "breaks ground" OR groundbreaking OR "plans to" OR "to build" OR "under construction")')
    parts.append('-(indeed OR linkedin OR careers OR "apply now" OR "grain buyers" OR "grain licensing")')

    tail = _tail_from_user_q(user_q, limit=4)
    if tail:
        parts.append(tail)

    q = " ".join(parts)
    log.info("Facility query=%s", q)
    return q


def _build_planning_query(user_q: str, city: Optional[str], county: Optional[str]) -> str:
    parts = ["Indiana", PLANNING_KEYWORDS]
    if county:
        parts.append(f'"{county}"')
    if city:
        parts.append(f'"{city}"')
    parts.append('(industrial OR warehouse OR distribution OR logistics OR manufacturing OR "cold storage" OR "spec building")')
    parts.append('("AgendaCenter" OR "ViewFile" OR "DocumentCenter" OR municode OR granicus OR packet OR agenda OR minutes OR "staff report")')
    parts.append('-(indeed OR linkedin OR careers OR "apply now")')

    tail = _tail_from_user_q(user_q, limit=3)
    if tail:
        parts.append(tail)

    q = " ".join(parts)
    log.info("Planning query=%s", q)
    return q


def _infer_project_type(title: str, snippet: str, is_planning: bool) -> str:
    text = _lower(title + " " + snippet)
    if is_planning:
        return "planning / zoning filing"
    if any(w in text for w in ("warehouse", "distribution center", "fulfillment", "cold storage", "logistics")):
        return "warehouse / logistics facility"
    if any(w in text for w in ("manufacturing plant", "manufacturing facility", "factory", "assembly plant")):
        return "manufacturing plant"
    if any(w in text for w in ("industrial park", "business park", "industrial complex")):
        return "business / industrial park"
    return "industrial / commercial project"


def _normalize(
    raw_items: List[Dict[str, Any]],
    city: Optional[str],
    county: Optional[str],
    user_q: str,
    source_tier: str,
    is_planning: bool,
    require_geo: bool,
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

        # ✅ THIS is the key fix: if user asked Boone/Hendricks/Plainfield/etc, we reject non-matching results.
        if require_geo and geo_score == 0:
            continue

        industrial_score = _score_industrial(title, snippet, url)
        forklift_score = max(1, min(5, 1 + (industrial_score // 6)))

        stage = "planning filing" if is_planning else "announcement"
        url_l = url.lower()
        if url_l.endswith(".pdf") or ".pdf" in url_l:
            stage = "agenda/packet (PDF)"
        elif "agenda" in _lower(title) or "minutes" in _lower(title):
            stage = "agenda/minutes"
        elif not isinstance(dt, datetime):
            stage = "not specified in snippet"

        projects.append({
            "project_name": title or ("Untitled document" if is_planning else "Untitled project"),
            "company": None,
            "project_type": _infer_project_type(title, snippet, is_planning),
            "scope": "local",
            "location_label": (county or city or "Indiana"),
            "original_area_label": original_area_label,
            "forklift_score": forklift_score,
            "forklift_label": "Planning doc" if is_planning else "Facility/news hit",
            "geo_match_score": geo_score,
            "match_city": match_city,
            "match_county": match_county,
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
        return (-geo, -(score * 10 + bonus), age_days)

    projects.sort(key=_sort_key)
    return projects


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


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def search_indiana_developments(user_q: str, days: int = 365, max_items: int = 30) -> List[Dict[str, Any]]:
    city, county = _extract_geo_hint(user_q)
    wants_planning = _is_planning_query(user_q)

    require_geo = bool(city or county)

    log.info(
        "search_indiana_developments INTEL_VERSION=%s city=%s county=%s require_geo=%s planning=%s q=%r",
        INTEL_VERSION, city, county, require_geo, wants_planning, user_q
    )

    max_items = max(1, min(int(max_items), CSE_MAX_RESULTS_HARD_CAP))

    projects: List[Dict[str, Any]] = []

    # 1) Facility/news first unless user clearly asked planning
    try:
        if not wants_planning:
            q = _build_facility_query(user_q, city, county)
            raw = _google_cse_search(q, max_results=max_items, days=days)
            raw = _dedupe_by_url(raw)
            projects.extend(_normalize(raw, city, county, user_q, "local", is_planning=False, require_geo=require_geo))

        # 2) Planning docs as backup (or primary if asked)
        q2 = _build_planning_query(user_q, city, county)
        raw2 = _google_cse_search(q2, max_results=max_items, days=days)
        raw2 = _dedupe_by_url(raw2)
        projects.extend(_normalize(raw2, city, county, user_q, "local", is_planning=True, require_geo=require_geo))

    except CSEQuotaError as e:
        log.warning("CSE quota error: %s", e)
        return []

    projects = _rank_projects(_dedupe_projects(projects))
    return projects


def _dedupe_projects(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for it in items:
        u = (it.get("url") or "").split("#")[0].strip()
        if not u:
            continue
        if u not in best:
            best[u] = it
            continue
        a = best[u]
        if int(it.get("industrial_signal_score") or 0) > int(a.get("industrial_signal_score") or 0):
            best[u] = it
    return list(best.values())


def render_developments_markdown(items: List[Dict[str, Any]]) -> str:
    if not items:
        return (
            "No matching results returned for that specific city/county.\n"
            "This is expected sometimes — and it’s better than showing the same statewide junk.\n"
            "Tip: try a planning-style query (agenda/packet/staff report) for that city/county."
        )

    lines: List[str] = []
    lines.append("Industrial / logistics results (geo-matched):")

    for i, item in enumerate(items[:15], start=1):
        title = item.get("project_name") or "Untitled"
        snippet = (item.get("snippet") or "").strip()
        url = item.get("url") or ""
        provider = item.get("provider") or ""
        year = item.get("timeline_year")
        stage = item.get("timeline_stage") or ""
        loc = item.get("location_label") or "Indiana"
        ptype = item.get("project_type") or "industrial / commercial project"
        score = item.get("forklift_score")
        mode = item.get("result_mode")

        lines.append(f"{i}. {title} — {loc}")
        meta_bits = [ptype, f"Mode: {mode}"]
        if provider:
            meta_bits.append(provider)
        if score:
            meta_bits.append(f"Relevance {score}/5")
        if stage:
            meta_bits.append(f"{stage}{f' ({year})' if year else ''}")

        lines.append("   " + " • ".join(meta_bits))
        if snippet:
            lines.append(f"   Snippet: {snippet}")
        if url:
            lines.append(f"   URL: {url}")

    return "\n".join(lines)


__all__ = ["search_indiana_developments", "render_developments_markdown"]
