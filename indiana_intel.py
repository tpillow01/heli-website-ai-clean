"""
indiana_intel.py

Indiana developments / lead finder for Tynan / Heli AI.

GOAL:
- Actually return industrial/logistics leads tied to a specific Indiana city/county when possible.
- Prefer:
  1) local announcements / expansions / groundbreakings
  2) planning/zoning agendas, packets, staff reports (often PDFs)
  3) economic development / municipal press releases

Key behaviors:
- Robust geo extraction (case-insensitive, handles trailing years like "2026")
- Multi-query strategy (facility/news + planning packets + local press release bias)
- Scoring instead of overly strict filtering
- Safe rate-limit handling + TTL cache
- If results are empty, returns helpful debug context in render output
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

# Each CSE request returns up to 10 results.
CSE_MAX_PAGES = int(os.environ.get("CSE_MAX_PAGES", "2"))  # default 2 pages = 20 results per query
CSE_MAX_RESULTS_HARD_CAP = 50

CACHE_TTL_SECONDS = int(os.environ.get("CSE_CACHE_TTL_SECONDS", "900"))  # 15 minutes

# ---------------------------------------------------------------------------
# Keywords / signals
# ---------------------------------------------------------------------------

FACILITY_TERMS = [
    "warehouse", "distribution", "distribution center", "distribution facility",
    "logistics", "fulfillment", "fulfilment", "3pl", "third-party logistics",
    "cold storage", "manufacturing", "plant", "factory", "spec building",
    "industrial park", "business park", "truck terminal", "crossdock", "cross-dock",
]

ANNOUNCEMENT_TERMS = [
    "announced", "announcement", "expands", "expansion", "to expand",
    "breaks ground", "breaking ground", "groundbreaking",
    "opens", "opening", "to open", "new facility", "new location",
    "investment", "jobs", "hiring", "development", "under construction",
]

PLANNING_TERMS = [
    "plan commission", "planning commission", "area plan commission",
    "board of zoning appeals", "bza",
    "metropolitan development commission", "mdc",
    "staff report", "public hearing", "rezoning", "rezone", "pud",
    "site plan", "development plan", "primary plat", "secondary plat",
    "ordinance", "variance", "special exception",
    "agenda", "packet", "minutes", "docket", "petition",
]

AGENDA_PLATFORMS = [
    "agendacenter", "documentcenter", "viewfile", "legistar",
    "municode", "granicus", "minutes", "agenda", "packet",
]

NEGATIVE_URL = [
    "facebook.com", "instagram.com", "twitter.com", "x.com", "youtube.com",
    "tripadvisor.com",
]

NEGATIVE_TEXT = [
    "tourism", "visitors bureau", "parks and recreation", "park and recreation",
    "apartment", "apartments", "subdivision", "condo", "housing",
    "school", "elementary", "middle school", "high school", "university", "college",
    "hospital", "medical center", "clinic",
    "church",
]

# Indiana county list (lowercase)
IN_COUNTIES = {
    "adams","allen","bartholomew","benton","blackford","boone","brown","carroll","cass","clark","clay",
    "clinton","crawford","daviess","dearborn","decatur","dekalb","delaware","duboise","elkhart","fayette",
    "floyd","fountain","franklin","fulton","gibson","grant","greene","hamilton","hancock","harrison",
    "hendricks","henry","howard","huntington","jackson","jasper","jay","jennings",
    "johnson","knox","kosciusko","lagrange","lake","laporte","lawrence","madison","marion","marshall",
    "martin","miami","monroe","montgomery","morgan","newton","noble","ohio","orange","owen","parke",
    "perry","pike","porter","posey","pulaski","putnam","randolph","ripley","rush","scott","shelby",
    "spencer","starke","steuben","sullivan","switzerland","tippecanoe","tipton","union","vanderburgh",
    "vermillion","vigo","wabash","warren","warrick","washington","wayne","wells","white","whitley"
}

# ---------------------------------------------------------------------------
# Simple TTL cache (in-process)
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


def _slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9\s-]", "", _lower(s))
    s = re.sub(r"\s+", "-", s).strip("-")
    return s


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
    Google CSE dateRestrict supports dN or mN.
    We'll approximate days>31 to months, capped at 24 months.
    """
    if not days or days <= 0:
        return None
    if days <= 31:
        return f"d{days}"
    months = max(1, int(round(days / 30)))
    if months <= 24:
        return f"m{months}"
    return "m24"


def _extract_geo_hint(q: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (city, county) where county is like "Boone County".
    Handles:
    - "Boone County" / "boone county"
    - "Hendricks 2026"
    - "in Whitestown Indiana" / "Whitestown, IN"
    - "Plainfield (Hendricks County)"
    - Strips trailing years/time phrases
    """
    if not q:
        return (None, None)

    text = q.strip()
    tl = text.lower()

    city: Optional[str] = None
    county: Optional[str] = None

    # 1) Explicit "X County" (case-insensitive)
    m = re.search(r"\b([a-z]+)\s+county\b", tl, flags=re.I)
    if m:
        c = m.group(1).strip().lower()
        if c in IN_COUNTIES:
            county = f"{c.title()} County"

    # 2) Parenthetical city format: "Whitestown (Boone County)"
    m = re.search(r"\b([A-Za-z][A-Za-z\s\.\-']+?)\s*\(\s*([A-Za-z]+)\s+County\s*\)", text, flags=re.I)
    if m:
        city = m.group(1).strip()
        c = m.group(2).strip().lower()
        if c in IN_COUNTIES:
            county = f"{c.title()} County"
        return (city, county)

    # 3) City + state: "Whitestown, IN" or "Whitestown Indiana"
    m = re.search(r"\b([A-Za-z][A-Za-z\s\.\-']{2,})\s*,?\s*(Indiana|IN)\b", text, flags=re.I)
    if m:
        city = m.group(1).strip()
    else:
        # 4) "in/near/around Whitestown" (stop before year/time words)
        m = re.search(r"\b(?:in|near|around)\s+([A-Za-z][A-Za-z\s\.\-']+)", text, flags=re.I)
        if m:
            raw = m.group(1).strip()
            # cut off common tails including years
            raw = re.split(r"\b(20\d{2}|this year|next year|last year|today|now|last|past|recent)\b", raw, flags=re.I)[0].strip()
            raw = re.sub(r"\b(Indiana|IN)\b\.?", "", raw, flags=re.I).strip()
            if raw:
                city = raw

    # 5) Bare county name ("Hendricks 2026", "Boone developments")
    if county is None:
        tokens = re.findall(r"[A-Za-z]+", tl)
        for tok in tokens:
            if tok in IN_COUNTIES:
                county = f"{tok.title()} County"
                break

    return (city, county)


def _is_planning_intent(user_q: str) -> bool:
    t = _lower(user_q)
    if any(w in t for w in PLANNING_TERMS):
        return True
    # construction intent should also use planning tier
    if any(w in t for w in ["under construction", "being built", "site plan", "rezon", "agenda", "packet"]):
        return True
    return False


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


def _google_cse_search(query: str, max_results: int = 20, days: Optional[int] = None) -> List[Dict[str, Any]]:
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
                    dt = _parse_date_from_pagemap(it)
                    out.append({"title": title, "snippet": snippet, "url": url, "provider": provider, "date": dt})
                    if len(out) >= max_results:
                        break
                break

            if resp.status_code in (429, 500, 502, 503, 504):
                _safe_http_error_log(resp, prefix="Google CSE transient error")
                if attempt >= CSE_MAX_RETRIES:
                    if resp.status_code == 429:
                        raise CSEQuotaError("Google CSE rate-limited (429). Reduce queries/pages or increase quota.")
                    break

                retry_after = resp.headers.get("Retry-After")
                wait = None
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


def _extract_metrics(text: str) -> Dict[str, Optional[str]]:
    t = _lower(text)
    sqft = None
    jobs = None
    invest = None

    m = re.search(r"\b(\d{2,4}[,\d]{0,4})\s*(square[-\s]?feet|sq\.?\s*ft|sf)\b", t)
    if m:
        sqft = m.group(1)

    m = re.search(r"\b(\d{2,6})\s+(new\s+)?jobs\b", t)
    if m:
        jobs = m.group(1)

    m = re.search(r"\$\s?(\d+(?:\.\d+)?)\s?(million|billion)\b", t)
    if m:
        invest = f"${m.group(1)} {m.group(2)}"

    return {"sqft": sqft, "jobs": jobs, "investment": invest}


def _is_negative(title: str, snippet: str, url: str) -> bool:
    u = _lower(url)
    txt = _lower(f"{title} {snippet}")
    if any(bad in u for bad in NEGATIVE_URL):
        return True
    if any(neg in txt for neg in NEGATIVE_TEXT):
        return True
    return False


def _geo_score(title: str, snippet: str, city: Optional[str], county: Optional[str]) -> int:
    txt = _lower(f"{title} {snippet}")
    score = 0
    if county:
        base = county.split()[0].lower()
        if base and base in txt:
            score += 2
    if city:
        c = city.lower()
        if c and c in txt:
            score += 2
    return score


def _facility_score(title: str, snippet: str, url: str) -> int:
    txt = _lower(f"{title} {snippet} {url}")
    score = 0

    # facility terms
    for w in FACILITY_TERMS:
        if w in txt:
            score += 3

    # announcement terms
    for w in ANNOUNCEMENT_TERMS:
        if w in txt:
            score += 2

    # PDFs often hide real details behind the doc itself
    if ".pdf" in txt:
        score += 2

    # metrics bonus
    metrics = _extract_metrics(txt)
    if metrics["sqft"]:
        score += 3
    if metrics["jobs"]:
        score += 2
    if metrics["investment"]:
        score += 2

    return score


def _looks_like_planning(title: str, snippet: str, url: str) -> bool:
    txt = _lower(f"{title} {snippet} {url}")
    if any(p in txt for p in AGENDA_PLATFORMS):
        return True
    if any(t in txt for t in PLANNING_TERMS):
        return True
    return False


def _build_queries(user_q: str, city: Optional[str], county: Optional[str]) -> List[Tuple[str, str]]:
    """
    Returns list of (mode, query)
    mode in {"facility", "planning", "press"}
    """
    area_bits: List[str] = []
    if county:
        area_bits.append(f'"{county}"')
        area_bits.append(county.split()[0])  # bare county word helps
    if city:
        area_bits.append(f'"{city}"')

    area = " ".join(area_bits).strip()

    facility_terms = ['"warehouse"', '"distribution center"', '"logistics"', '"manufacturing plant"', '"fulfillment"', '"cold storage"']
    facility_group = "(" + " OR ".join(facility_terms) + ")"

    # Query 1: facilities/news/announcements
    q1 = (
        "Indiana "
        + area
        + " "
        + facility_group
        + ' (announced OR expansion OR groundbreaking OR "now hiring" OR construction OR "site plan")'
    )

    # Query 2: planning packets (agendas, staff reports, PDFs)
    q2 = (
        "Indiana "
        + area
        + ' (agenda OR packet OR "staff report" OR docket OR minutes OR "plan commission" OR BZA OR rezoning OR "site plan")'
        + " (warehouse OR logistics OR distribution OR industrial OR manufacturing)"
        + " (AgendaCenter OR DocumentCenter OR ViewFile OR municode OR granicus OR legistar OR pdf)"
    )

    # Query 3: municipal press release / economic development pages
    q3 = (
        "Indiana "
        + area
        + ' (news OR "press release" OR announcement OR "economic development")'
        + " (warehouse OR distribution OR logistics OR manufacturing OR industrial)"
    )

    return [("facility", q1), ("planning", q2), ("press", q3)]


def _normalize(raw_items: List[Dict[str, Any]], city: Optional[str], county: Optional[str], source_tier: str, mode: str) -> List[Dict[str, Any]]:
    projects: List[Dict[str, Any]] = []
    original_area = county or city or "Indiana"

    for it in raw_items:
        title = it.get("title") or ""
        snippet = it.get("snippet") or ""
        url = it.get("url") or ""
        provider = it.get("provider") or ""
        dt = it.get("date")

        if not url or _is_negative(title, snippet, url):
            continue

        geo = _geo_score(title, snippet, city, county)
        fac = _facility_score(title, snippet, url)
        planning_like = _looks_like_planning(title, snippet, url)

        # If mode is planning, allow planning-like docs with lower facility score.
        # If mode is facility/press, require *some* facility signal.
        if mode == "planning":
            if not planning_like and fac < 4:
                continue
        else:
            if fac < 5 and not planning_like:
                continue

        metrics = _extract_metrics(f"{title} {snippet}")
        forklift_score = max(1, min(5, 1 + (fac + geo) // 6))

        projects.append({
            "project_name": title or "Untitled",
            "company": None,
            "project_type": "planning / zoning filing" if planning_like else "facility/news hit",
            "scope": "local" if geo > 0 else "statewide",
            "location_label": (county or city or "Indiana") if geo > 0 else "Indiana",
            "original_area_label": original_area,
            "forklift_score": forklift_score,
            "forklift_label": "Ranked by industrial relevance",
            "geo_match_score": min(2, geo // 2),
            "match_city": bool(city and city.lower() in _lower(f"{title} {snippet}")),
            "match_county": bool(county and county.split()[0].lower() in _lower(f"{title} {snippet}")),
            "sqft": metrics.get("sqft"),
            "jobs": metrics.get("jobs"),
            "investment": metrics.get("investment"),
            "timeline_stage": "planning filing" if planning_like else "announcement",
            "timeline_year": dt.year if isinstance(dt, datetime) else None,
            "raw_date": dt,
            "url": url,
            "provider": provider,
            "snippet": snippet,
            "source_tier": source_tier,
            "result_mode": mode,
            "industrial_signal_score": fac,
        })

    return projects


def _rank_projects(projects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    now = datetime.utcnow()

    def _sort_key(p: Dict[str, Any]) -> Tuple[int, int, int]:
        geo = int(p.get("geo_match_score") or 0)
        score = int(p.get("industrial_signal_score") or 0)
        dt = p.get("raw_date")
        age_days = 9999
        if isinstance(dt, datetime):
            age_days = (now - dt).days
        return (-geo, -score, age_days)

    projects.sort(key=_sort_key)
    return projects


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_indiana_developments(user_q: str, days: int = 365, max_items: int = 30) -> List[Dict[str, Any]]:
    """
    Main entrypoint.

    NOTE:
    - days=365 is strict. Many "real" municipal announcements won't rank well with that limit.
      If you want better lead volume, consider calling this with days=730 (2 years).
    """
    city, county = _extract_geo_hint(user_q)
    planning_intent = _is_planning_intent(user_q)

    log.info("Geo hint: city=%s county=%s planning_intent=%s", city, county, planning_intent)

    max_items = max(1, min(int(max_items), CSE_MAX_RESULTS_HARD_CAP))

    queries = _build_queries(user_q, city, county)

    # If the user explicitly asked planning-y things, run planning first
    if planning_intent:
        queries = sorted(queries, key=lambda x: 0 if x[0] == "planning" else 1)

    combined_raw: List[Dict[str, Any]] = []
    debug_queries: List[str] = []

    try:
        for mode, q in queries:
            debug_queries.append(f"[{mode}] {q}")
            raw = _google_cse_search(q, max_results=min(max_items, 25), days=days)
            combined_raw.extend(raw)

            # early exit if we already have plenty of raw items
            if len(combined_raw) >= max_items:
                break
    except CSEQuotaError as e:
        log.warning("CSE quota error: %s", e)
        return []

    combined_raw = _dedupe_by_url(combined_raw)

    # Normalize/scoring
    projects: List[Dict[str, Any]] = []
    # Treat results as "local" if we have any geo hint; otherwise "statewide"
    tier = "local" if (city or county) else "statewide"

    # Split by modes to apply slightly different acceptance
    # (We don't store original mode per raw item, so we just run a couple passes)
    projects.extend(_normalize(combined_raw, city, county, source_tier=tier, mode="facility"))
    projects.extend(_normalize(combined_raw, city, county, source_tier=tier, mode="planning"))
    projects.extend(_normalize(combined_raw, city, county, source_tier=tier, mode="press"))

    projects = _dedupe_projects(projects)
    projects = _rank_projects(projects)

    # Attach debug info so render can show it when empty
    for p in projects[:]:
        p["_debug_city"] = city
        p["_debug_county"] = county
        p["_debug_days"] = days

    # If empty, store one “debug envelope” item so render can show what it tried
    if not projects:
        projects = [{
            "_debug_only": True,
            "_debug_city": city,
            "_debug_county": county,
            "_debug_days": days,
            "_debug_queries": debug_queries[:],
        }]

    return projects


def _dedupe_projects(projects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for p in projects:
        url = (p.get("url") or "").strip()
        if not url:
            # allow debug-only item
            if p.get("_debug_only"):
                out.append(p)
            continue
        key = url.split("#")[0]
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def render_developments_markdown(items: List[Dict[str, Any]]) -> str:
    # Debug-only envelope
    if items and items[0].get("_debug_only"):
        city = items[0].get("_debug_city")
        county = items[0].get("_debug_county")
        days = items[0].get("_debug_days")
        qs = items[0].get("_debug_queries") or []

        lines = []
        lines.append("No high-confidence industrial/logistics hits after filtering.")
        lines.append("")
        lines.append(f"Detected geo → city={city} county={county} | timeframe={days} days")
        lines.append("")
        lines.append("Search queries used:")
        for q in qs[:6]:
            lines.append(f"- {q}")
        lines.append("")
        lines.append("Tip: If this area is quiet, try 730 days (2 years) or ask for planning packets/agenda packets specifically.")
        return "\n".join(lines)

    if not items:
        return (
            "No results returned (empty). If you saw a 429 in logs, you're rate-limited by Google CSE.\n"
            "Fix: lower CSE_MAX_PAGES, wait a bit, and/or increase Google API quota/billing."
        )

    lines: List[str] = []
    lines.append("Industrial / logistics results (ranked web hits):")

    for i, item in enumerate(items[:15], start=1):
        title = item.get("project_name") or "Untitled"
        snippet = (item.get("snippet") or "").strip()
        url = item.get("url") or ""
        provider = item.get("provider") or ""
        loc = item.get("location_label") or item.get("original_area_label") or "Indiana"
        score = item.get("forklift_score")
        geo_score = item.get("geo_match_score") or 0
        mode = item.get("result_mode") or "unknown"
        sqft = item.get("sqft")
        jobs = item.get("jobs")
        inv = item.get("investment")

        lines.append(f"{i}. {title} — {loc}")
        meta = [f"Mode: {mode}"]
        if provider:
            meta.append(provider)
        if score:
            meta.append(f"Relevance {score}/5")
        if geo_score:
            meta.append(f"Geo match {geo_score}/2")
        if sqft:
            meta.append(f"{sqft} sf")
        if jobs:
            meta.append(f"{jobs} jobs")
        if inv:
            meta.append(inv)

        lines.append("   " + " • ".join(meta))
        if snippet:
            lines.append(f"   Snippet: {snippet}")
        if url:
            lines.append(f"   URL: {url}")

    return "\n".join(lines)


__all__ = ["search_indiana_developments", "render_developments_markdown"]
