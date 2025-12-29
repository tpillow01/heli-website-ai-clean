"""
indiana_intel.py

Indiana developments / lead finder for Tynan / Heli AI.

Fixes included (Dec 2025):
- Works for ALL 92 Indiana counties.
- If user specifies a CITY (e.g., Whitestown) with no county:
  - Adds city-government domain candidates (e.g., whitestown.in.gov) into local-gov query.
  - Optionally infers the county from the city via a quick CSE lookup (best-effort).
- Geo-lock no longer depends only on title/snippet:
  - Counts host/domain match as geo match (e.g., whitestown.in.gov implies Whitestown).
- Local-government "agenda/minutes/petition" filter is more permissive:
  - Accepts DocumentCenter/AgendaCenter/ViewFile/center.egov links and PDFs even if snippet is thin.
- HARD RULE:
  - If user asked for a specific city or county, do NOT return statewide filler.
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
# Indiana counties (92)
# ---------------------------------------------------------------------------

INDIANA_COUNTIES: List[str] = [
    "Adams","Allen","Bartholomew","Benton","Blackford","Boone","Brown","Carroll","Cass","Clark",
    "Clay","Clinton","Crawford","Daviess","Dearborn","Decatur","DeKalb","Delaware","Dubois",
    "Elkhart","Fayette","Floyd","Fountain","Franklin","Fulton","Gibson","Grant","Greene",
    "Hamilton","Hancock","Harrison","Hendricks","Henry","Howard","Huntington","Jackson","Jasper",
    "Jay","Jefferson","Jennings","Johnson","Knox","Kosciusko","LaGrange","Lake","LaPorte",
    "Lawrence","Madison","Marion","Marshall","Martin","Miami","Monroe","Montgomery","Morgan",
    "Newton","Noble","Ohio","Orange","Owen","Parke","Perry","Pike","Porter","Posey","Pulaski",
    "Putnam","Randolph","Ripley","Rush","St. Joseph","Scott","Shelby","Spencer","Starke",
    "Steuben","Sullivan","Switzerland","Tippecanoe","Tipton","Union","Vanderburgh","Vermillion",
    "Vigo","Wabash","Warren","Warrick","Washington","Wayne","Wells","White","Whitley",
]

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", (s or "").lower())).strip()

_COUNTY_NORM_TO_CANON: Dict[str, str] = {}
for c in INDIANA_COUNTIES:
    _COUNTY_NORM_TO_CANON[_norm(c)] = c

_COUNTY_ALIASES: Dict[str, str] = {
    "st joseph": "St. Joseph",
    "saint joseph": "St. Joseph",
    "laporte": "LaPorte",
    "la porte": "LaPorte",
    "lagrange": "LaGrange",
    "la grange": "LaGrange",
    "dekalb": "DeKalb",
    "de kalb": "DeKalb",
}
for k, v in _COUNTY_ALIASES.items():
    _COUNTY_NORM_TO_CANON[_norm(k)] = v

# ---------------------------------------------------------------------------
# Keywords
# ---------------------------------------------------------------------------

BASE_KEYWORDS = (
    '(warehouse OR "distribution center" OR "distribution facility" OR '
    '"distribution hub" OR logistics OR "logistics center" OR '
    '"logistics facility" OR "logistics hub" OR "fulfillment center" OR '
    '"industrial park" OR "business park" OR "industrial complex" OR '
    '"manufacturing plant" OR "manufacturing facility" OR plant OR '
    '"production plant" OR "assembly plant" OR "cold storage" OR facility)'
)

LOCAL_GOV_KEYWORDS = (
    '("plan commission" OR "area plan commission" OR '
    '"board of zoning appeals" OR BZA OR agenda OR minutes OR docket OR '
    "petition OR rezoning OR rezone OR PUD OR "
    '"development plan" OR "site plan" OR '
    '"primary plat" OR "secondary plat" OR '
    '"staff report" OR "public hearing" OR ordinance OR variance OR '
    '"special exception" OR "zoning map" OR "rezon*" OR '
    '"use variance" OR "development standards")'
)

INDUSTRIAL_SOFT = (
    '(industrial OR warehouse OR "distribution" OR logistics OR manufacturing OR '
    '"cold storage" OR "spec building" OR "truck terminal" OR "industrial park")'
)

_LOCAL_GOV_URL_HINTS = [
    "agendacenter", "documentcenter", "viewfile", "legistar", "center.egov",
    "/documents/", "/wp-content/uploads/", "/minutes", "/agenda",
]

# ---------------------------------------------------------------------------
# Filters / heuristics
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "what","are","there","any","new","or","in","the","last","month","months","recent","recently",
    "project","projects","have","has","been","announced","announcement","for","about","on","of",
    "a","an","county","indiana","logistics","warehouse","warehouses","distribution","center",
    "centers","companies","coming","to","area","city","kind","sort","type","planned","plan",
    "announce","expanded","expansion","hiring","jobs","agenda","minutes","petition","rezoning",
    "rezone","pud","site","development","permit","permits","days","weeks","years","pdf"
}

_FORKLIFT_POSITIVE = [
    "warehouse","distribution center","distribution facility","distribution hub",
    "fulfillment center","fulfillment facility",
    "logistics center","logistics facility","logistics hub","logistics park",
    "industrial park","business park","industrial complex",
    "manufacturing plant","manufacturing facility",
    "production plant","assembly plant",
    "cold storage","3pl","third-party logistics","third party logistics",
]

_PROJECT_NEGATIVE_URL = [
    "facebook.com","instagram.com","twitter.com","x.com","youtube.com","tripadvisor.com"
]

# Keep this list conservative. Over-filtering is one reason you saw empty results.
_PROJECT_NEGATIVE_TEXT = [
    "visit ","tourism","visitors bureau",
    "apartments","housing development","subdivision","condominiums",
    "senior living","assisted living",
    "elementary school","middle school","high school","university","college",
]

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _lower(s: Any) -> str:
    return str(s or "").lower()

def _canonicalize_url(url: str) -> str:
    try:
        u = urlparse(url)
        qs = [
            (k, v)
            for (k, v) in parse_qsl(u.query, keep_blank_values=True)
            if k.lower() not in {"utm_source","utm_medium","utm_campaign","utm_term","utm_content","gclid","fbclid"}
        ]
        new_query = urlencode(qs, doseq=True)
        clean = urlunparse((u.scheme, u.netloc, u.path, u.params, new_query, ""))
        return clean.rstrip("/")
    except Exception:
        return (url or "").rstrip("/")

def _days_to_date_restrict(days: Optional[int]) -> Optional[str]:
    """
    Google CSE dateRestrict supports d, w, m, y.
    """
    if not days or days <= 0:
        return None
    if days <= 31:
        return f"d{days}"
    if days <= 365:
        weeks = max(1, int(round(days / 7)))
        return f"w{weeks}"
    years = max(1, int(round(days / 365)))
    return f"y{years}"

def _parse_date_from_pagemap(it: Dict[str, Any]) -> Optional[datetime]:
    pagemap = it.get("pagemap") or {}
    meta_list = pagemap.get("metatags") or []
    if not isinstance(meta_list, list):
        return None

    keys = (
        "article:published_time","article:modified_time","og:published_time","og:updated_time",
        "date","dc.date","dc.date.issued","pubdate","publishdate","datepublished",
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

def _geo_lock_required(city: Optional[str], county: Optional[str]) -> bool:
    return bool(city or county)

# ---------------------------------------------------------------------------
# County/city extraction
# ---------------------------------------------------------------------------

def _extract_county_from_text(q: str) -> Optional[str]:
    t = _norm(q)

    # explicit "... county"
    m = re.search(r"\b([a-z]+\s*[a-z]*)\s+county\b", t)
    if m:
        cand = _norm(m.group(1))
        canon = _COUNTY_NORM_TO_CANON.get(cand)
        if canon:
            return f"{canon} County"

    # bare county name anywhere
    keys = sorted(_COUNTY_NORM_TO_CANON.keys(), key=len, reverse=True)
    for k in keys:
        if not k:
            continue
        if re.search(rf"\b{re.escape(k)}\b", t):
            canon = _COUNTY_NORM_TO_CANON[k]
            return f"{canon} County"

    return None

def _extract_city_from_text(q: str) -> Optional[str]:
    if not q:
        return None

    # Leading "Whitestown IN ..."
    m0 = re.match(r"^\s*([A-Za-z][A-Za-z\s]{1,40}?)(?:,?\s*)(IN|Indiana)\b", q.strip(), flags=re.I)
    if m0:
        city = m0.group(1).strip()
        if "county" not in city.lower():
            return " ".join(city.split()[:3])

    # "... in City, IN"
    m1 = re.search(
        r"\b(?:in|around|near)\s+([A-Za-z][A-Za-z\s]{1,40}?)(?:,?\s*)(IN|Indiana)\b",
        q,
        flags=re.I,
    )
    if m1:
        city = m1.group(1).strip()
        if "county" not in city.lower():
            return " ".join(city.split()[:3])

    # "City, IN" anywhere
    m2 = re.search(r"\b([A-Za-z][A-Za-z\s]{1,40}?)(?:,?\s*)(IN|Indiana)\b", q, flags=re.I)
    if m2:
        city = m2.group(1).strip()
        if len(city.split()) <= 3 and "county" not in city.lower():
            return " ".join(city.split()[:3])

    # "... in City" (weak)
    m3 = re.search(r"\b(?:in|around|near)\s+([A-Za-z][A-Za-z\s]{1,30}?)(?:[,\?\.\!]|$)", q, flags=re.I)
    if m3:
        city = m3.group(1).strip()
        if "county" in city.lower():
            return None
        if re.search(r"\b(last|past|days|months|weeks|years|since|recent)\b", city, flags=re.I):
            return None
        return " ".join(city.split()[:3])

    return None

def _county_slug(county: str) -> str:
    base = county.replace("County", "").strip()
    s = _norm(base).replace(" ", "")
    return s

def _candidate_county_gov_sites(county: Optional[str]) -> List[str]:
    if not county:
        return []
    slug = _county_slug(county)

    overrides: Dict[str, List[str]] = {
        "marion": ["indy.gov", "indianapolis.in.gov"],
        "stjoseph": ["sjc.in.gov"],
    }

    sites: List[str] = []
    if slug in overrides:
        sites.extend(overrides[slug])

    sites.extend([
        f"co.{slug}.in.us",
        f"{slug}county.in.gov",
        f"{slug}.in.gov",
        f"{slug}county.in.us",
        f"{slug}.in.us",
    ])

    out: List[str] = []
    seen = set()
    for s in sites:
        s = s.strip().lower()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out

def _city_slug(city: str) -> str:
    return _norm(city).replace(" ", "")

def _candidate_city_sites(city: Optional[str]) -> List[str]:
    if not city:
        return []
    slug = _city_slug(city)
    # Indiana municipalities commonly use cityname.in.gov
    candidates = [
        f"{slug}.in.gov",
        f"{slug}.in.us",
        f"townof{slug}.org",
        f"townof{slug}.com",
        f"cityof{slug}.org",
        f"cityof{slug}.com",
    ]
    out: List[str] = []
    seen = set()
    for s in candidates:
        s = s.strip().lower()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out

def _infer_county_from_city_via_cse(city: str) -> Optional[str]:
    """
    Best-effort: use Google CSE to infer county from city when county not provided.
    Example query: '"Whitestown" Indiana county'
    Then scan title/snippet for any Indiana county name.
    """
    if not city or not GOOGLE_CSE_KEY or not GOOGLE_CSE_CX:
        return None

    q = f"\"{city}\" Indiana county"
    items = _google_cse_search(q, max_results=10, days=3650, file_type=None, _internal_call=True)

    joined = " ".join([_lower(it.get("title", "")) + " " + _lower(it.get("snippet", "")) for it in items])
    # look for "<County> County" or bare county name
    for canon in INDIANA_COUNTIES:
        c_norm = _norm(canon)
        if re.search(rf"\b{re.escape(c_norm)}\s+county\b", _norm(joined)):
            return f"{canon} County"
        if re.search(rf"\b{re.escape(c_norm)}\b", _norm(joined)):
            # Only accept bare county name if "county" is mentioned somewhere nearby
            if " county" in _norm(joined):
                return f"{canon} County"

    return None

def _extract_geo_hint(q: str) -> Tuple[Optional[str], Optional[str]]:
    city = _extract_city_from_text(q)
    county = _extract_county_from_text(q)

    # If city exists and county missing, try to infer county (best-effort)
    if city and not county:
        inferred = _infer_county_from_city_via_cse(city)
        if inferred:
            county = inferred

    log.info("Geo hint: city=%s county=%s", city, county)
    return (city, county)

# ---------------------------------------------------------------------------
# Google CSE
# ---------------------------------------------------------------------------

def _google_cse_search(
    query: str,
    max_results: int = 30,
    days: Optional[int] = None,
    file_type: Optional[str] = None,
    _internal_call: bool = False,
) -> List[Dict[str, Any]]:
    if not GOOGLE_CSE_KEY or not GOOGLE_CSE_CX:
        log.warning(
            "GOOGLE_CSE_KEY or GOOGLE_CSE_CX not set; returning empty list. "
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
    seen_urls: set[str] = set()
    start = 1

    while len(out) < max_results and start <= 91:
        params = dict(base_params)
        params["start"] = start

        try:
            resp = requests.get(
                GOOGLE_CSE_ENDPOINT, params=params, headers=REQUEST_HEADERS, timeout=12
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.warning("Google CSE request failed (start=%s): %s", start, e)
            break

        items = data.get("items", []) or []
        if not _internal_call:
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

            # enforce cutoff if we have a date
            if cutoff and isinstance(dt, datetime) and dt < cutoff:
                continue

            out.append(
                {"title": title, "snippet": snippet, "url": canon, "provider": provider, "date": dt}
            )
            if len(out) >= max_results:
                break

        start += 10

    return out

# ---------------------------------------------------------------------------
# Query builders
# ---------------------------------------------------------------------------

def _build_query(user_q: str, city: Optional[str], county: Optional[str]) -> str:
    parts: List[str] = ["Indiana", BASE_KEYWORDS]

    if county:
        parts.append(f"\"{county}\"")
        parts.append(f"\"{county.replace(' County','')}\"")

    if city:
        parts.append(f"\"{city}\"")
        parts.append(f"\"{city}, IN\"")

    cleaned = re.sub(r"[“”\"']", " ", user_q or "")
    tokens = re.findall(r"[A-Za-z0-9]+", cleaned)
    extra_tokens: List[str] = []
    for tok in tokens:
        if tok.lower() in _STOPWORDS:
            continue
        extra_tokens.append(tok)
    if extra_tokens:
        parts.append(" ".join(extra_tokens[:8]))

    q = " ".join(parts)
    log.info("Google CSE query: %s", q)
    return q

def _build_local_gov_query(county: Optional[str], city: Optional[str]) -> str:
    county_sites = _candidate_county_gov_sites(county)
    city_sites = _candidate_city_sites(city)

    site_terms: List[str] = []
    for s in county_sites + city_sites:
        site_terms.append(f"site:{s}")

    # Broad Indiana government fallback
    site_terms.extend([
        "site:*.in.gov",
        "site:*.in.us",
        "site:in.gov",
        "site:in.us",
    ])

    site_block = "(" + " OR ".join(site_terms) + ")"

    geo_bits: List[str] = []
    if county:
        geo_bits.append(f"\"{county}\"")
        geo_bits.append(f"\"{county.replace(' County','')}\"")
    if city:
        geo_bits.append(f"\"{city}\"")
        geo_bits.append(f"\"{city}, IN\"")

    geo_clause = "(" + " OR ".join(geo_bits) + ")" if geo_bits else ""

    civicplus_hint = '("AgendaCenter" OR "ViewFile" OR "DocumentCenter" OR "minutes" OR "agenda" OR "packet")'

    q = f"{site_block} {LOCAL_GOV_KEYWORDS} {geo_clause} {INDUSTRIAL_SOFT} {civicplus_hint}".strip()
    log.info("Google CSE local-gov query: %s", q)
    return q

# ---------------------------------------------------------------------------
# Classification / scoring
# ---------------------------------------------------------------------------

def _looks_like_hit(title: str, snippet: str, url: str, allow_local_gov: bool) -> bool:
    text = _lower(f"{title} {snippet}")
    url_l = _lower(url or "")

    for bad in _PROJECT_NEGATIVE_URL:
        if bad in url_l:
            return False

    for neg in _PROJECT_NEGATIVE_TEXT:
        if neg in text:
            return False

    if allow_local_gov:
        # Accept if it looks like an agenda/minutes/petition OR the URL strongly suggests it
        if any(h in url_l for h in _LOCAL_GOV_URL_HINTS):
            return True
        if re.search(r"\.(pdf|doc|docx)\b", url_l):
            return True
        # Otherwise require at least one civic keyword
        return any(k in text for k in ("plan commission","area plan commission","board of zoning appeals","bza","agenda","minutes","petition","rezon","pud","site plan","development plan","staff report"))

    # non-local-gov: require industrial facility keyword
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

def _host_contains(host: str, needle: str) -> bool:
    if not host or not needle:
        return False
    h = _norm(host).replace(" ", "")
    n = _norm(needle).replace(" ", "")
    return bool(n) and (n in h)

def _geo_match_scores(title: str, snippet: str, url: str, city: Optional[str], county: Optional[str]) -> Tuple[int, bool, bool]:
    """
    Geo match now considers:
    - title/snippet mentions
    - host/domain contains city or county slug (critical for agendas/minutes PDFs)
    """
    text = _lower(title + " " + snippet)
    host = ""
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        host = ""

    match_city = False
    match_county = False

    if county:
        base = county.replace(" County", "").strip()
        base_l = base.lower()
        if base_l and re.search(rf"\b{re.escape(base_l)}\b", text):
            match_county = True
        if county.lower() in text:
            match_county = True

        # host-based county match
        if _host_contains(host, base) or _host_contains(host, _county_slug(county)):
            match_county = True

    if city:
        c = city.strip()
        c_l = c.lower()
        if c_l and re.search(rf"\b{re.escape(c_l)}\b", text):
            match_city = True

        # host-based city match
        if _host_contains(host, c) or _host_contains(host, _city_slug(c)):
            match_city = True

    if match_city and match_county:
        return 2, True, True
    if match_city or match_county:
        return 1, match_city, match_county
    return 0, False, False

def _infer_project_type(title: str, snippet: str, allow_local_gov: bool) -> str:
    if allow_local_gov:
        return "planning / zoning filing"
    text = _lower(title + " " + snippet)
    if any(w in text for w in ("warehouse", "distribution center", "distribution facility", "fulfillment center")):
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
    allow_local_gov = (source_tier == "local_gov")
    original_area_label = county or city or "Indiana"
    locked = _geo_lock_required(city, county)

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
        geo_score, match_city, match_county = _geo_match_scores(title, snippet, url, city, county)
        project_type = _infer_project_type(title, snippet, allow_local_gov=allow_local_gov)

        # HARD FILTER: if user asked for a specific city/county, require geo match (now includes host-based matches)
        if locked and not (match_city or match_county):
            continue

        timeline_year: Optional[int] = dt.year if isinstance(dt, datetime) else None
        timeline_stage = "agenda/minutes" if allow_local_gov else ("announcement" if timeline_year else "not specified in snippet")

        projects.append(
            {
                "project_name": title or "Untitled project",
                "company": None,
                "project_type": project_type,
                "scope": "local" if geo_score > 0 else "statewide",
                "location_label": county or city or "Indiana",
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
        return (-geo, -score, age_days)

    projects.sort(key=_sort_key)
    return projects

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_indiana_developments(user_q: str, days: int = 365, max_items: int = 30) -> List[Dict[str, Any]]:
    """
    Tiers:
      1) local industrial/news
      2) local_gov agendas/minutes/petitions (PDF-first)
      3) statewide ONLY if user did NOT request a specific city/county

    HARD RULE:
      If user asked for a specific city or county, do NOT return statewide substitutes.
    """
    city, county = _extract_geo_hint(user_q)
    locked = _geo_lock_required(city, county)

    projects: List[Dict[str, Any]] = []

    # Tier 1: local industrial/news
    query_local = _build_query(user_q, city, county)
    raw_local = _google_cse_search(query_local, max_results=max_items, days=days, file_type=None)
    if raw_local:
        projects.extend(_normalize_projects(raw_local, city, county, user_q, source_tier="local"))

    # Tier 2: local government docs (PDF-first)
    q_gov = _build_local_gov_query(county=county, city=city)
    raw_gov_pdf = _google_cse_search(q_gov, max_results=max_items, days=days, file_type="pdf")
    if raw_gov_pdf:
        projects.extend(_normalize_projects(raw_gov_pdf, city, county, user_q, source_tier="local_gov"))

    # If still nothing, try HTML gov pages
    if not projects:
        raw_gov = _google_cse_search(q_gov, max_results=max_items, days=days, file_type=None)
        if raw_gov:
            projects.extend(_normalize_projects(raw_gov, city, county, user_q, source_tier="local_gov"))

    # If user specified location, return only location-locked results (could be empty)
    if locked:
        return _rank_projects(projects)[:max_items]

    # Tier 3: statewide (only if not locked)
    if not projects:
        query_statewide = _build_query(user_q, city=None, county=None)
        raw_statewide = _google_cse_search(query_statewide, max_results=max_items, days=days, file_type=None)
        if raw_statewide:
            projects = _normalize_projects(raw_statewide, None, None, user_q, source_tier="statewide")

    return _rank_projects(projects)[:max_items]

def render_developments_markdown(items: List[Dict[str, Any]]) -> str:
    if not items:
        return (
            "No location-specific results were found for that timeframe.\n"
            "Try:\n"
            "- Increase days to 730\n"
            "- Use: agenda packet / staff report / docket / petition\n"
            "- Provide county name if you only gave a city\n"
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
