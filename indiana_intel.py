"""
indiana_intel.py

Indiana developments lead-finder for Tynan / Heli AI site.

Responsibilities:
- Take a natural-language question (e.g. "new warehouses in Boone County in the last 90 days")
- Hit Google Custom Search JSON API (Programmable Search Engine) for
  Indiana industrial / logistics / manufacturing / commercial projects
- Normalize results into simple dicts that the chat layer can format
- Provide a plain-text summary (mostly for debugging or backup use)

Environment variables required:
- GOOGLE_CSE_KEY : Google API key for Custom Search JSON API
- GOOGLE_CSE_CX  : Programmable Search Engine ID (cx) configured to search the web
"""

from __future__ import annotations

import os
import re
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

log = logging.getLogger("indiana_intel")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s"
    )

# ---------------------------------------------------------------------------
# Config: Google CSE
# ---------------------------------------------------------------------------
GOOGLE_CSE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"
GOOGLE_CSE_KEY = os.environ.get("GOOGLE_CSE_KEY")
GOOGLE_CSE_CX = os.environ.get("GOOGLE_CSE_CX")

# Base industrial / logistics / commercial keywords for Indiana
BASE_KEYWORDS = (
    'Indiana (warehouse OR "distribution center" OR logistics OR manufacturing '
    'OR plant OR factory OR industrial OR fulfillment OR "business park" '
    'OR "industrial park" OR "logistics park" OR headquarters OR facility)'
)

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _lower(s: Any) -> str:
    return str(s or "").lower()


def _extract_geo_hint(q: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Very light-weight extractor:
    - Finds 'X County' → county='X County'
    - Tries to grab a city name after 'in ' if it looks like 'Plainfield' or 'Plainfield, IN'
    Returns (city, county).
    """
    if not q:
        return (None, None)
    text = q.strip()

    # County: e.g. "in Boone County", "Boone County leads", etc.
    m_county = re.search(r"\b([A-Za-z]+)\s+County\b", text)
    county = None
    if m_county:
        county = f"{m_county.group(1).strip()} County"

    # City: look for "in Plainfield", "around Plainfield, IN", etc.
    m_city = re.search(r"\b(?:in|around|near)\s+([A-Za-z\s]+?)(?:,|\?|\.|$)", text)
    city = None
    if m_city:
        raw = m_city.group(1).strip()
        raw = re.sub(r"\b(Indiana|IN)\b\.?", "", raw, flags=re.I).strip()
        if raw:
            city = raw

    return (city, county)


def _extract_year_from_question(q: str) -> Optional[int]:
    """
    Try to find a 4-digit year like 2024, 2025, 2026 in the user's question.
    Used only as a soft hint for timeline interpretation / filtering.
    """
    if not q:
        return None
    for m in re.finditer(r"\b(20[2-4][0-9])\b", q):
        try:
            year = int(m.group(1))
            if 2000 <= year <= 2100:
                return year
        except Exception:
            continue
    return None


_STOPWORDS = {
    "what", "are", "there", "any", "new", "or", "in", "the", "last", "month",
    "months", "recent", "recently", "project", "projects", "have", "has",
    "been", "announced", "announcement", "for", "about", "on", "of", "a",
    "an", "county", "indiana", "logistics", "warehouse", "distribution",
    "center", "centers", "companies", "coming", "to", "area", "city",
    "kind", "sort", "type", "planned", "plan", "announce", "announced",
}


def _build_query(user_q: str, city: Optional[str], county: Optional[str]) -> str:
    """
    Build a Google CSE query anchored on Indiana industrial/commercial keywords,
    with optional city / county bias, plus a small keyword tail from the question.
    (No domain restriction – we want as many relevant hits as possible.)
    """
    parts: List[str] = [BASE_KEYWORDS]

    if county:
        parts.append(f'"{county}"')
    if city:
        parts.append(f'"{city}"')

    # Clean the user question and add a few non-boring keywords as a soft hint
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
                # Try ISO first
                try:
                    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                except Exception:
                    dt = None
                    # Fallback short formats
                    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
                        try:
                            dt = datetime.strptime(raw[:10], fmt)
                            break
                        except Exception:
                            dt = None
                    if not dt:
                        continue

                # Normalize to UTC naive
                if dt.tzinfo is not None:
                    dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
                return dt

    return None


def _google_cse_search(query: str) -> List[Dict[str, Any]]:
    """
    Thin wrapper around Google Custom Search JSON API.
    Returns a list of normalized raw results:
      { title, snippet, url, provider, date (datetime or None) }
    """
    if not GOOGLE_CSE_KEY or not GOOGLE_CSE_CX:
        log.warning(
            "GOOGLE_CSE_KEY or GOOGLE_CSE_CX not set or empty; returning empty result list. "
            f"GOOGLE_CSE_KEY present={bool(GOOGLE_CSE_KEY)}, GOOGLE_CSE_CX present={bool(GOOGLE_CSE_CX)}"
        )
        return []

    params = {
        "key": GOOGLE_CSE_KEY,
        "cx": GOOGLE_CSE_CX,
        "q": query,
        "num": 10,
    }

    try:
        resp = requests.get(GOOGLE_CSE_ENDPOINT, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.warning("Google CSE request failed: %s", e)
        return []

    items = data.get("items", []) or []
    log.info("Google CSE returned %s items", len(items))

    out: List[Dict[str, Any]] = []
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
                "date": dt,  # datetime or None
            }
        )

    return out

# ---------------------------------------------------------------------------
# HTML fetch + fact extraction
# ---------------------------------------------------------------------------

def _fetch_html(url: str) -> str:
    """
    Fetch page HTML for deeper facts. We keep this lightweight and tolerant of failures.
    """
    if not url:
        return ""
    try:
        resp = requests.get(
            url,
            timeout=8,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; TynanIntelBot/1.0)"
            },
        )
        resp.raise_for_status()
    except Exception as e:
        log.info("Failed to fetch page HTML: %s (%s)", url, e)
        return ""

    ctype = resp.headers.get("Content-Type", "").lower()
    if "text/html" not in ctype:
        return ""
    return resp.text or ""


def _find_first_group(patterns, text: str) -> Optional[str]:
    for pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            return m.group(1).strip()
    return None


def _guess_company_from_title_snippet(title: str, snippet: str) -> Optional[str]:
    """
    Very rough heuristic: look at patterns like:
      'ACME Corp to build new distribution center in Plainfield, Indiana'
      'Amazon announces new fulfillment center in ...'
    We NEVER pretend we know the company if we don't see a clear pattern.
    """
    candidates = [title, snippet]
    for text in candidates:
        if not text:
            continue
        t = text.replace("–", "-")
        for kw in (" to build", " to locate", " to open", " announces", " plans ", " expands", " investing", " breaks ground"):
            idx = t.lower().find(kw.strip())
            if idx > 3:
                name = t[:idx].strip(" -–:|")
                if name and not name.lower().startswith(("city of", "town of", "county of")):
                    return name
    return None


def _extract_project_facts_from_html(html: str) -> Dict[str, Optional[str]]:
    """
    Try to pull out sq ft, jobs, investment, and (if needed) a year from HTML.
    We deliberately keep everything as strings – no numeric guessing.
    """
    if not html:
        return {
            "sqft": None,
            "jobs": None,
            "investment": None,
            "year": None,
        }

    # Lowercase for pattern matching
    text = _lower(html)

    # Sq ft: 100,000 square feet / sq ft / SF
    sqft_match = re.search(r'([\d,]{4,})\s*(square\s*feet|sq\.?\s*ft|sf\b)', text)
    sqft = f"{sqft_match.group(1)} sq ft" if sqft_match else None

    # Jobs: 200 jobs, 300 new jobs
    jobs_match = re.search(r'(\d{2,4})\s+(?:new\s+)?jobs\b', text)
    jobs = f"{jobs_match.group(1)} jobs" if jobs_match else None

    # Investment: $100 million, $250M, etc.
    inv_match = re.search(r'\$\s*([\d,.]+)\s*(million|billion|m|bn)?', text)
    if inv_match:
        amt = inv_match.group(1)
        unit = inv_match.group(2) or ""
        unit = unit.strip().lower()
        if unit in {"m", "million"}:
            unit = "million"
        elif unit in {"bn", "billion"}:
            unit = "billion"
        else:
            unit = ""
        investment = f"${amt} {unit}".strip()
    else:
        investment = None

    # Year hint if we didn't get one from metadata
    year_match = re.search(r'\b(20[2-4][0-9])\b', text)
    year = year_match.group(1) if year_match else None

    return {
        "sqft": sqft,
        "jobs": jobs,
        "investment": investment,
        "year": year,
    }

# ---------------------------------------------------------------------------
# Classification & filtering
# ---------------------------------------------------------------------------

def _classify_scope_and_location(
    title: str,
    snippet: str,
    city: Optional[str],
    county: Optional[str],
) -> Tuple[str, str]:
    """
    Very light classification into:
      - scope: "local" (explicitly mentions city/county) or "statewide"
      - location_label: human-readable label for display.
    """
    text = _lower(title) + " " + _lower(snippet)
    loc_label = "Indiana"

    scope = "statewide"
    if county:
        base = county.split()[0].lower()
        if base and base in text:
            scope = "local"
        loc_label = county
    if city:
        c = city.lower()
        if c and c in text:
            scope = "local"
        if not county:
            loc_label = city

    return scope, loc_label


def _infer_project_type(title: str, snippet: str) -> str:
    """
    Heuristic to categorize project type from title/snippet.
    We keep this conservative – no wild guessing.
    """
    text = _lower(title + " " + snippet)

    if any(w in text for w in ("warehouse", "fulfillment center", "distribution center", "distribution hub")):
        return "warehouse / logistics facility"
    if any(w in text for w in ("logistics hub", "logistics park")):
        return "warehouse / logistics facility"
    if any(w in text for w in ("manufacturing plant", "factory", "plant expansion", "production plant")):
        return "manufacturing plant"
    if any(w in text for w in ("business park", "industrial park")):
        return "business / industrial park"
    if any(w in text for w in ("headquarters", "hq", "office building", "showroom")):
        return "HQ / office project"

    return "Industrial / commercial project"


# Positive phrases that tend to indicate a real project/development
_PROJECT_POSITIVE = [
    "project",
    "development",
    "development project",
    "distribution center",
    "fulfillment center",
    "logistics center",
    "logistics hub",
    "logistics park",
    "warehouse",
    "manufacturing plant",
    "production plant",
    "factory",
    "industrial park",
    "business park",
    "to locate in",
    "plans to build",
    "will build",
    "to build a",
    "broke ground",
    "groundbreaking",
    "expansion",
    "new facility",
    "facility opening",
    "complex",
    "campus",
    "headquarters",
    "hq",
]

# Negative signals: generic homepages, FAQs, tourism, Facebook chatter, incidents, etc.
_PROJECT_NEGATIVE = [
    "official website",
    "faq",
    "faqs",
    "civicengage",
    "visit plainfield",
    "visit hendricks county",
    "events, shopping & family fun",
    "welcome to plainfield",
    "quarterly welcome",
    "police shooting",
    "shooting incident",
    "facebook.com",
    "plainfield chatter",
    "news flash",
    "city of plainfield",
]


def _looks_like_project_hit(title: str, snippet: str, url: str) -> bool:
    """
    Decide whether this CSE hit looks like a concrete project / facility
    vs. generic marketing, tourism, FAQ, or random news.
    """
    text = _lower(f"{title} {snippet}")
    url_l = _lower(url or "")

    # Hard filters first
    if "facebook.com" in url_l:
        return False
    if any(neg in text for neg in _PROJECT_NEGATIVE):
        return False

    # Require at least one positive signal
    if any(pos in text for pos in _PROJECT_POSITIVE):
        return True

    return False

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_projects(
    raw_items: List[Dict[str, Any]],
    city: Optional[str],
    county: Optional[str],
    user_q: str,
) -> List[Dict[str, Any]]:
    """
    Convert raw CSE results into the structured dicts used by the chat layer.
    We do NOT invent square footage, job counts, or dollars – those stay None
    unless we explicitly parse them.
    """
    inferred_year = _extract_year_from_question(user_q)
    original_area_label = county or city or "Indiana"

    projects: List[Dict[str, Any]] = []
    for it in raw_items:
        title = it.get("title") or ""
        snippet = it.get("snippet") or ""
        url = it.get("url") or ""
        provider = it.get("provider") or ""
        dt = it.get("date")  # datetime or None

        # Filter out obvious non-project hits
        if not _looks_like_project_hit(title, snippet, url):
            continue

        scope, location_label = _classify_scope_and_location(title, snippet, city, county)
        project_type = _infer_project_type(title, snippet)

        # Fetch HTML and extract deeper facts (best-effort; failures are fine)
        html = _fetch_html(url) if url else ""
        facts = _extract_project_facts_from_html(html)
        company_guess = _guess_company_from_title_snippet(title, snippet)

        # Timeline: use metadata date if present, otherwise fallback to HTML year hint
        timeline_year: Optional[int] = dt.year if isinstance(dt, datetime) else None
        if not timeline_year and facts.get("year"):
            try:
                timeline_year = int(facts["year"])
            except Exception:
                timeline_year = None

        if inferred_year and (timeline_year is not None) and (timeline_year != inferred_year):
            timeline_stage = "outside requested timeframe"
        else:
            timeline_stage = "announcement" if timeline_year else "not specified in snippet"

        projects.append(
            {
                # Core identity
                "project_name": title or "Untitled project",
                "company": company_guess,  # may be None
                "project_type": project_type,

                # Geography
                "scope": scope,  # "local" or "statewide"
                "location_label": location_label,
                "original_area_label": original_area_label,

                # Scale / economics
                "sqft": facts.get("sqft"),
                "jobs": facts.get("jobs"),
                "investment": facts.get("investment"),

                # Timeline
                "timeline_stage": timeline_stage,
                "timeline_year": timeline_year,

                # Source
                "url": url,
                "provider": provider,
                "snippet": snippet,
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

    - Extracts simple city/county hints from the user question.
    - Builds a Google CSE query anchored on Indiana industrial/commercial projects.
    - First attempts a county/city-biased search.
    - If no results at all, falls back to a statewide search (no city/county constraints)
      so the chat layer can still talk about major Indiana projects.
    - Returns a list of normalized project dicts:

      {
        "project_name": str,
        "company": Optional[str],
        "project_type": str,
        "scope": "local" | "statewide",
        "location_label": str,
        "original_area_label": str,
        "sqft": Optional[str],
        "jobs": Optional[str],
        "investment": Optional[str],
        "timeline_stage": str,
        "timeline_year": Optional[int],
        "url": str,
        "provider": str,
        "snippet": str,
      }

    NOTE: We currently do NOT hard-filter by `days` – your app passes ~10 years
    to ensure we almost always have something. If you want strict recency, add
    a filter here based on the `date` field we already parse.
    """
    city, county = _extract_geo_hint(user_q)
    original_area_label = county or city or "Indiana"
    _ = original_area_label  # reserved for future use

    # 1) County/city-biased search
    query_local = _build_query(user_q, city, county)
    raw_local = _google_cse_search(query_local)

    # 2) If nothing at all came back, fall back to a statewide search
    if not raw_local:
        log.info("No local results; trying statewide fallback query")
        query_statewide = _build_query(user_q, city=None, county=None)
        raw_statewide = _google_cse_search(query_statewide)
        if not raw_statewide:
            return []

        projects = _normalize_projects(raw_statewide, city=None, county=None, user_q=user_q)
        return projects[:max_items]

    # 3) We have some local-biased hits; normalize them.
    projects = _normalize_projects(raw_local, city=city, county=county, user_q=user_q)
    return projects[:max_items]


def render_developments_markdown(items: List[Dict[str, Any]]) -> str:
    """
    Simple markdown-ish debug formatter. This is what your /api/chat mode
    passes into the model as context, so we include the richer facts here.
    """
    if not items:
        return (
            "No web results were found for that location and timeframe. "
            "Try adjusting the date range or phrasing."
        )

    lines: List[str] = []
    lines.append("Recent Indiana projects (web search hits):")

    for i, item in enumerate(items[:15], start=1):
        title = item.get("project_name") or item.get("title") or "Untitled"
        snippet = (item.get("snippet") or "").strip()
        url = item.get("url") or ""
        provider = item.get("provider") or ""
        year = item.get("timeline_year")
        stage = item.get("timeline_stage") or ""
        loc = item.get("location_label") or item.get("original_area_label") or "Indiana"
        company = item.get("company") or "not specified"
        sqft = item.get("sqft") or "not specified"
        jobs = item.get("jobs") or "not specified"
        investment = item.get("investment") or "not specified"

        lines.append(f"{i}. {title} — {loc}")
        meta_bits = []
        if provider:
            meta_bits.append(provider)
        if stage:
            if year:
                meta_bits.append(f"{stage} ({year})")
            else:
                meta_bits.append(stage)
        if meta_bits:
            lines.append("   " + " • ".join(meta_bits))

        lines.append(f"   Company: {company}")
        lines.append(f"   Type: {item.get('project_type') or 'not specified'}")
        lines.append(f"   Scale: sqft={sqft}; jobs={jobs}; investment={investment}")
        if snippet:
            lines.append(f"   Snippet: {snippet}")
        if url:
            lines.append(f"   URL: {url}")

    return "\n".join(lines)


__all__ = [
    "search_indiana_developments",
    "render_developments_markdown",
]
