"""
indiana_intel.py

Indiana developments lead-finder for Tynan / Heli AI site.

Goals:
- Given a natural-language question like:
    "Any new warehouses or distribution centers announced in Hendricks County
     in the last 12 months?"
- Use Google Custom Search JSON API (Programmable Search Engine)
  to find *actual facility projects*:
    - New / expanded warehouses, distribution centers, logistics hubs
    - Manufacturing / production plants
    - Industrial / business parks
    - Major offices / HQs with potential warehouse/logistics tie-in
- Extract:
    - Company coming to the area (best-effort)
    - Project type (warehouse, plant, office, park, etc.)
    - Project scope (sq ft, jobs, $, if present)
    - Rough timing (announcement / groundbreaking / opening)
    - Source URL

Environment variables required:
- GOOGLE_CSE_KEY : Google API key for Custom Search JSON API
- GOOGLE_CSE_CX  : Programmable Search Engine ID (cx) configured to search the web
"""

from __future__ import annotations

import os
import re
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None  # type: ignore

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

# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

STOPWORDS = {
    "what", "are", "there", "any", "new", "or", "in", "the", "last", "month",
    "months", "recent", "recently", "project", "projects", "have", "has",
    "been", "announced", "announcement", "for", "about", "on", "of", "a",
    "an", "county", "indiana", "logistics", "warehouse", "distribution",
    "center", "centers", "companies", "coming", "to", "area", "city",
    "kind", "sort", "type", "news", "update",
}


def _extract_geo_hint(q: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Very light-weight extractor:
    - Finds 'X County' → county='X County'
    - Tries to grab a city name after 'in ' if it looks like 'Greenfield' or 'Avon, IN'
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

    # City: look for "in Avon", "around Avon, IN", etc.
    m_city = re.search(r"\b(?:in|around|near)\s+([A-Za-z\s]+?)(?:,|\?|\.|$)", text)
    city = None
    if m_city:
        raw = m_city.group(1).strip()
        raw = re.sub(r"\b(Indiana|IN)\b\.?", "", raw, flags=re.I).strip()
        if raw:
            city = raw

    return (city, county)


def _build_query(user_q: str, city: Optional[str], county: Optional[str]) -> str:
    """
    Build a Google CSE query that is biased toward *real facility projects*.

    We look for phrases like:
    - "new distribution center", "to build a new warehouse", "manufacturing plant",
      "to locate in", "groundbreaking", "investment", "creates X jobs".
    """
    geo_bits: List[str] = []
    if county:
        geo_bits.append(f'"{county}"')
    if city:
        geo_bits.append(f'"{city}, Indiana" OR "{city} IN" OR "{city}"')
    geo_bits.append('"Indiana"')

    core_phrases = [
        '"new distribution center"',
        '"new warehouse"',
        '"logistics center"',
        '"fulfillment center"',
        '"manufacturing plant"',
        '"production plant"',
        '"industrial park"',
        '"business park"',
        '"to locate in"',
        '"plans to build"',
        '"will build"',
        '"broke ground"',
        '"groundbreaking"',
        '"expansion"',
        '"investing"',
        '"investment"',
        '"creates" OR "creating" OR "add" jobs',
    ]

    # Pull a few non-stopword tokens from the user question as soft hints
    cleaned = re.sub(r"[“”\"']", " ", user_q or "")
    tokens = re.findall(r"[A-Za-z0-9]+", cleaned)
    extra_tokens = []
    for tok in tokens:
        tl = tok.lower()
        if tl in STOPWORDS:
            continue
        extra_tokens.append(tok)
    tail = " ".join(extra_tokens[:6]) if extra_tokens else ""

    parts: List[str] = []
    parts.append("(" + " OR ".join(geo_bits) + ")")
    parts.append("(" + " OR ".join(core_phrases) + ")")
    if tail:
        parts.append("(" + tail + ")")

    query = " ".join(parts)
    log.info("Google CSE query: %s", query)
    return query


# ---------------------------------------------------------------------------
# Google CSE / HTML fetch
# ---------------------------------------------------------------------------

def _parse_date_from_pagemap(it: Dict[str, Any]) -> Optional[datetime]:
    """
    Try to sniff out a publication/update date from Google CSE pagemap metatags.
    Returns datetime (UTC, naive) or None.
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
                try:
                    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                except Exception:
                    dt = None
                    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
                        try:
                            dt = datetime.strptime(raw[:10], fmt)
                            dt = dt.replace(tzinfo=None)
                            break
                        except Exception:
                            dt = None
                    if not dt:
                        continue
                if dt.tzinfo is not None:
                    dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
                return dt

    return None


def _fetch_page_text(url: str, timeout: int = 8) -> str:
    """
    Fetch the HTML for a page and return a condensed text blob (title, headings,
    and first paragraphs). Requires BeautifulSoup to be installed; otherwise
    returns an empty string and we fall back to the Google snippet.
    """
    if BeautifulSoup is None:
        return ""

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; TynanIndianaIntel/1.0)"
        }
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        log.info("Failed to fetch page HTML: %s (%s)", url, e)
        return ""

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception as e:
        log.info("BeautifulSoup parsing failed for %s (%s)", url, e)
        return ""

    pieces: List[str] = []

    if soup.title and soup.title.string:
        pieces.append(soup.title.string)

    for tag in soup.find_all(["h1", "h2", "h3"]):
        text = tag.get_text(separator=" ", strip=True)
        if text:
            pieces.append(text)

    paragraphs = 0
    for p in soup.find_all("p"):
        text = p.get_text(separator=" ", strip=True)
        if text:
            pieces.append(text)
            paragraphs += 1
        if paragraphs >= 6:
            break

    text = " ".join(pieces)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:5000]  # hard cap


def _google_cse_search(
    query: str,
    days: Optional[int],
    max_items: int,
) -> List[Dict[str, Any]]:
    """
    Call Google CSE and return normalized raw items (no filtering yet).
    """
    if not GOOGLE_CSE_KEY or not GOOGLE_CSE_CX:
        log.warning(
            "GOOGLE_CSE_KEY or GOOGLE_CSE_CX not set or empty; returning empty list "
            f"(GOOGLE_CSE_KEY present={bool(GOOGLE_CSE_KEY)}, GOOGLE_CSE_CX present={bool(GOOGLE_CSE_CX)})"
        )
        return []

    max_items = max(1, min(max_items, 30))
    out: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()
    start = 1

    while len(out) < max_items and start <= 91:
        remaining = max_items - len(out)
        num = min(10, remaining)

        params = {
            "key": GOOGLE_CSE_KEY,
            "cx": GOOGLE_CSE_CX,
            "q": query,
            "num": num,
            "start": start,
        }
        if days and days > 0:
            params["dateRestrict"] = f"d{days}"

        try:
            resp = requests.get(GOOGLE_CSE_ENDPOINT, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.warning("Google CSE request failed (start=%s): %s", start, e)
            break

        items = data.get("items", []) or []
        if not items:
            break

        for it in items:
            if not isinstance(it, dict):
                continue
            url = it.get("link") or ""
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            title = it.get("title") or ""
            snippet = it.get("snippet") or it.get("htmlSnippet") or ""
            provider = it.get("displayLink") or ""
            dt = _parse_date_from_pagemap(it)

            out.append(
                {
                    "title": title,
                    "snippet": snippet,
                    "url": url,
                    "provider": provider,
                    "date_dt": dt,  # may be None
                }
            )

        start += len(items)
        if len(items) < num:
            break

    log.info("Google CSE returned %s unique items", len(out))
    return out


# ---------------------------------------------------------------------------
# Extraction: decide if it's a real project and pull scope/company/timing
# ---------------------------------------------------------------------------

PROJECT_POSITIVE_KEYWORDS = (
    "new facility",
    "new distribution center",
    "new warehouse",
    "new logistics center",
    "build a new",
    "plans to build",
    "will build",
    "to build a",
    "to locate in",
    "will locate in",
    "plans to locate in",
    "manufacturing plant",
    "production plant",
    "factory",
    "assembly plant",
    "industrial park",
    "business park",
    "logistics park",
    "broke ground",
    "groundbreaking",
    "expansion",
    "expanding",
    "investing",
    "investment",
    "sq ft",
    "square-foot",
    "square feet",
    "jobs",
)

PROJECT_NEGATIVE_KEYWORDS = (
    "annual report",
    "tif annual report",
    "ordinances",
    "resolutions",
    "agenda",
    "meeting minutes",
    "visitor bureau",
    "visit hendricks county",
    "events, shopping & family fun",
    "things to do",
    "partners with",
    "raise funds",
    "fundraiser",
    "charity event",
    "concert",
    "festival",
    "pavilion center",
    "park pavilion",
    "mental health",
    "addiction services",
    "convention center",
    "stadium",
    "tourism",
)


def _looks_like_real_project(title: str, snippet: str, page_text: str) -> bool:
    """
    Decide whether this looks like an actual facility project worth showing.

    - Requires at least one strong positive signal (in snippet or page text).
    - Rejects obvious admin/tourism/fundraiser noise.
    """
    text = f"{title} {snippet} {page_text}".lower()

    if any(k in text for k in PROJECT_NEGATIVE_KEYWORDS):
        return False

    if not any(k in text for k in PROJECT_POSITIVE_KEYWORDS):
        return False

    return True


def _extract_scope(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Grab square footage, jobs, and investment from a text blob.
    Returns (sqft_str, jobs_str, investment_str).
    """
    if not text:
        return (None, None, None)

    sqft = None
    jobs = None
    investment = None

    # sq ft
    m_sqft = re.search(
        r"([\d,]+)\s*(square[-\s]?foot|square[-\s]?feet|sq\.?\s*ft|sf)",
        text,
        flags=re.I,
    )
    if m_sqft:
        sqft = m_sqft.group(1).replace(",", "")

    # jobs
    m_jobs = re.search(r"(\d{2,5})\s+(?:new\s+)?jobs", text, flags=re.I)
    if m_jobs:
        jobs = m_jobs.group(1)

    # investment
    m_inv = re.search(r"\$[\d,]+(?:\.\d+)?\s*(million|billion)?", text, flags=re.I)
    if m_inv:
        investment = m_inv.group(0)

    return sqft, jobs, investment


def _classify_facility_type(text: str) -> str:
    """
    Classify as warehouse / plant / office / industrial park / other.
    """
    t = text.lower()

    if any(k in t for k in ("distribution center", "fulfillment center", "logistics center", "logistics hub")):
        return "Warehouse / logistics center"
    if "warehouse" in t:
        return "Warehouse"
    if any(k in t for k in ("manufacturing plant", "production plant", "factory", "assembly plant")):
        return "Manufacturing / production plant"
    if any(k in t for k in ("industrial park", "business park", "commerce park", "logistics park")):
        return "Industrial / business park"
    if any(k in t for k in ("headquarters", "hq", "office building", "corporate office")):
        return "Office / headquarters"
    return "Industrial / commercial project"


def _infer_company_name(text: str, title: str, provider: str, url: str) -> str:
    """
    Best-effort company extraction:
    - Look for "X will build", "X plans to build", etc.
    - Fall back to the provider or host name.
    """
    t = text[:400]
    patterns = [
        r"([A-Z][A-Za-z0-9&\.\-]*(?:\s+[A-Z][A-Za-z0-9&\.\-]*){0,4})\s+(?:will|plans|plan|announced|announces|to)\b",
        r"([A-Z][A-Za-z0-9&\.\-]*(?:\s+[A-Z][A-Za-z0-9&\.\-]*){0,4})\s+is\s+investing\b",
    ]
    for pat in patterns:
        m = re.search(pat, t)
        if m:
            name = m.group(1).strip()
            if len(name.split()) <= 6:
                return name

    # Fallback: first chunk of title before dash/pipe
    base_title = re.split(r"[-|–—]", title)[0].strip()
    if base_title and len(base_title.split()) <= 6:
        return base_title

    # Fallback to provider or host
    if provider:
        return provider

    host = re.sub(r"^https?://", "", url).split("/")[0]
    host = host.replace("www.", "")
    return host or "Unknown"


def _infer_timeline(dt: Optional[datetime], text: str) -> Tuple[str, Optional[int]]:
    """
    Return (stage_label, year) like:
      ("announcement", 2025)
      ("groundbreaking", 2024)
      ("opening", 2026)
    """
    if not dt:
        return ("not specified", None)

    year = dt.year
    low = text.lower()

    if any(k in low for k in ("broke ground", "groundbreaking", "shovels in the ground", "shovel-ready")):
        return ("groundbreaking", year)
    if any(k in low for k in ("opens in", "opening in", "set to open", "will open", "operational in")):
        return ("opening / operational", year)
    if any(k in low for k in ("expansion", "expanding", "expand", "expanded")):
        return ("expansion announcement", year)
    return ("announcement", year)


def _score_relevance(user_q: str, text: str) -> int:
    """
    Simple overlap score between user question and text.
    """
    cleaned_q = re.sub(r"[“”\"']", " ", user_q or "")
    q_tokens = {
        t.lower()
        for t in re.findall(r"[A-Za-z0-9]+", cleaned_q)
        if t.lower() not in STOPWORDS
    }
    haystack = text.lower()
    score = 0
    for tok in q_tokens:
        if tok and tok in haystack:
            score += 1
    return score


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_indiana_developments(
    user_q: str,
    days: int = 365,
    max_items: int = 20,
) -> List[Dict[str, Any]]:
    """
    Main entrypoint.

    Returns a list of *actual projects* (filtered and enriched), each like:

    {
      "project_name": str,
      "company": str,
      "project_type": str,          # e.g. "Warehouse", "Manufacturing / production plant"
      "location_label": str,        # "Hendricks County, Indiana" etc.
      "sqft": Optional[str],
      "jobs": Optional[str],
      "investment": Optional[str],
      "timeline_stage": str,        # "announcement", "groundbreaking", ...
      "timeline_year": Optional[int],
      "url": str,
      "provider": str,
      "raw_snippet": str,
    }
    """
    city_hint, county_hint = _extract_geo_hint(user_q)
    query = _build_query(user_q, city_hint, county_hint)

    raw_items = _google_cse_search(query, days=days, max_items=max_items)
    if not raw_items:
        return []

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    cutoff = now - timedelta(days=max(days, 1))

    projects: List[Dict[str, Any]] = []

    for item in raw_items:
        title = item.get("title") or ""
        snippet = item.get("snippet") or ""
        url = item.get("url") or ""
        provider = item.get("provider") or ""
        dt = item.get("date_dt")

        if dt and dt < cutoff:
            continue

        page_text = _fetch_page_text(url)
        combined_text = " ".join([title, snippet, page_text])
        combined_text = re.sub(r"\s+", " ", combined_text).strip()

        # Hard filter: only actual facility projects
        if not _looks_like_real_project(title, snippet, combined_text):
            continue

        # Extract scope + classification
        sqft, jobs, investment = _extract_scope(combined_text)
        project_type = _classify_facility_type(combined_text)
        company = _infer_company_name(combined_text, title, provider, url)
        stage, year = _infer_timeline(dt, combined_text)
        score = _score_relevance(user_q, combined_text)

        if county_hint:
            location_label = f"{county_hint}, Indiana"
        elif city_hint:
            location_label = f"{city_hint}, Indiana"
        else:
            location_label = "Indiana"

        projects.append(
            {
                "project_name": title or "Unnamed project",
                "company": company,
                "project_type": project_type,
                "location_label": location_label,
                "sqft": sqft,
                "jobs": jobs,
                "investment": investment,
                "timeline_stage": stage,
                "timeline_year": year,
                "url": url,
                "provider": provider,
                "raw_snippet": snippet,
                "_score": score,
                "_date_dt": dt or datetime.min,
            }
        )

    if not projects:
        return []

    # Sort by relevance then recency
    projects.sort(key=lambda p: (p["_score"], p["_date_dt"]), reverse=True)
    for p in projects:
        p.pop("_score", None)
        p.pop("_date_dt", None)

    return projects


def render_developments_markdown(items: List[Dict[str, Any]]) -> str:
    """
    Turn the structured project list into a compact, readable summary string
    for your chat UI.

    If there are no items, returns a clear "no real projects found" message.
    """
    if not items:
        return (
            "Based on web search results, there do not appear to be clearly "
            "identified new warehouse, logistics, or manufacturing projects in this area "
            "within the recent period covered by this search. Most public results are "
            "generic county / tourism / public-service pages that do not represent "
            "specific new facilities."
        )

    area_label = items[0].get("location_label") or "Indiana"

    lines: List[str] = []
    lines.append(
        f"Here are some confirmed facility projects connected to {area_label} based on web search results.\n"
    )

    for proj in items[:15]:
        name = proj.get("project_name") or "Unnamed project"
        company = proj.get("company") or "Unknown"
        ptype = proj.get("project_type") or "Industrial / commercial project"
        sqft = proj.get("sqft")
        jobs = proj.get("jobs")
        invest = proj.get("investment")
        stage = proj.get("timeline_stage") or "not specified"
        year = proj.get("timeline_year")
        url = proj.get("url") or ""
        snippet = proj.get("raw_snippet") or ""

        # Project heading
        lines.append(f"{name}")
        lines.append(f"Company: {company}")
        lines.append(f"Type: {ptype}")

        # Scope line
        scope_bits = []
        if sqft:
            scope_bits.append(f"~{sqft} sq ft")
        if jobs:
            scope_bits.append(f"{jobs} jobs")
        if invest:
            scope_bits.append(f"{invest} investment")
        if scope_bits:
            lines.append("Scope: " + ", ".join(scope_bits))
        else:
            lines.append("Scope: not specified in available text")

        # Timeline
        if year:
            lines.append(f"Timeline: {stage} ({year})")
        else:
            lines.append(f"Timeline: {stage}")

        # Source
        if url:
            lines.append(f"Source: {url}")

        # Brief notes
        s_clean = " ".join(snippet.split())
        if s_clean:
            if len(s_clean) > 260:
                s_clean = s_clean[:257].rstrip() + "..."
            lines.append(f"Notes: {s_clean}")

        lines.append("")

    return "\n".join(lines).rstrip()


__all__ = [
    "search_indiana_developments",
    "render_developments_markdown",
]
