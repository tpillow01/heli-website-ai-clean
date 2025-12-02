"""
indiana_intel.py

Indiana developments lead-finder for Tynan / Heli AI site.

Responsibilities:
- Take a natural-language question (e.g. "new warehouses in Boone County in the last 90 days")
- Hit Google Custom Search JSON API (Programmable Search Engine) for
  Indiana industrial / logistics / manufacturing / commercial projects
- Normalize results into simple dicts
- Provide a compact text summary for the chat UI or as context into GPT

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
    - Tries to grab a city name after 'in ' if it looks like 'Greenwood' or 'Greenwood, IN'
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

    # City: look for "in Greenwood", "around Greenwood, IN", etc.
    m_city = re.search(r"\b(?:in|around|near)\s+([A-Za-z\s]+?)(?:,|\?|\.|$)", text)
    city = None
    if m_city:
        raw = m_city.group(1).strip()
        raw = re.sub(r"\b(Indiana|IN)\b\.?", "", raw, flags=re.I).strip()
        if raw:
            city = raw

    return (city, county)


_STOPWORDS = {
    "what", "are", "there", "any", "new", "or", "in", "the", "last", "month",
    "months", "recent", "recently", "project", "projects", "have", "has",
    "been", "announced", "announcement", "for", "about", "on", "of", "a",
    "an", "county", "indiana", "logistics", "warehouse", "distribution",
    "center", "centers", "companies", "coming", "to", "area", "city",
    "kind", "sort", "type",
}


def _build_query(user_q: str, city: Optional[str], county: Optional[str]) -> str:
    """
    Build a Google CSE query anchored on Indiana industrial/commercial keywords,
    with optional city / county bias, plus a small keyword tail from the question.
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


def _parse_date_from_pagemap(it: Dict[str, Any]) -> Optional[str]:
    """
    Try to sniff out a publication/update date from Google CSE pagemap metatags.
    Returns ISO string (UTC, naive) or None.
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
                            dt = dt.replace(tzinfo=None)
                            break
                        except Exception:
                            dt = None
                    if not dt:
                        continue

                # Normalize to UTC naive
                if dt.tzinfo is not None:
                    dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
                return dt.isoformat()

    return None


def _score_relevance(user_q: str, title: str, snippet: str) -> int:
    """
    Crude relevance score based on overlapping non-stopword tokens between
    the user question and the result's title+snippet.
    """
    cleaned_q = re.sub(r"[“”\"']", " ", user_q or "")
    q_tokens = {
        t.lower()
        for t in re.findall(r"[A-Za-z0-9]+", cleaned_q)
        if t.lower() not in _STOPWORDS
    }

    haystack = f"{title} {snippet}".lower()
    score = 0
    for tok in q_tokens:
        if tok and tok in haystack:
            score += 1
    return score


def _classify_and_describe_project(title: str, snippet: str) -> Tuple[str, str, bool]:
    """
    Classify what kind of project this is, and produce a short 'what this is'
    description PLUS a forklift_relevant flag indicating whether this is
    something Tynan should realistically care about.

    forklift_relevant == True only for:
    - Warehouse / logistics
    - Manufacturing / plant
    - Industrial / business park
    - Office / HQ (borderline – might have associated warehouse)
    """
    text = f"{title} {snippet}".lower()

    # Tourism / events / visitor bureau
    if any(k in text for k in ("visit hendricks county", "visitors bureau", "tourism", "shopping & family fun")):
        proj_type = "Tourism / visitor information (not a specific project)"
        desc = "Tourism / visitor information site, not a specific warehouse, logistics, or manufacturing project."
        return proj_type, desc, False

    # Ordinances, county government pages
    if any(k in text for k in ("ordinances", "resolutions", "county council", "board of commissioners", "clerk's office")):
        proj_type = "County government information (not a specific project)"
        desc = "County government / ordinances information page, not a distinct construction or industrial project."
        return proj_type, desc, False

    # Encyclopedia / general county history
    if any(k in text for k in ("encyclopedia of indianapolis", "history of hendricks county", "county profile")):
        proj_type = "General county information (not a specific project)"
        desc = "General informational / historical page about the county, not a specific facility or project."
        return proj_type, desc, False

    # Civic / arts / events
    if any(k in text for k in ("theatre", "theater", "performing arts", "concert", "music venue", "stage", "arts center", "event space", "civic center")):
        proj_type = "Civic / arts venue"
        desc = "Performing arts and events venue (theatre / event space / public gathering space), not a warehouse or manufacturing facility."
        return proj_type, desc, False

    # Public services / recycling / utilities
    if any(k in text for k in ("yard waste", "recycling center", "solid waste", "transfer station", "landfill", "compost", "hazardous waste")):
        proj_type = "Public services / recycling facility"
        desc = "Public yard-waste / recycling / solid waste facility; could involve loaders or light equipment, but not a core warehouse/logistics or manufacturing lead."
        return proj_type, desc, False

    # Warehouse / logistics
    if any(k in text for k in ("distribution center", "fulfillment center", "logistics park", "logistics center", "cross-dock", "bulk warehouse")):
        proj_type = "Warehouse / logistics"
        desc = "Warehouse or logistics facility focused on storage, distribution, or fulfillment operations."
        return proj_type, desc, True

    if "warehouse" in text:
        proj_type = "Warehouse / logistics"
        desc = "Warehouse facility that may involve storage, order picking, shipping, and receiving operations."
        return proj_type, desc, True

    # Manufacturing / plant
    if any(k in text for k in ("manufacturing", "manufacturing plant", "production plant", "factory", "assembly plant", "industrial plant")):
        proj_type = "Manufacturing / plant"
        desc = "Manufacturing or production plant involving industrial processes and material handling."
        return proj_type, desc, True

    # Industrial / business park
    if any(k in text for k in ("industrial park", "business park", "commerce park", "logistics park")):
        proj_type = "Industrial / business park"
        desc = "Industrial or business park with multiple commercial / industrial buildings and potential warehouse or manufacturing tenants."
        return proj_type, desc, True

    # Office / HQ
    if any(k in text for k in ("headquarters", "hq", "office building", "corporate office")):
        proj_type = "Office / headquarters"
        desc = "Office or headquarters project; forklift opportunities depend on associated warehouse / logistics space."
        return proj_type, desc, True  # borderline but worth showing

    # Fallback: treat as generic industrial/commercial; low-confidence forklift relevance
    proj_type = "Industrial / commercial project (unspecified)"
    snippet_clean = " ".join((snippet or "").split())
    if snippet_clean:
        desc = snippet_clean
    else:
        desc = "Industrial / commercial project; the snippet did not provide additional details."
    # If we don't see any strong industrial/logistics/manufacturing cues, be conservative:
    forklift_relevant = any(
        k in text
        for k in ("facility", "plant", "industrial", "logistics", "warehouse", "distribution")
    )
    return proj_type, desc, forklift_relevant


def _build_scope_summary(snippet: str) -> str:
    """
    Try to pull out rough scope from the snippet:
    - Square footage
    - Jobs created
    - Investment amount
    Returns a short human-readable string, or 'not specified in snippet'.
    """
    if not snippet:
        return "not specified in snippet"

    text = " ".join(snippet.split())
    low = text.lower()

    sqft = None
    jobs = None
    investment = None

    # Approx square footage
    m_sqft = re.search(
        r"([\d,]+)\s*(square[-\s]?foot|square[-\s]?feet|sq\.?\s*ft|sf)",
        low,
        flags=re.I,
    )
    if m_sqft:
        sqft_raw = m_sqft.group(1)
        sqft = sqft_raw.replace(",", "")

    # Jobs
    m_jobs = re.search(r"(\d{2,5})\s+(?:new\s+)?jobs", low, flags=re.I)
    if m_jobs:
        jobs = m_jobs.group(1)

    # Investment
    m_inv = re.search(r"\$[\d,\.]+\s*(million|billion)?", text, flags=re.I)
    if m_inv:
        investment = m_inv.group(0)

    bits: List[str] = []
    if sqft:
        bits.append(f"approx. {sqft}-square-foot facility")
    if jobs:
        bits.append(f"around {jobs} jobs")
    if investment:
        bits.append(f"{investment} investment")

    if bits:
        return ", ".join(bits)

    return "not specified in snippet"


def _infer_timeline_label(date_str: Optional[str], snippet: str) -> str:
    """
    Turn date + wording into a friendlier timeline label:
    - 'groundbreaking year 2024'
    - 'opening/operational year 2026'
    - 'announcement year 2025'
    """
    if not date_str:
        return "not specified in snippet"

    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        year = dt.year
    except Exception:
        return "not specified in snippet"

    low = (snippet or "").lower()

    if any(k in low for k in ("broke ground", "groundbreaking", "shovels in the ground", "shovel-ready")):
        return f"groundbreaking year {year}"
    if any(k in low for k in ("opens in", "opening in", "set to open", "will open", "operational in")):
        return f"opening/operational year {year}"
    if any(k in low for k in ("expansion", "expanding", "expand", "expanded")):
        return f"expansion announced year {year}"
    return f"announcement year {year}"


def _short_snippet(snippet: str, max_len: int = 260) -> str:
    """
    Clean and truncate the Google snippet so it can be shown as 'Notes:'.
    """
    if not snippet:
        return "No additional details available in the snippet."

    s = " ".join(snippet.split())
    if len(s) <= max_len:
        return s
    return s[: max_len - 3].rstrip() + "..."


def _infer_company_name(it: Dict[str, Any], url: str, title: str) -> Optional[str]:
    """
    Best-effort attempt to infer the company / developer name from Google's pagemap
    and the URL. Never guesses wildly; falls back to the host name if needed.
    """
    pagemap = it.get("pagemap") or {}

    # Organization-like objects
    for key in ("organization", "localbusiness", "corporation", "place"):
        entries = pagemap.get(key) or []
        if isinstance(entries, list):
            for ent in entries:
                if isinstance(ent, dict):
                    name = ent.get("name") or ent.get("legalname") or ent.get("legal_name")
                    if name:
                        return str(name).strip()

    # Metatags like og:site_name
    metas = pagemap.get("metatags") or []
    if isinstance(metas, list):
        for m in metas:
            if not isinstance(m, dict):
                continue
            for key in ("og:site_name", "twitter:title", "application-name"):
                if key in m and m[key]:
                    cand = str(m[key]).strip()
                    if cand and len(cand) >= 3:
                        return cand

    # Try to peel a clean name from the title before a dash / pipe
    base_title = re.split(r"[-|–—]", title)[0].strip()
    if base_title and len(base_title.split()) <= 6:
        return base_title

    # Fallback to host
    host = re.sub(r"^https?://", "", url).split("/")[0]
    host = host.replace("www.", "")
    return host or None


def _google_cse_search(
    query: str,
    max_items: int = 10,
    days: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Thin wrapper around Google Custom Search JSON API.
    Returns a list of normalized raw results (up to max_items).
    Uses pagination and optional dateRestrict (days).
    """
    if not GOOGLE_CSE_KEY or not GOOGLE_CSE_CX:
        log.warning(
            "GOOGLE_CSE_KEY or GOOGLE_CSE_CX not set or empty; returning empty result list. "
            f"GOOGLE_CSE_KEY present={bool(GOOGLE_CSE_KEY)}, GOOGLE_CSE_CX present={bool(GOOGLE_CSE_CX)}"
        )
        return []

    max_items = max(1, min(max_items, 30))  # hard safety cap

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
        log.info("Google CSE returned %s items for start=%s", len(items), start)
        if not items:
            break

        for it in items:
            if not isinstance(it, dict):
                continue

            title = it.get("title") or ""
            snippet = it.get("snippet") or it.get("htmlSnippet") or ""
            url = it.get("link") or ""
            provider = it.get("displayLink") or ""

            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            dt_iso = _parse_date_from_pagemap(it)
            company_name = _infer_company_name(it, url, title)
            project_type, what_this_is, forklift_relevant = _classify_and_describe_project(title, snippet)

            out.append(
                {
                    "title": title,
                    "snippet": snippet,
                    "url": url,
                    "provider": provider,
                    "date": dt_iso,            # may be None
                    "company_name": company_name,
                    "project_type": project_type,
                    "what_this_is": what_this_is,
                    "forklift_relevant": forklift_relevant,
                }
            )

        start += len(items)
        if len(items) < num:
            break

    return out


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

    - Extracts simple city/county hints from the user question.
    - Builds a Google CSE query anchored on Indiana industrial/commercial projects.
    - Uses Google dateRestrict + local filtering to bias towards the last `days`.
    - Returns up to `max_items` normalized dicts:
      {
        "title": str,
        "snippet": str,
        "url": str,
        "provider": str,
        "date": ISO string or None,
        "city_hint": Optional[str],
        "county_hint": Optional[str],
        "score": int,                 # crude relevance score
        "project_type": str,
        "what_this_is": str,
        "company_name": Optional[str],
        "forklift_relevant": bool,
      }
    """
    city, county = _extract_geo_hint(user_q)
    query = _build_query(user_q, city, county)

    raw_results = _google_cse_search(query, max_items=max_items, days=days)
    if not raw_results:
        return []

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    cutoff = now - timedelta(days=max(days, 1))

    enriched: List[Dict[str, Any]] = []
    for r in raw_results:
        title = r.get("title") or ""
        snippet = r.get("snippet") or ""
        dt_iso = r.get("date")

        # Parse date, if present
        parsed_dt: Optional[datetime] = None
        if dt_iso:
            try:
                parsed_dt = datetime.fromisoformat(dt_iso.replace("Z", "+00:00"))
                if parsed_dt.tzinfo is not None:
                    parsed_dt = parsed_dt.astimezone(timezone.utc).replace(tzinfo=None)
            except Exception:
                parsed_dt = None

        # Soft filter by recency only if we actually got a date
        if parsed_dt and parsed_dt < cutoff:
            continue

        score = _score_relevance(user_q, title, snippet)

        r["city_hint"] = city
        r["county_hint"] = county
        r["score"] = score
        r["_parsed_dt"] = parsed_dt  # private helper key for sorting

        enriched.append(r)

    # Sort by (score desc, date desc, title)
    def _sort_key(item: Dict[str, Any]) -> Tuple[int, datetime, str]:
        score = int(item.get("score") or 0)
        dt = item.get("_parsed_dt") or datetime.min
        title = item.get("title") or ""
        return (score, dt, title.lower())

    enriched.sort(key=_sort_key, reverse=True)

    for r in enriched:
        r.pop("_parsed_dt", None)

    return enriched


def render_developments_markdown(items: List[Dict[str, Any]]) -> str:
    """
    Structured plain-text summary for chat UI or as context into GPT.

    Behavior:
    - Only shows items where forklift_relevant == True (warehouse/logistics/manufacturing/etc.).
    - If none exist, explains clearly that no such projects were found and
      notes that search hits were general county / tourism / public-service pages.
    """
    if not items:
        return (
            "No web results were found for that location and timeframe. "
            "Try adjusting the date range or phrasing."
        )

    # Split into forklift-relevant and not
    forklift_items = [it for it in items if it.get("forklift_relevant")]
    non_forklift_items = [it for it in items if not it.get("forklift_relevant")]

    # Heading based on first item's location hints
    first = items[0]
    county_hint = (first.get("county_hint") or "").strip()
    city_hint = (first.get("city_hint") or "").strip()

    if county_hint:
        area_label = county_hint
    elif city_hint:
        area_label = f"{city_hint}, Indiana"
    else:
        area_label = "Indiana"

    # If we have no forklift-relevant results, say that plainly
    if not forklift_items:
        # Optional: describe what the non-forklift hits were, at a high level
        categories = set()
        for it in non_forklift_items:
            pt = (it.get("project_type") or "").strip()
            if pt:
                categories.add(pt)

        cat_list = ", ".join(sorted(categories)) if categories else "general information pages"

        return (
            f"Based on web search results, there do not appear to be clearly identified new "
            f"warehouse, logistics, or manufacturing projects in {area_label} within the recent period "
            f"covered by this search.\n\n"
            f"Most of the search hits were {cat_list}, which are not specific industrial projects "
            f"and have been omitted from this list so you don’t chase bad leads."
        )

    # Otherwise, show only forklift-relevant projects
    lines: List[str] = []
    lines.append(
        f"Here are some warehouse, logistics, manufacturing, or industrial park projects connected to {area_label} based on web search results.\n"
    )

    for item in forklift_items[:15]:
        title = item.get("title") or "Untitled"
        snippet = (item.get("snippet") or "").strip()
        url = item.get("url") or ""
        provider = item.get("provider") or ""
        date = item.get("date") or ""
        proj_type = item.get("project_type") or "Industrial / commercial project"
        what_this_is = item.get("what_this_is") or ""
        company_name = item.get("company_name") or ""
        city_hint = (item.get("city_hint") or "").strip()
        county_hint = (item.get("county_hint") or "").strip()

        # Normalize date display
        if date:
            try:
                dt = datetime.fromisoformat(date.replace("Z", "+00:00"))
                if dt.tzinfo is not None:
                    dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
                date = dt.strftime("%Y-%m-%d")
            except Exception:
                pass

        # Location label
        if county_hint:
            loc_line = f"{county_hint}, Indiana"
        elif city_hint:
            loc_line = f"{city_hint}, Indiana"
        else:
            loc_line = "Indiana"

        # Timeline label
        timeline_label = _infer_timeline_label(date if date else None, snippet)

        # Scope summary & notes
        scope_summary = _build_scope_summary(snippet)
        notes = _short_snippet(snippet)

        # Project heading
        lines.append(f"{title} – {loc_line}")

        # Type
        lines.append(f"Type: {proj_type}")

        # Company / Developer
        if company_name:
            lines.append(f"Company / Developer: {company_name}")
        else:
            lines.append("Company / Developer: not clearly specified in the search result; check the source site for details.")

        # What this is (plain English)
        if what_this_is:
            lines.append(f"What this is: {what_this_is}")

        # Scope
        lines.append(f"Scope: {scope_summary}")

        # Timeline
        lines.append(f"Timeline: {timeline_label}")

        # Source (clean URL)
        if url:
            lines.append(f"Source: {url}")

        # Notes from snippet
        lines.append(f"Notes: {notes}")
        lines.append("")

    return "\n".join(lines).rstrip()


__all__ = [
    "search_indiana_developments",
    "render_developments_markdown",
]
