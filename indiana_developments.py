"""
indiana_developments.py

Lightweight web scraper for Indiana industrial / warehouse / logistics developments,
used by the `indiana_developments` chat mode in heli_backup_ai.py.
"""

from __future__ import annotations

import re
import urllib.parse
from datetime import datetime, timedelta
from typing import Any, Dict, List

import requests
from bs4 import BeautifulSoup

# Basic user-agent so news sites don't instantly reject us
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; TynanIndianaIntelBot/1.0; +https://tynanequipment.com)"
}


def _extract_counties_and_cities(text: str) -> List[str]:
    """
    Very simple extractor:
    - Looks for 'Boone County', 'Hendricks County', etc.
    - Also picks up explicit city names if you include them in the question.
    """
    text = text or ""
    hits: List[str] = []

    # County pattern: 'Something County'
    for m in re.findall(r"\b([A-Z][a-z]+ County)\b", text):
        hits.append(m.strip())

    # A few common Indiana city names you might mention often.
    # You can add more over time if you want.
    CITY_CANDIDATES = [
        "Indianapolis",
        "Lebanon",
        "Whitestown",
        "Plainfield",
        "Avon",
        "Brownsburg",
        "Zionsville",
        "Greenwood",
        "Columbus",
        "Terre Haute",
        "Fort Wayne",
        "Evansville",
    ]
    lower = text.lower()
    for city in CITY_CANDIDATES:
        if city.lower() in lower:
            hits.append(city)

    # Deduplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for h in hits:
        if h.lower() not in seen:
            seen.add(h.lower())
            uniq.append(h)
    return uniq


def _fetch_google_news(query: str) -> List[Dict[str, Any]]:
    """
    Pull articles from Google News RSS for the given query.
    Returns a list of dicts with title, link, published date, and source.
    """
    encoded_q = urllib.parse.quote(query)
    url = (
        "https://news.google.com/rss/search?q="
        f"{encoded_q}&hl=en-US&gl=US&ceid=US:en"
    )

    resp = requests.get(url, headers=_HEADERS, timeout=10)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "xml")  # works fine for RSS
    items: List[Dict[str, Any]] = []

    for item in soup.find_all("item"):
        title_tag = item.find("title")
        link_tag = item.find("link")
        date_tag = item.find("pubDate")
        source_tag = item.find("source")

        title = (title_tag.get_text(strip=True) if title_tag else "").replace(" - Google News", "")
        link = link_tag.get_text(strip=True) if link_tag else ""
        date_str = date_tag.get_text(strip=True) if date_tag else ""
        source = source_tag.get_text(strip=True) if source_tag else ""

        # Parse pubDate if we can
        published: datetime | None = None
        if date_str:
            try:
                # Example: Tue, 19 Nov 2024 10:30:00 GMT
                published = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %Z")
            except Exception:
                published = None

        items.append(
            {
                "title": title,
                "link": link,
                "published": published,
                "date_str": date_str,
                "source": source,
            }
        )

    return items


def search_indiana_developments(user_q: str, days: int = 730) -> List[Dict[str, Any]]:
    """
    Main entry used by /api/chat.

    - Builds one or more Google News queries based on the user's text.
    - Pulls recent industrial / warehouse / logistics development news in Indiana.
    - Filters by date window (last `days` days).
    """
    base_terms = "Indiana warehouse OR distribution center OR logistics park OR industrial park"
    locs = _extract_counties_and_cities(user_q)

    queries: List[str] = []
    if locs:
        for loc in locs:
            # e.g. "Boone County Indiana warehouse OR distribution center ..."
            queries.append(f"{loc} Indiana {base_terms}")
    else:
        # Fallback: statewide search
        queries.append(base_terms)

    cutoff = datetime.utcnow() - timedelta(days=days)
    all_items: List[Dict[str, Any]] = []
    seen_links = set()

    for q in queries:
        try:
            feed_items = _fetch_google_news(q)
        except Exception:
            # If one query fails, just skip it
            continue

        for it in feed_items:
            link = it.get("link") or ""
            if not link or link in seen_links:
                continue

            published = it.get("published")
            # If we don't have a parsed date, keep it but it'll be treated as "unknown"
            if isinstance(published, datetime) and published < cutoff:
                continue

            seen_links.add(link)
            # Attach the query/loc info as a weak 'location' hint
            it["location_hint"] = ", ".join(locs) if locs else "Indiana"
            all_items.append(it)

    # Sort newest first when we have dates, otherwise just keep original order
    all_items.sort(
        key=lambda x: x["published"] or datetime.utcnow(), reverse=True
    )

    # Cap to something reasonable; the LLM doesn't need 100 articles
    return all_items[:25]


def render_developments_markdown(items: List[Dict[str, Any]]) -> str:
    """
    Turn the scraped items into a markdown block that your /api/chat mode
    feeds into the OpenAI model as 'intel'.
    """
    if not items:
        return (
            "No recent Indiana industrial / warehouse / logistics developments were "
            "found in the requested timeframe."
        )

    lines: List[str] = []
    lines.append("**Recent Indiana industrial / warehouse developments:**")

    for it in items:
        title = it.get("title", "").strip()
        link = it.get("link", "").strip()
        date_str = it.get("date_str", "").strip()
        source = it.get("source", "").strip()
        loc_hint = it.get("location_hint", "").strip()

        bullet = "- "
        if date_str:
            bullet += f"{date_str} — "
        bullet += title if title else "(no title)"

        meta_bits = []
        if source:
            meta_bits.append(source)
        if loc_hint:
            meta_bits.append(loc_hint)
        if meta_bits:
            bullet += " (" + ", ".join(meta_bits) + ")"

        lines.append(bullet)
        if link:
            lines.append(f"  {link}")

    return "\n".join(lines)
