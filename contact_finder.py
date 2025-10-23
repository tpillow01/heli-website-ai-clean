# contact_finder.py
import csv, os, time, re
from typing import List, Dict, Tuple
from flask import Blueprint, request, jsonify, current_app
from difflib import SequenceMatcher
from functools import lru_cache

contact_finder_bp = Blueprint("contact_finder", __name__)

# ---- Config ----
CSV_PATH_ENV = "CONTACTS_CSV"  # set this in your env (absolute or relative path)
DEFAULT_PAGE_SIZE = 25
MAX_PAGE_SIZE = 100

# Expected CSV columns (case-insensitive match by header text)
REQUIRED_HEADERS = [
    "Company Name", "First Name", "Last Name", "Title",
    "Website", "Email 1", "Email 2", "Company Phone 1", "Company Phone 2",
    "Contact City", "Contact State", "Company Location"
]

# ---- In-memory index ----
_contacts: List[Dict] = []
_csv_mtime: float = -1.0
_csv_path: str = ""

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()

def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def _validate_headers(headers: List[str]) -> Tuple[bool, List[str]]:
    lh = [h.strip().lower() for h in headers]
    missing = [h for h in REQUIRED_HEADERS if h.lower() not in lh]
    return (len(missing) == 0, missing)

def _load_csv_if_changed(force=False):
    global _contacts, _csv_mtime, _csv_path
    path = os.environ.get(CSV_PATH_ENV, "").strip()
    if not path:
        raise RuntimeError(
            f"Environment variable {CSV_PATH_ENV} is not set. "
            "Point it to your contacts CSV file."
        )
    _csv_path = path

    try:
        mtime = os.path.getmtime(path)
    except FileNotFoundError:
        raise RuntimeError(f"Contacts CSV not found at: {path}")

    if force or mtime != _csv_mtime:
        rows = []
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter=",")
            ok, missing = _validate_headers(reader.fieldnames or [])
            if not ok:
                raise RuntimeError(
                    "Contacts CSV is missing required headers: "
                    + ", ".join(missing)
                )
            for r in reader:
                row = {k: (v or "").strip() for k, v in r.items()}
                # Precompute normalized fields
                row["_company_norm"] = _normalize(row.get("Company Name", ""))
                row["_person_norm"] = _normalize(
                    f"{row.get('First Name','')} {row.get('Last Name','')}"
                )
                rows.append(row)
        _contacts = rows
        _csv_mtime = mtime
        current_app.logger.info(
            f"[Contact Finder] Loaded {len(_contacts)} contacts from {path}"
        )

def _ensure_loaded():
    try:
        _load_csv_if_changed(force=False)
    except Exception as e:
        current_app.logger.error(f"[Contact Finder] Load error: {e}")
        raise

@lru_cache(maxsize=512)
def _search_company_cached(query_norm: str) -> List[int]:
    """
    Returns a list of indices into _contacts ranked by relevance to query_norm.
    Cache key is the normalized query.
    """
    # 1) Exact company matches (case-insensitive)
    exact_idx = [i for i, r in enumerate(_contacts) if r["_company_norm"] == query_norm]

    if exact_idx:
        return exact_idx

    # 2) Partial contains
    contains_idx = [i for i, r in enumerate(_contacts) if query_norm in r["_company_norm"]]

    # 3) Fuzzy fallback (top candidates)
    if not contains_idx:
        scored = []
        for i, r in enumerate(_contacts):
            score = _similar(query_norm, r["_company_norm"])
            if score >= 0.65:  # threshold; adjust as needed
                scored.append((score, i))
        scored.sort(reverse=True, key=lambda t: t[0])
        return [i for _, i in scored[:500]]  # cap size for speed

    return contains_idx

def _paginate(items: List[Dict], page: int, page_size: int) -> Tuple[List[Dict], int]:
    total = len(items)
    start = max(0, (page - 1) * page_size)
    end = min(total, start + page_size)
    return items[start:end], total

def _dedupe_by_person(records: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for r in records:
        key = (_normalize(r.get("First Name","")), _normalize(r.get("Last Name","")), _normalize(r.get("Title","")), _normalize(r.get("Email 1","")), _normalize(r.get("Email 2","")))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out

def _format_contact_line(r: Dict) -> str:
    """
    Pretty, multi-line "card" per contact:
    - Name — Title
      • Location: City, ST
      • Email: a@b.com, c@d.com
      • Phone: 555-..., 555-...
      • Website: example.com
    """
    first = (r.get("First Name", "") or "").strip()
    last  = (r.get("Last Name", "") or "").strip()
    name  = " ".join([x for x in [first, last] if x])

    title = (r.get("Title", "") or "").strip()
    city  = (r.get("Contact City", "") or "").strip()
    state = (r.get("Contact State", "") or "").strip()
    emails = ", ".join([e for e in [(r.get("Email 1") or "").strip(),
                                    (r.get("Email 2") or "").strip()] if e])
    phones = ", ".join([p for p in [(r.get("Company Phone 1") or "").strip(),
                                    (r.get("Company Phone 2") or "").strip()] if p])
    website = (r.get("Website", "") or "").strip()

    # Top line: Name — Title (title optional)
    top = f"- **{name}**" if name else "- **(Name not specified)**"
    if title:
        top += f" — {title}"

    lines = [top]

    # Subsequent lines only if present
    loc_bits = ", ".join([b for b in [city, state] if b])
    if loc_bits:
        lines.append(f"  • Location: {loc_bits}")
    if emails:
        lines.append(f"  • Email: {emails}")
    if phones:
        lines.append(f"  • Phone: {phones}")
    if website:
        lines.append(f"  • Website: {website}")

    return "\n".join(lines)

def _format_company_block(company: str, rows: List[Dict], page: int, page_size: int, total: int) -> str:
    header = f"**Contacts for:** {company}\n\n"
    # Blank line between contacts for readability
    body = "\n\n".join(_format_contact_line(r) for r in rows) if rows else "_No contacts found._"
    footer = ""
    if total > page_size:
        last_page = (total + page_size - 1) // page_size
        footer = f"\n\n_Page {page} of {last_page} • {total} matches_"
    return header + body + footer

def _extract_company_query(raw: str) -> str:
    """
    Try to pull a company name out of natural language like:
    "give me contacts for FCI Construction"
    Fallback: use entire raw text.
    """
    m = re.search(r"(?:for|at|from)\s+(.+)$", raw, re.IGNORECASE)
    q = m.group(1).strip() if m else raw.strip()
    # strip common leading filler
    q = re.sub(r"^(contacts?\s+for|find\s+contacts\s+for)\s+", "", q, flags=re.IGNORECASE)
    return q

# ---------- Routes ----------

@contact_finder_bp.route("/api/contacts/search", methods=["POST"])
def contacts_search():
    """
    JSON in: { "q": "FCI Construction", "page": 1, "page_size": 25 }
    JSON out: { "query": "...", "total": N, "page": 1, "page_size": 25, "results": [ ... ] }
    """
    _ensure_loaded()

    data = request.get_json(silent=True) or {}
    raw_q = data.get("q", "")
    if not raw_q:
        return jsonify({"error": "Missing 'q'"}), 400

    page = int(data.get("page", 1) or 1)
    page_size = int(data.get("page_size", DEFAULT_PAGE_SIZE) or DEFAULT_PAGE_SIZE)
    page_size = max(1, min(page_size, MAX_PAGE_SIZE))

    query = _extract_company_query(raw_q)
    qn = _normalize(query)
    if not qn:
        return jsonify({"error": "Empty query after parsing"}), 400

    # hot reload if CSV file changed
    _load_csv_if_changed(force=False)
    # clear cache if reloaded (mtime changed) — safe because cache key is only query string
    _search_company_cached.cache_clear()

    idx_list = _search_company_cached(qn)
    records = [_contacts[i] for i in idx_list]
    # Keep only rows whose company name actually includes the query words (loose filter)
    # This avoids fuzzy false positives crowding page 1.
    words = [w for w in qn.split() if w]
    if words:
        records = [r for r in records if all(w in r["_company_norm"] for w in words)] or records

    # Dedupe and sort by company then last name
    records = _dedupe_by_person(records)
    company_display = records[0].get("Company Name") if records else query
    records.sort(key=lambda r: (r.get("Company Name","").lower(), r.get("Last Name","").lower(), r.get("First Name","").lower()))

    page_rows, total = _paginate(records, page, page_size)

    return jsonify({
        "query": query,
        "company_display": company_display,
        "total": total,
        "page": page,
        "page_size": page_size,
        "results": [
            {
                "company": r.get("Company Name",""),
                "first_name": r.get("First Name",""),
                "last_name": r.get("Last Name",""),
                "title": r.get("Title",""),
                "website": r.get("Website",""),
                "email_1": r.get("Email 1",""),
                "email_2": r.get("Email 2",""),
                "company_phone_1": r.get("Company Phone 1",""),
                "company_phone_2": r.get("Company Phone 2",""),
                "contact_city": r.get("Contact City",""),
                "contact_state": r.get("Contact State",""),
                "company_location": r.get("Company Location","")
            } for r in page_rows
        ]
    })

@contact_finder_bp.route("/api/chat_contact_finder", methods=["POST"])
def chat_contact_finder():
    """
    Chat-mode wrapper: returns a friendly markdown block you can paste into your chat UI.
    JSON in: { "message": "give me contacts for FCI Construction", "page": 1, "page_size": 25 }
    JSON out: { "text": "**Contacts for:** ...\n- Name — Title • City, ST • Email • Phone" }
    """
    _ensure_loaded()

    data = request.get_json(silent=True) or {}
    raw = data.get("message", "")
    page = int(data.get("page", 1) or 1)
    page_size = int(data.get("page_size", DEFAULT_PAGE_SIZE) or DEFAULT_PAGE_SIZE)
    page_size = max(1, min(page_size, MAX_PAGE_SIZE))

    if not raw.strip():
        return jsonify({"text": "_Please type a company name, e.g., ‘contacts for Acme Corp’._"}), 400

    query = _extract_company_query(raw)
    qn = _normalize(query)
    if not qn:
        return jsonify({"text": "_I couldn’t parse a company from that. Try ‘contacts for Acme Corp’._"}), 400

    _load_csv_if_changed(force=False)
    _search_company_cached.cache_clear()

    idx_list = _search_company_cached(qn)
    records = [_contacts[i] for i in idx_list]
    words = [w for w in qn.split() if w]
    if words:
        records = [r for r in records if all(w in r["_company_norm"] for w in words)] or records

    records = _dedupe_by_person(records)
    company_display = records[0].get("Company Name") if records else query
    records.sort(key=lambda r: (r.get("Company Name","").lower(), r.get("Last Name","").lower(), r.get("First Name","").lower()))

    page_rows, total = _paginate(records, page, page_size)
    text = _format_company_block(company_display, page_rows, page, page_size, total)
    return jsonify({"text": text})
