# contact_finder.py
import csv
import os
import re
from typing import List, Dict, Tuple
from flask import Blueprint, request, jsonify, current_app
from difflib import SequenceMatcher
from functools import lru_cache

contact_finder_bp = Blueprint("contact_finder", __name__)

# ---- Config ----
CSV_PATH_ENV = "CONTACTS_CSV"  # set this in your env (absolute or relative path)
DEFAULT_PAGE_SIZE = 25
MAX_PAGE_SIZE = 100

# Only truly required columns should be enforced
REQUIRED_HEADERS = [
    "Company Name",
    "First Name",
    "Last Name",
]

# Optional columns should not crash the app if missing
OPTIONAL_HEADERS = [
    "Title",
    "Website",
    "Email 1",
    "Email 2",
    "Company Phone 1",
    "Company Phone 2",
    "Contact City",
    "Contact State",
    "Company Location",
    "Postal Code",
    "Country",
]

# Common alternate header names mapped to canonical names
HEADER_ALIASES = {
    "company": "Company Name",
    "company name": "Company Name",
    "account name": "Company Name",
    "account": "Company Name",
    "customer": "Company Name",
    "customer name": "Company Name",
    "business": "Company Name",
    "business name": "Company Name",

    "first": "First Name",
    "first name": "First Name",
    "firstname": "First Name",

    "last": "Last Name",
    "last name": "Last Name",
    "lastname": "Last Name",

    "name": "Full Name",
    "contact name": "Full Name",
    "full name": "Full Name",

    "job title": "Title",
    "title": "Title",

    "website": "Website",
    "web site": "Website",
    "site": "Website",
    "url": "Website",

    "email": "Email 1",
    "primary email": "Email 1",
    "email 1": "Email 1",
    "email1": "Email 1",
    "secondary email": "Email 2",
    "email 2": "Email 2",
    "email2": "Email 2",

    "phone": "Company Phone 1",
    "phone 1": "Company Phone 1",
    "company phone": "Company Phone 1",
    "company phone 1": "Company Phone 1",
    "main phone": "Company Phone 1",

    "mobile": "Company Phone 2",
    "phone 2": "Company Phone 2",
    "company phone 2": "Company Phone 2",
    "secondary phone": "Company Phone 2",

    "city": "Contact City",
    "contact city": "Contact City",
    "mailing city": "Contact City",

    "state": "Contact State",
    "contact state": "Contact State",
    "mailing state": "Contact State",
    "mailing state/province": "Contact State",

    "location": "Company Location",
    "company location": "Company Location",
    "mailing street": "Company Location",

    "mailing zip/postal code": "Postal Code",
    "zip": "Postal Code",
    "postal code": "Postal Code",

    "mailing country": "Country",
    "country": "Country",
}

# ---- In-memory index ----
_contacts: List[Dict] = []
_csv_mtime: float = -1.0
_csv_path: str = ""


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _canonicalize_header(header: str) -> str:
    raw = (header or "").strip()
    key = raw.lower()
    return HEADER_ALIASES.get(key, raw)


def _split_full_name(full_name: str) -> Tuple[str, str]:
    parts = [p for p in (full_name or "").strip().split() if p]
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], " ".join(parts[1:])


def _prepare_headers(headers: List[str]) -> Tuple[List[str], List[str]]:
    canonical_headers = [_canonicalize_header(h) for h in headers]
    lower_headers = [h.strip().lower() for h in canonical_headers]
    missing_required = [h for h in REQUIRED_HEADERS if h.lower() not in lower_headers]
    return canonical_headers, missing_required


def _normalize_company_name(company_name: str) -> str:
    company_name = (company_name or "").strip()
    # Strip leading prefixes like "(Maxon Corp)Honeywell International Inc."
    company_name = re.sub(r"^\([^)]*\)", "", company_name).strip()
    # Normalize repeated whitespace
    company_name = re.sub(r"\s+", " ", company_name)
    return company_name


def _normalize_row(raw_row: Dict[str, str]) -> Dict[str, str]:
    row = {
        _canonicalize_header(k): (v or "").strip()
        for k, v in raw_row.items()
        if k is not None
    }

    for col in OPTIONAL_HEADERS:
        row.setdefault(col, "")

    for col in REQUIRED_HEADERS:
        row.setdefault(col, "")

    # Support CSVs with a single name column
    if (not row.get("First Name") and not row.get("Last Name")) and row.get("Full Name"):
        first, last = _split_full_name(row.get("Full Name", ""))
        row["First Name"] = first
        row["Last Name"] = last

    row["Company Name"] = _normalize_company_name(row.get("Company Name", ""))

    row["_company_norm"] = _normalize(row.get("Company Name", ""))
    row["_person_norm"] = _normalize(
        f"{row.get('First Name', '')} {row.get('Last Name', '')}"
    )

    return row


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

            raw_headers = reader.fieldnames or []
            _, missing_required = _prepare_headers(raw_headers)

            if missing_required:
                raise RuntimeError(
                    "Contacts CSV is missing required headers: "
                    + ", ".join(missing_required)
                )

            for raw_row in reader:
                row = _normalize_row(raw_row)

                if not row.get("Company Name") and not row.get("First Name") and not row.get("Last Name"):
                    continue

                rows.append(row)

        _contacts = rows
        _csv_mtime = mtime
        _search_company_cached.cache_clear()

        sample_companies = [r.get("Company Name", "") for r in rows[:10]]
        nonblank_company_count = sum(1 for r in rows if (r.get("Company Name") or "").strip())

        current_app.logger.info(
            f"[Contact Finder] Loaded {len(_contacts)} contacts from {path}"
        )
        current_app.logger.info(
            f"[Contact Finder] Nonblank Company Name count: {nonblank_company_count}"
        )
        current_app.logger.info(
            f"[Contact Finder] Sample Company Name values: {sample_companies}"
        )


def _ensure_loaded():
    try:
        _load_csv_if_changed(force=False)
    except Exception as e:
        current_app.logger.error(f"[Contact Finder] Load error: {e}")
        raise


@lru_cache(maxsize=512)
def _search_company_cached(query_norm: str) -> List[int]:
    exact_idx = [i for i, r in enumerate(_contacts) if r["_company_norm"] == query_norm]
    if exact_idx:
        return exact_idx

    contains_idx = [i for i, r in enumerate(_contacts) if query_norm in r["_company_norm"]]

    if not contains_idx:
        scored = []
        for i, r in enumerate(_contacts):
            score = _similar(query_norm, r["_company_norm"])
            if score >= 0.60:
                scored.append((score, i))
        scored.sort(reverse=True, key=lambda t: t[0])
        return [i for _, i in scored[:500]]

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
        key = (
            _normalize(r.get("Company Name", "")),
            _normalize(r.get("First Name", "")),
            _normalize(r.get("Last Name", "")),
            _normalize(r.get("Title", "")),
            _normalize(r.get("Email 1", "")),
            _normalize(r.get("Email 2", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(r)

    return out


def _format_contact_line(r: Dict) -> str:
    first = (r.get("First Name", "") or "").strip()
    last = (r.get("Last Name", "") or "").strip()
    name = " ".join([x for x in [first, last] if x])

    title = (r.get("Title", "") or "").strip()
    city = (r.get("Contact City", "") or "").strip()
    state = (r.get("Contact State", "") or "").strip()
    postal_code = (r.get("Postal Code", "") or "").strip()
    country = (r.get("Country", "") or "").strip()

    emails = ", ".join([
        e for e in [
            (r.get("Email 1") or "").strip(),
            (r.get("Email 2") or "").strip(),
        ] if e
    ])
    phones = ", ".join([
        p for p in [
            (r.get("Company Phone 1") or "").strip(),
            (r.get("Company Phone 2") or "").strip(),
        ] if p
    ])
    website = (r.get("Website", "") or "").strip()
    location = (r.get("Company Location", "") or "").strip()

    top = f"- **{name}**" if name else "- **(Name not specified)**"
    if title:
        top += f" — {title}"

    lines = [top]

    loc_bits = ", ".join([b for b in [city, state] if b])
    if loc_bits:
        lines.append(f"  • Location: {loc_bits}")
    elif location:
        lines.append(f"  • Address: {location}")

    address_bits = ", ".join([b for b in [location, postal_code, country] if b])
    if address_bits and not loc_bits:
        lines.append(f"  • Address: {address_bits}")

    if emails:
        lines.append(f"  • Email: {emails}")
    if phones:
        lines.append(f"  • Phone: {phones}")
    if website:
        lines.append(f"  • Website: {website}")

    return "\n".join(lines)


def _format_company_block(company: str, rows: List[Dict], page: int, page_size: int, total: int) -> str:
    header = f"**Contacts for:** {company}\n\n"
    body = "\n\n".join(_format_contact_line(r) for r in rows) if rows else "_No contacts found._"
    footer = ""
    if total > page_size:
        last_page = (total + page_size - 1) // page_size
        footer = f"\n\n_Page {page} of {last_page} • {total} matches_"
    return header + body + footer


def _extract_company_query(raw: str) -> str:
    raw = (raw or "").strip()

    m = re.search(r"(?:for|at|from)\s+(.+)$", raw, re.IGNORECASE)
    q = m.group(1).strip() if m else raw

    q = re.sub(r"^(contacts?\s+for|find\s+contacts\s+for)\s+", "", q, flags=re.IGNORECASE)
    q = re.sub(r"^(find me\s+contacts\s+for)\s+", "", q, flags=re.IGNORECASE)

    generic_phrases = {
        "i need contacts",
        "need contacts",
        "contacts",
        "find contacts",
        "find me contacts",
    }

    if q.lower() in generic_phrases:
        return ""

    return q


@contact_finder_bp.route("/api/contacts/search", methods=["POST"])
def contacts_search():
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

    _load_csv_if_changed(force=False)

    idx_list = _search_company_cached(qn)
    records = [_contacts[i] for i in idx_list]

    words = [w for w in qn.split() if w]
    if words:
        records = [r for r in records if all(w in r["_company_norm"] for w in words)] or records

    records = _dedupe_by_person(records)
    company_display = records[0].get("Company Name") if records else query
    records.sort(
        key=lambda r: (
            r.get("Company Name", "").lower(),
            r.get("Last Name", "").lower(),
            r.get("First Name", "").lower(),
        )
    )

    page_rows, total = _paginate(records, page, page_size)

    return jsonify({
        "query": query,
        "company_display": company_display,
        "total": total,
        "page": page,
        "page_size": page_size,
        "results": [
            {
                "company": r.get("Company Name", ""),
                "first_name": r.get("First Name", ""),
                "last_name": r.get("Last Name", ""),
                "title": r.get("Title", ""),
                "website": r.get("Website", ""),
                "email_1": r.get("Email 1", ""),
                "email_2": r.get("Email 2", ""),
                "company_phone_1": r.get("Company Phone 1", ""),
                "company_phone_2": r.get("Company Phone 2", ""),
                "contact_city": r.get("Contact City", ""),
                "contact_state": r.get("Contact State", ""),
                "company_location": r.get("Company Location", ""),
                "postal_code": r.get("Postal Code", ""),
                "country": r.get("Country", ""),
            }
            for r in page_rows
        ],
    })


@contact_finder_bp.route("/api/chat_contact_finder", methods=["POST"])
def chat_contact_finder():
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
        return jsonify({"text": "_Please give me a company name, for example: contacts for Honeywell._"}), 400

    _load_csv_if_changed(force=False)

    idx_list = _search_company_cached(qn)
    records = [_contacts[i] for i in idx_list]

    current_app.logger.info(
        f"[Contact Finder] Query='{query}' normalized='{qn}' raw_matches={len(records)}"
    )
    if records[:5]:
        current_app.logger.info(
            f"[Contact Finder] Top raw company matches: {[r.get('Company Name', '') for r in records[:5]]}"
        )

    words = [w for w in qn.split() if w]
    if words:
        records = [r for r in records if all(w in r["_company_norm"] for w in words)] or records

    current_app.logger.info(
        f"[Contact Finder] Filtered matches after word check: {len(records)}"
    )

    records = _dedupe_by_person(records)
    company_display = records[0].get("Company Name") if records else query
    records.sort(
        key=lambda r: (
            r.get("Company Name", "").lower(),
            r.get("Last Name", "").lower(),
            r.get("First Name", "").lower(),
        )
    )

    page_rows, total = _paginate(records, page, page_size)
    text = _format_company_block(company_display, page_rows, page, page_size, total)

    return jsonify({"text": text})