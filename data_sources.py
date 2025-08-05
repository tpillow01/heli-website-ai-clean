# data_sources.py
"""
Inquiry-mode data utilities.

What this module does:
- Robustly load customer_report.csv and customer_billing.csv (handles encodings).
- Normalize and match customer names across both files (exact → loose → ID).
- Aggregate report R12 (with header-variant detection) and billing patterns.
- Use the segment printed in the report (e.g., 'C2') when present.
- Provide recent-invoice lists for the UI/AI when the rep asks.
"""

from __future__ import annotations

import os
import re
import unicodedata
import difflib
from functools import lru_cache
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import pandas as pd

# -------------------------------------------------------------------------
# File paths (override via env if needed)
CUSTOMER_REPORT_CSV  = os.getenv("CUSTOMER_REPORT_CSV",  "customer_report.csv")
CUSTOMER_BILLING_CSV = os.getenv("CUSTOMER_BILLING_CSV", "customer_billing.csv")

# -------------------------------------------------------------------------
# Text normalization (names)
def _norm_name(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    s = re.sub(r"[^\w\s&]", "", s)
    s = s.replace("&", " and ")
    s = re.sub(r"\s+", " ", s).strip()
    for suf in (" inc", " llc", " co", " corp", " corporation", " company", " ltd", " limited"):
        if s.endswith(suf):
            s = s[: -len(suf)].strip()
    return s

# -------------------------------------------------------------------------
# Robust CSV reader (keeps text; strips header whitespace)
def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    attempts = [
        {"encoding": "utf-8",     "encoding_errors": "strict"},
        {"encoding": "utf-8-sig", "encoding_errors": "replace"},
        {"encoding": "cp1252",    "encoding_errors": "replace"},
        {"encoding": "latin1",    "encoding_errors": "replace"},
    ]
    last_err = None
    for opts in attempts:
        try:
            df = pd.read_csv(
                path,
                low_memory=False,
                dtype=str,
                keep_default_na=False,
                **opts
            ).fillna("")
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except Exception as e:
            last_err = e
    # final best-effort
    df = pd.read_csv(path, low_memory=False, dtype=str, keep_default_na=False).fillna("")
    df.columns = [str(c).strip() for c in df.columns]
    return df

# -------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_customer_report() -> List[Dict]:
    return _read_csv(CUSTOMER_REPORT_CSV).to_dict(orient="records")

@lru_cache(maxsize=1)
def load_customer_billing() -> List[Dict]:
    return _read_csv(CUSTOMER_BILLING_CSV).to_dict(orient="records")

# Debug header helpers
def get_report_headers() -> List[str]:
    return [str(c) for c in _read_csv(CUSTOMER_REPORT_CSV).columns]

def get_billing_headers() -> List[str]:
    return [str(c) for c in _read_csv(CUSTOMER_BILLING_CSV).columns]

# -------------------------------------------------------------------------
# Helpers to pull ID / Name from rows
def _pick_id(row: Dict, fallback_index: int) -> str:
    for key in ("Ship to ID", "Sold to ID", "Ship to ID ", "Sold To BP", "Ship to code"):
        val = str(row.get(key, "")).strip()
        if val:
            return val
    return str(fallback_index)

def _pick_name(row: Dict) -> str:
    for key in ("Ship to Name", "Sold to Name", "CUSTOMER", "Customer", "account_name", "company"):
        val = str(row.get(key, "")).strip()
        if val:
            return val
    for v in row.values():
        if str(v).strip():
            return str(v).strip()
    return "Unnamed"

# For (optional) UI target lists
def make_inquiry_targets() -> List[Dict]:
    targets: Dict[str, Dict[str, str]] = {}
    combined = load_customer_report() + load_customer_billing()
    for idx, r in enumerate(combined):
        cid = _pick_id(r, idx)
        name = _pick_name(r)
        if cid not in targets:
            targets[cid] = {"id": cid, "label": name}
        elif targets[cid]["label"] == "Unnamed" and name != "Unnamed":
            targets[cid]["label"] = name
    return sorted(targets.values(), key=lambda x: _norm_name(x["label"]))

def find_customer_id_by_name(name: str) -> Optional[str]:
    if not name:
        return None
    want = _norm_name(name)
    labels = make_inquiry_targets()
    for item in labels:
        if _norm_name(item["label"]) == want:
            return item["id"]
    for item in labels:
        if want and want in _norm_name(item["label"]):
            return item["id"]
    options = [it["label"] for it in labels]
    match = difflib.get_close_matches(name, options, n=1, cutoff=0.6)
    if match:
        lbl = match[0]
        for it in labels:
            if it["label"] == lbl:
                return it["id"]
    return None

# -------------------------------------------------------------------------
# Name-first row lookup across both files; exact → loose → ID fallback
def find_inquiry_rows_flexible(customer_id: Optional[str] = None,
                               customer_name: Optional[str] = None) -> Dict[str, List[Dict]]:
    want = _norm_name(customer_name or "")
    id_val = (customer_id or "").strip()

    rep_all = load_customer_report()
    bil_all = load_customer_billing()

    def norm_of(row: Dict) -> str:
        for k in ("CUSTOMER", "Customer", "Ship to Name", "Sold to Name"):
            v = row.get(k, "")
            if v:
                return _norm_name(v)
        return ""

    report_hits: List[Dict] = []
    billing_hits: List[Dict] = []

    # 1) exact normalized name
    if want:
        report_hits = [r for r in rep_all if norm_of(r) == want]
        billing_hits = [r for r in bil_all if norm_of(r) == want]

    # 2) loose if nothing found: substring / token-subset
    if not report_hits and not billing_hits and want:
        def loose_match(row: Dict) -> bool:
            cand = norm_of(row)
            if not cand:
                return False
            if cand == want or want in cand or cand in want:
                return True
            wt = {t for t in want.split() if len(t) > 1}
            ct = {t for t in cand.split() if len(t) > 1}
            return wt.issubset(ct)
        report_hits = [r for r in rep_all if loose_match(r)]
        billing_hits = [r for r in bil_all if loose_match(r)]

    # 3) id fallback
    if not report_hits and not billing_hits and id_val:
        for idx, r in enumerate(rep_all):
            if _pick_id(r, idx) == id_val:
                report_hits.append(r)
        for idx, r in enumerate(bil_all):
            if _pick_id(r, idx) == id_val:
                billing_hits.append(r)

    return {"report": report_hits, "billing": billing_hits}

# -------------------------------------------------------------------------
# Parsing helpers
def _to_float(x) -> float:
    if x is None:
        return 0.0
    s = str(x).strip()
    if not s or s == "-":
        return 0.0
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()").replace(",", "").replace("$", "")
    try:
        val = float(s)
    except Exception:
        return 0.0
    return -val if neg else val

def _to_date(x) -> Optional[datetime]:
    if not x:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%Y/%m/%d"):
        try:
            return datetime.strptime(str(x).split(" ")[0], fmt)
        except Exception:
            continue
    return None

# -------------------------------------------------------------------------
# Department normalization
def normalize_department(raw: str) -> str:
    r = str(raw).strip().lower()
    if "service" in r:
        return "SERVICE"
    if "part" in r:
        return "PARTS"
    if "rental" in r:
        return "RENTAL"
    if "new" in r:
        return "NEW_EQUIP"
    if "used" in r or "preowned" in r:
        return "USED_EQUIP"
    return "OTHER"

OFFERING_KEYS = ["SERVICE", "PARTS", "RENTAL", "NEW_EQUIP", "USED_EQUIP"]
NICE = {
    "SERVICE": "Service",
    "PARTS": "Parts",
    "RENTAL": "Rental",
    "NEW_EQUIP": "New Equipment",
    "USED_EQUIP": "Used Equipment",
}

# -------------------------------------------------------------------------
# Aggregations
def _aggregate_report_r12(report_rows: List[Dict]) -> Dict[str, float]:
    """
    Build an R12 view from the customer_report.csv:
    - Prefer a precomputed Aftermarket R12 total if present (common variants).
    - Otherwise sum Parts/Service/Rental R12 (common header variants).
    - Keep Equipment R36 as context only (not counted in R12 total).
    """
    def pick(row: Dict, candidates: List[str]) -> float:
        for c in candidates:
            if c in row:
                return _to_float(row.get(c))
        lowmap = {str(k).strip().lower(): k for k in row.keys()}
        for c in candidates:
            key = lowmap.get(c.strip().lower())
            if key:
                return _to_float(row.get(key))
        return 0.0

    buckets = {"SERVICE": 0.0, "PARTS": 0.0, "RENTAL": 0.0, "NEW_EQUIP": 0.0, "USED_EQUIP": 0.0}
    total_r12_aftermarket = 0.0
    saw_aftermarket_total = False

    AM_R12 = [
        "Revenue Rolling 12 Months - Aftermarket",
        "R12 Aftermarket Revenue",
        "Aftermarket R12 Revenue",
        "R12 Aftermarket",
        "Aftermarket Rolling 12",
    ]
    PARTS_R12   = ["Parts Revenue R12", "R12 Parts Revenue"]
    SERVICE_R12 = ["Service Revenue R12 (Includes GM)", "Service Revenue R12", "R12 Service Revenue"]
    RENTAL_R12  = ["Rental Revenue R12", "R12 Rental Revenue"]

    NEW_R36 = ["New Equip R36 Revenue", "R36 New Equipment Revenue", "New Equipment R36"]
    USED_R36 = ["Used Equip R36 Revenue", "R36 Used Equipment Revenue", "Used Equipment R36"]

    for r in report_rows:
        am_total = pick(r, AM_R12)
        if am_total:
            saw_aftermarket_total = True
            total_r12_aftermarket += am_total

        buckets["PARTS"]   += pick(r, PARTS_R12)
        buckets["SERVICE"] += pick(r, SERVICE_R12)
        buckets["RENTAL"]  += pick(r, RENTAL_R12)

        buckets["NEW_EQUIP"]  += pick(r, NEW_R36)
        buckets["USED_EQUIP"] += pick(r, USED_R36)

    total_r12 = total_r12_aftermarket if saw_aftermarket_total else (
        buckets["PARTS"] + buckets["SERVICE"] + buckets["RENTAL"]
    )

    return {"total_r12": total_r12, **buckets}

def _aggregate_billing(billing_rows: List[Dict]) -> Dict:
    by_dept: Dict[str, float] = defaultdict(float)
    by_month: Dict[str, float] = defaultdict(float)
    dates: List[datetime] = []
    invoices = 0
    last_invoice: Optional[datetime] = None

    now = datetime.utcnow()
    last_365_total = 0.0

    for r in billing_rows:
        dep = normalize_department(r.get("Type") or r.get("Department"))
        amt = _to_float(r.get("REVENUE"))
        by_dept[dep] += amt

        d = _to_date(r.get("Date") or r.get("Doc. Date"))
        if d:
            dates.append(d)
            by_month[f"{d.year:04d}-{d.month:02d}"] += amt
            if (last_invoice is None) or (d > last_invoice):
                last_invoice = d
            if (now - d).days <= 365:
                last_365_total += amt
        invoices += 1

    avg_days = None
    if len(dates) >= 2:
        ds = sorted(dates)
        deltas = [(ds[i] - ds[i-1]).days for i in range(1, len(ds))]
        if deltas:
            avg_days = sum(deltas) / len(deltas)

    offerings_count = sum(1 for k in OFFERING_KEYS if by_dept.get(k, 0.0) > 0)
    top_months = sorted(by_month.items(), key=lambda kv: kv[1], reverse=True)[:3]
    top_offerings = sorted([(k, v) for k, v in by_dept.items() if k in OFFERING_KEYS],
                           key=lambda kv: kv[1], reverse=True)[:3]

    return {
        "by_dept": dict(by_dept),
        "by_month": dict(by_month),
        "last_invoice": last_invoice.isoformat() if last_invoice else "",
        "invoice_count": invoices,
        "avg_days_between_invoices": avg_days,
        "offerings_count": offerings_count,
        "top_months": top_months,
        "top_offerings": top_offerings,
        "total_last_365": last_365_total,
    }

# -------------------------------------------------------------------------
# Segmentation & progression
def classify_account_size(total_r12: float) -> str:
    if total_r12 >= 200_000: return "A"
    if total_r12 >= 80_000:  return "B"
    if total_r12 >= 10_000:  return "C"
    return "D"

def classify_relationship(offerings_count: int, had_revenue: bool) -> str:
    # Legend: 1 = 4+, 2 = 3, 3 = 1–2, P = No Revenue
    if not had_revenue:
        return "P"
    if offerings_count >= 4:
        return "1"
    if offerings_count == 3:
        return "2"
    if offerings_count == 2:
        return "2"
    return "3"

def next_relationship(current: str) -> Optional[str]:
    return {"P": "3", "3": "2", "2": "1"}.get(current)

def relationship_requirements_for_next(current: str, offerings_count: int) -> Tuple[str, int]:
    nxt = next_relationship(current)
    if not nxt:
        return ("1", 0)
    if current == "P":
        return ("3", 1)
    if current == "3":
        need = max(0, 3 - offerings_count)
        return ("2", max(1, need))
    if current == "2":
        need = max(0, 4 - offerings_count)
        return ("1", max(1, need))
    return (nxt, 0)

def next_size_letter(current: str) -> Optional[str]:
    # Better sizes move toward the left on the grid: D → C → B → A
    order = ["D", "C", "B", "A"]
    try:
        i = order.index(current)
    except ValueError:
        return None
    return order[i + 1] if i + 1 < len(order) else None

def size_target_revenue(letter: str) -> Optional[int]:
    return {"D": 10_000, "C": 80_000, "B": 200_000}.get(letter)

# -------------------------------------------------------------------------
# Visit focus & next-level actions
def _recommend_focus(by_dept: Dict[str, float], last_invoice_iso: str) -> List[str]:
    recs: List[str] = []
    ranked = sorted([(k, v) for k, v in by_dept.items() if k in OFFERING_KEYS],
                    key=lambda kv: kv[1], reverse=True)
    if ranked:
        leader = ranked[0][0]
        if leader == "RENTAL":
            recs.append("Open a rent-to-own discussion; compare TCO vs purchase with service plan.")
        if leader == "SERVICE":
            recs.append("Propose a Preventive Maintenance agreement and bundle Parts kits.")
        if leader == "PARTS":
            recs.append("Offer Parts bundles and add a Service PM to anchor frequency.")
        if leader in ("NEW_EQUIP", "USED_EQUIP"):
            recs.append("Review fleet age/usage; present replacement options with financing & service.")
    gaps = [k for k in ["SERVICE", "PARTS", "RENTAL", "NEW_EQUIP"] if by_dept.get(k, 0.0) == 0]
    if gaps:
        recs.append("Add another offering this quarter: " + ", ".join(NICE[g] for g in gaps[:2]) + ".")
    if not last_invoice_iso:
        recs.append("No recent invoices — schedule a discovery visit and quick load survey.")
    return recs

def _actions_next_level(size_letter: str, rel_code: str, offerings_count: int, by_dept: Dict[str, float]) -> List[str]:
    actions: List[str] = []
    nxt_rel, addl_needed = relationship_requirements_for_next(rel_code, offerings_count)
    if nxt_rel and nxt_rel != rel_code:
        present = {k for k in OFFERING_KEYS if by_dept.get(k, 0.0) > 0}
        missing = [k for k in OFFERING_KEYS if k not in present]
        if rel_code == "P":
            actions.append("Create first revenue: book a PM Service or short-term Rental (fastest path to tier 3).")
        elif rel_code == "3":
            actions.append(f"Reach tier 2 by adding {addl_needed} new offering(s) to reach 3 total this quarter.")
        elif rel_code == "2":
            actions.append("Reach tier 1 by adding one additional distinct offering (4+ total).")
        if missing:
            actions.append("Candidate offering(s) to add: " + ", ".join(NICE[m] for m in missing[:max(1, addl_needed)]) + ".")
    nxt_size = next_size_letter(size_letter)
    if nxt_size:
        target = size_target_revenue(size_letter)
        if target:
            actions.append(f"To move {size_letter}→{nxt_size}, plan pipeline toward ≥ ${target:,.0f} in R12 revenue.")
    return actions

# -------------------------------------------------------------------------
# Recent invoices helper
def recent_invoices(billing_rows: List[Dict], limit: int = 10) -> List[Dict]:
    records = []
    for r in billing_rows:
        d = _to_date(r.get("Date") or r.get("Doc. Date"))
        if not d:
            continue
        records.append({
            "date": d,
            "type": (r.get("Type") or r.get("Department") or "").strip(),
            "amount": _to_float(r.get("REVENUE")),
            "desc": r.get("Description", "").strip() if r.get("Description") else "",
            "row": r
        })
    records.sort(key=lambda x: x["date"], reverse=True)
    out = []
    for it in records[:limit]:
        out.append({
            "Date": it["date"].strftime("%Y-%m-%d"),
            "Type": it["type"],
            "REVENUE": it["amount"],
            "Description": it["desc"]
        })
    return out

# -------------------------------------------------------------------------
# Main entry: build inquiry brief (name-first; ID optional)
def build_inquiry_brief(name_or_id: str) -> Optional[Dict]:
    probe_name = (name_or_id or "").strip()
    rows = find_inquiry_rows_flexible(customer_id=None, customer_name=probe_name)

    if not rows["report"] and not rows["billing"]:
        cid_guess = (name_or_id or "").strip()
        if cid_guess:
            rows = find_inquiry_rows_flexible(customer_id=cid_guess, customer_name=None)

    report_rows = rows["report"]
    billing_rows = rows["billing"]
    if not report_rows and not billing_rows:
        return None

    # Display name
    disp_name = None
    for r in (report_rows + billing_rows):
        nm = _pick_name(r)
        if nm and nm != "Unnamed":
            disp_name = nm
            break
    if not disp_name:
        disp_name = probe_name or "Unnamed"

    # Display ID if we can find one; otherwise "n/a"
    display_id = "n/a"
    for idx, r in enumerate(report_rows + billing_rows):
        display_id = _pick_id(r, idx)
        if display_id:
            break

    # ── Pull SEGMENT from report if it exists (e.g., 'C2') ───────────────
    seg_from_report = None
    for r in report_rows:
        for key in ("R12 Segment (Ship to ID)", "R12 Segment (Sold to ID)", "R12 Segment"):
            val = str(r.get(key, "")).strip()
            if val:
                seg_from_report = val.upper().replace(" ", "")
                break
        if seg_from_report:
            break

    # Aggregations
    rep = _aggregate_report_r12(report_rows)
    bil = _aggregate_billing(billing_rows)

    # If report says "C2", honor it exactly; else compute from data
    if seg_from_report and len(seg_from_report) == 2 and seg_from_report[0] in "ABCD":
        size_letter = seg_from_report[0]
        rel_code    = seg_from_report[1]
        total_r12_for_size = rep["total_r12"] if rep["total_r12"] > 0 else bil.get("total_last_365", 0.0)
    else:
        total_r12_for_size = rep["total_r12"] if rep["total_r12"] > 0 else bil.get("total_last_365", 0.0)
        size_letter = classify_account_size(total_r12_for_size)
        had_rev     = (total_r12_for_size > 0) or any(v > 0 for v in bil["by_dept"].values())
        rel_code    = classify_relationship(bil["offerings_count"], had_rev)

    visit_focus = _recommend_focus(bil["by_dept"], bil["last_invoice"])
    next_level_actions = _actions_next_level(size_letter, rel_code, bil["offerings_count"], bil["by_dept"])

    # Build context block for the AI
    lines: List[str] = []
    lines.append("<CONTEXT:INQUIRY>")
    lines.append(f"Customer: {disp_name} (ID: {display_id})")
    lines.append(f"Segmentation: Size={size_letter}  Relationship={rel_code}")
    lines.append(f"R12 Revenue (approx): ${total_r12_for_size:,.0f}")
    lines.append(
        "By Offering (R12-ish): "
        + ", ".join([f"{NICE[k]} ${rep.get(k, 0.0):,.0f}" for k in ["SERVICE", "PARTS", "RENTAL", "NEW_EQUIP", "USED_EQUIP"]])
    )
    lines.append(
        f"Billing: invoices={bil['invoice_count']}  "
        f"last_invoice={bil['last_invoice'] or 'N/A'}  "
        f"avg_days_between_invoices={bil['avg_days_between_invoices'] or 'N/A'}"
    )
    if bil["top_months"]:
        lines.append("Top Months: " + ", ".join([f"{m} ${v:,.0f}" for m, v in bil["top_months"]]))
    if bil["top_offerings"]:
        lines.append("Top Offerings: " + ", ".join([f"{NICE.get(k, k)} ${v:,.0f}" for k, v in bil["top_offerings"]]))
    lines.append("</CONTEXT:INQUIRY>")

    return {
        "context_block": "\n".join(lines),
        "size_letter": size_letter,
        "relationship_code": rel_code,
        "inferred_name": disp_name,
        "recent_invoices": recent_invoices(billing_rows, limit=10),  # for optional use
    }
# =========================
# MAP / GEO HELPERS (UPDATED)
# =========================
def _safe_float(x) -> Optional[float]:
    try:
        s = str(x).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None

def _parse_county_state(v: str) -> Tuple[str, str]:
    """
    Accepts values like 'Marion IN', 'Marion, IN', 'Marion County IN', 'IN'
    Returns (county, state2)
    """
    s = (str(v) or "").strip()
    if not s:
        return ("", "")
    s = re.sub(r"\s+", " ", s.replace(",", " ")).strip()
    parts = s.split(" ")
    state = ""
    county = ""
    # grab last 2-letter token as state
    for token in reversed(parts):
        if re.fullmatch(r"[A-Za-z]{2}", token):
            state = token.upper()
            break
    # county = everything before that, minus "county"
    if state:
        before = " ".join(parts[: parts.index(state)])
        before = before.replace("County", "").replace("county", "").strip()
        county = before
    else:
        # only a county word or nothing provided
        if len(parts) == 1 and len(parts[0]) == 2:
            state = parts[0].upper()
        else:
            county = s
    return (county, state)

def _guess_state_from_row(row: Dict) -> str:
    # prefer explicit 'State' if present
    for k in ("State", "STATE"):
        v = str(row.get(k, "")).strip()
        if v and len(v) == 2:
            return v.upper()
    # try County State style
    for k in ("County State", "County/State", "CountyState"):
        v = str(row.get(k, "")).strip()
        if v:
            _, st = _parse_county_state(v)
            if st:
                return st
    # last resort from ZIP (US-only heuristic not implemented)
    return ""

def _build_full_address_from_report(r: Dict) -> str:
    addr = str(r.get("Address", "")).strip()
    city = str(r.get("City", "")).strip()
    state = _guess_state_from_row(r)
    zc = str(r.get("Zip Code", "") or r.get("ZIP", "")).strip()
    parts = [p for p in [addr, city, state, zc, "USA"] if p]
    return ", ".join(parts)

def _sales_rep_of(r: Dict) -> str:
    return (str(r.get("Sales Rep Name", "")).strip()
            or str(r.get("Sales Rep", "")).strip()
            or "(unknown)")

def _read_customer_locations() -> List[Dict]:
    """
    Expect columns (case-insensitive / forgiving):
      - Account Name
      - City
      - Address
      - County State
      - Zip Code
      - Min of Latitude
      - Min of Longitude
    """
    path = os.getenv("CUSTOMER_LOCATION_CSV", "customer_location.csv")
    if not os.path.exists(path):
        return []
    df = _read_csv(path)
    # normalize headers commonly seen
    ren = {}
    for c in df.columns:
        lc = c.strip().lower()
        if lc == "account name":
            ren[c] = "Account Name"
        elif lc == "city":
            ren[c] = "City"
        elif lc in ("address", "street"):
            ren[c] = "Address"
        elif lc in ("county state", "county/state", "countystate"):
            ren[c] = "County State"
        elif lc in ("zip", "zip code", "postal code"):
            ren[c] = "Zip Code"
        elif lc in ("min of latitude", "latitude", "lat"):
            ren[c] = "Min of Latitude"
        elif lc in ("min of longitude", "longitude", "lon", "lng"):
            ren[c] = "Min of Longitude"
    if ren:
        df = df.rename(columns=ren)

    rows = df.to_dict(orient="records")
    # dedupe by (name norm) so we don't drop multiple *distinct* addresses for same name.
    # Your ask was: "if same place listed multiple times, only one pin".
    # We'll dedupe by (name, city, address) when available; else by name.
    seen = set()
    out = []
    for r in rows:
        name = str(r.get("Account Name", "")).strip()
        city = str(r.get("City", "")).strip()
        addr = str(r.get("Address", "")).strip()
        key = ( _norm_name(name), _norm_name(city), _norm_name(addr) ) if (city or addr) else (_norm_name(name),)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out

def _segment_from_report_rows(report_rows: List[Dict]) -> Optional[str]:
    for r in report_rows:
        for k in ("R12 Segment (Ship to ID)", "R12 Segment (Sold to ID)", "R12 Segment"):
            v = str(r.get(k, "")).strip()
            if v:
                s = v.upper().replace(" ", "")
                if len(s) == 2 and s[0] in "ABCD":
                    return s
    return None

def _territory_from_rows(report_rows: List[Dict], billing_rows: List[Dict]) -> str:
    # Prefer report's Sales Rep Name
    for r in report_rows:
        rep = _sales_rep_of(r)
        if rep and rep != "(unknown)":
            return rep
    # fallback: sometimes billing has "Sales Rep"
    for r in billing_rows:
        rep = (str(r.get("Sales Rep", "")).strip())
        if rep:
            return rep
    return "(unknown)"

def get_locations_with_geo() -> List[Dict]:
    """
    Returns a list of dicts per map pin:
      {
        "label": <Account Name>,
        "lat": <float>, "lon": <float>,
        "sales_rep": <territory/rep or '(unknown)'>,
        "state": <2-letter or ''>,
        "county": <string or ''>,
        "zip": <string or ''>,
        "segment": <e.g., 'C2' or 'P'>,
        "size_letter": <'A'|'B'|'C'|'D' or ''>,
        "r12_total": <float>,
      }
    Ensures no 'undefined' values in popups.
    """
    loc_rows = _read_customer_locations()
    rep_all = load_customer_report()
    bil_all = load_customer_billing()

    pins: List[Dict] = []

    for loc in loc_rows:
        name = (loc.get("Account Name") or "").strip()
        if not name:
            # try any non-empty value as a label
            name = _pick_name(loc)

        lat = _safe_float(loc.get("Min of Latitude"))
        lon = _safe_float(loc.get("Min of Longitude"))
        if lat is None or lon is None:
            # skip entries without coordinates
            continue

        # state/county/zip from location file
        county_state_raw = loc.get("County State", "")
        county, state = _parse_county_state(county_state_raw)
        if not state:
            state = str(loc.get("State", "")).strip().upper()
        zipc = str(loc.get("Zip Code", "")).strip()

        # match rows from both sources by normalized name
        want = _norm_name(name)
        report_rows = [r for r in rep_all if _norm_name(_pick_name(r)) == want]
        billing_rows = [r for r in bil_all if _norm_name(_pick_name(r)) == want]

        # territory/rep
        territory = _territory_from_rows(report_rows, billing_rows)

        # aggregates for segment & R12
        seg = _segment_from_report_rows(report_rows)
        rep_agg = _aggregate_report_r12(report_rows) if report_rows else {"total_r12": 0.0}
        bil_agg = _aggregate_billing(billing_rows) if billing_rows else {
            "by_dept": {}, "total_last_365": 0.0, "offerings_count": 0,
            "last_invoice": "", "invoice_count": 0, "avg_days_between_invoices": None,
            "top_months": [], "top_offerings": []
        }

        total_for_size = rep_agg.get("total_r12", 0.0) or bil_agg.get("total_last_365", 0.0)

        if seg:
            size_letter = seg[0]
            rel_code = seg[1]
        else:
            size_letter = classify_account_size(total_for_size) if (report_rows or billing_rows) else ""
            had_rev = (total_for_size > 0) or any((v or 0.0) > 0.0 for v in bil_agg.get("by_dept", {}).values())
            rel_code = classify_relationship(bil_agg.get("offerings_count", 0), had_rev) if (report_rows or billing_rows) else "P"

        segment_str = f"{size_letter}{rel_code}" if (size_letter and rel_code) else "P"

        pins.append({
            "label": name,
            "lat": lat,
            "lon": lon,
            "sales_rep": territory or "(unknown)",
            "state": state or "",
            "county": county or "",
            "zip": zipc or "",
            "segment": segment_str,
            "size_letter": size_letter or "",
            "r12_total": float(total_for_size or 0.0),
        })

    return pins
