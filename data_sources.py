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
# MAP / GEO HELPERS (CSV with lat/lon) — with R12 enrichment + fuzzy name matching
# =========================
import math  # ensure this is imported at top with others

CUSTOMER_LOCATION_CSV = os.getenv("CUSTOMER_LOCATION_CSV", "customer_location.csv")

@lru_cache(maxsize=1)
def load_customer_location() -> List[Dict]:
    """
    Expected columns:
      - Account Name
      - City
      - Address
      - County State
      - Zip Code
      - Min of Latitude
      - Min of Longitude
    """
    df = _read_csv(CUSTOMER_LOCATION_CSV)
    if df.empty:
        return []
    # normalize common header quirks
    rename_map = {
        "Min of Latitude": "lat",
        "Min of Longitude": "lon",
        "Zip Code": "zip",
        "County State": "county_state",
        "Account Name": "account_name",
        "City": "city",
        "Address": "address",
    }
    for src, dst in rename_map.items():
        if src in df.columns:
            df[dst] = df[src]

    # strip whitespace on object cols
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()

    # coerce lat/lon to floats; leave invalid as None (filtered later)
    def _to_float_or_none(x):
        try:
            s = str(x).strip()
            if not s or s.lower() == "nan":
                return None
            val = float(s)
            if not math.isfinite(val):
                return None
            return val
        except Exception:
            return None

    if "lat" in df.columns:
        df["lat"] = df["lat"].apply(_to_float_or_none)
    if "lon" in df.columns:
        df["lon"] = df["lon"].apply(_to_float_or_none)

    return df.to_dict(orient="records")

def _guess_state_from_cs(cs: str) -> str:
    s = (cs or "").strip()
    # match trailing ", IN" or just "IN"
    m = re.search(r"\b([A-Z]{2})\b\s*$", s.upper())
    return m.group(1) if m else ""

def _build_name_index_for_report() -> Dict[str, Dict]:
    """
    Build lookup: normalized name -> {
        segment_from_report (if present),
        sales_rep,
        size_letter,
        relationship_code,
        r12  (R12 Aftermarket or sum of parts/service/rental; fallback = last_365 from billing)
    }
    """
    rep_rows = load_customer_report()
    bil_rows = load_customer_billing()

    def nm(r: Dict) -> str:
        return _norm_name(_pick_name(r))

    rep_groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in rep_rows:
        k = nm(r)
        if k:
            rep_groups[k].append(r)

    bil_groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in bil_rows:
        k = nm(r)
        if k:
            bil_groups[k].append(r)

    all_names = set(rep_groups.keys()) | set(bil_groups.keys())
    out: Dict[str, Dict] = {}

    for nkey in all_names:
        rep_list = rep_groups.get(nkey, [])
        bil_list = bil_groups.get(nkey, [])

        # 1) aggregate
        rep_agg = _aggregate_report_r12(rep_list) if rep_list else {"total_r12": 0.0}
        bil_agg = _aggregate_billing(bil_list) if bil_list else {
            "by_dept": {},
            "total_last_365": 0.0,
            "offerings_count": 0,
        }

        # 2) segment from report, if present (e.g., "C2")
        seg_from_report = None
        for r in rep_list:
            for key in ("R12 Segment (Ship to ID)", "R12 Segment (Sold to ID)", "R12 Segment"):
                val = str(r.get(key, "")).strip()
                if val:
                    seg_from_report = val.upper().replace(" ", "")
                    break
            if seg_from_report:
                break

        # 3) pick an R12 value for sizing (report preferred, else last 365 billing)
        r12_for_size = rep_agg.get("total_r12", 0.0)
        if not r12_for_size or r12_for_size <= 0:
            r12_for_size = float(bil_agg.get("total_last_365", 0.0) or 0.0)

        # 4) compute size/relationship
        if seg_from_report and len(seg_from_report) == 2 and seg_from_report[0] in "ABCD":
            size_letter = seg_from_report[0]
            relationship_code = seg_from_report[1]
        else:
            size_letter = classify_account_size(r12_for_size)
            had_rev = (r12_for_size > 0) or any(v > 0 for v in bil_agg.get("by_dept", {}).values())
            relationship_code = classify_relationship(bil_agg.get("offerings_count", 0), had_rev)

        # 5) sales rep (territory) from report rows if available
        sales_rep = ""
        for r in rep_list:
            v = (r.get("Sales Rep Name") or r.get("Sales Rep") or "").strip()
            if v:
                sales_rep = v
                break

        out[nkey] = {
            "segment_from_report": seg_from_report or "",
            "sales_rep": sales_rep,
            "size_letter": size_letter or "",
            "relationship_code": relationship_code or "",
            "r12": float(r12_for_size or 0.0),
        }

    return out

def _closest_name_key(nkey: str, keys: List[str], cutoff: float = 0.86) -> Optional[str]:
    """Return best fuzzy match for nkey among keys, if similarity >= cutoff."""
    if not nkey or not keys:
        return None
    # quick exact pass
    if nkey in keys:
        return nkey
    # fuzzy
    try:
        cand = difflib.get_close_matches(nkey, keys, n=1, cutoff=cutoff)
        if cand:
            return cand[0]
    except Exception:
        pass
    # token-subset loose match
    wt = {t for t in nkey.split() if len(t) > 1}
    best = None
    best_score = 0
    for k in keys:
        ct = {t for t in k.split() if len(t) > 1}
        if not wt:
            continue
        score = len(wt & ct) / len(wt)
        if score > best_score and score >= 0.67:
            best = k
            best_score = score
    return best

def get_locations_with_geo() -> List[Dict]:
    """
    Return unique map points from customer_location.csv (lat/lon present),
    enriched with:
      - sales_rep (from report: Sales Rep Name)
      - segment (prefer report; else computed size+relationship)
      - size_letter, relationship_code
      - r12 (report aftermarket R12 or sum; fallback = billing last 365)
      - state, county, zip, city, address
    Deduped by normalized account name. Rows with invalid coords are skipped.
    Uses fuzzy matching to align location names to report/billing names.
    """
    loc_rows = load_customer_location()
    if not loc_rows:
        return []

    name_meta = _build_name_index_for_report()
    meta_keys = list(name_meta.keys())

    seen_names = set()
    out: List[Dict] = []

    for r in loc_rows:
        label = (r.get("account_name") or r.get("Account Name") or "").strip()
        if not label:
            continue
        nkey = _norm_name(label)
        if not nkey or nkey in seen_names:
            continue

        # coords (skip any invalid)
        lat_raw = r.get("lat") or r.get("Min of Latitude")
        lon_raw = r.get("lon") or r.get("Min of Longitude")
        try:
            lat = float(str(lat_raw).strip())
            lon = float(str(lon_raw).strip())
            if not (math.isfinite(lat) and math.isfinite(lon)):
                continue
        except Exception:
            continue

        # address-ish info
        city   = (r.get("city") or r.get("City") or "").strip()
        addr   = (r.get("address") or r.get("Address") or "").strip()
        cs     = (r.get("county_state") or r.get("County State") or "").strip()
        _zip   = (r.get("zip") or r.get("Zip Code") or "").strip()
        state  = _guess_state_from_cs(cs)
        county = cs.split(",")[0].strip() if "," in cs else ""

        # enrich from meta — exact or fuzzy
        meta = name_meta.get(nkey)
        if not meta:
            # try fuzzy on normalized keys (handles small typos like CYRSTAL → CRYSTAL)
            close_key = _closest_name_key(nkey, meta_keys, cutoff=0.86)
            if close_key:
                meta = name_meta.get(close_key)

        # safe defaults if still missing
        size_letter = meta.get("size_letter", "") if meta else ""
        rel_code    = meta.get("relationship_code", "") if meta else ""
        seg_display = (meta.get("segment_from_report") or (f"{size_letter}{rel_code}" if size_letter and rel_code else "")) if meta else ""
        sales_rep   = (meta.get("sales_rep", "") if meta else "") or "-"          # never "undefined" in JSON
        r12_val     = float(meta.get("r12", 0.0)) if meta else 0.0

        out.append({
            "label": label,
            "sales_rep": sales_rep,
            "segment": seg_display,
            "size": size_letter or "-",
            "relationship": rel_code or "-",
            "r12": r12_val,
            "lat": lat,
            "lon": lon,
            "state": state or "-",
            "county": county or "-",
            "zip": _zip or "-",
            "city": city or "",
            "address": addr or "",
        })
        seen_names.add(nkey)

    return out

