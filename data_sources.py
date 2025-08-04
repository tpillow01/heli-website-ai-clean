# data_sources.py
"""
Load inquiry CSVs and compute:
- Customer fuzzy match (by name)
- Billing aggregation (what/when/how often they spend)
- Segmentation:
    Account Size (A/B/C/D) from R12 revenue
    Relationship (1/2/3/P) from # of distinct offerings
- A preformatted <CONTEXT:INQUIRY> block for the AI
"""

from __future__ import annotations
import os, json, pandas as pd
from functools import lru_cache
from typing import List, Dict, Optional, Tuple
import difflib
from datetime import datetime
from collections import defaultdict

# ── file paths ────────────────────────────────────────────────────────────
CUSTOMER_REPORT_CSV  = os.getenv("CUSTOMER_REPORT_CSV",  "customer_report.csv")
CUSTOMER_BILLING_CSV = os.getenv("CUSTOMER_BILLING_CSV", "customer_billing.csv")

# ── robust CSV reader ────────────────────────────────────────────────────
def _read_csv(path: str) -> pd.DataFrame:
    """Read CSV with encoding fallbacks; return empty DF if missing."""
    if not os.path.exists(path):
        return pd.DataFrame()
    last_err = None
    attempts = [
        {"encoding": "utf-8",     "encoding_errors": "strict"},
        {"encoding": "utf-8-sig", "encoding_errors": "replace"},
        {"encoding": "cp1252",    "encoding_errors": "replace"},
        {"encoding": "latin1",    "encoding_errors": "replace"},
    ]
    for opts in attempts:
        try:
            return pd.read_csv(path, low_memory=False, **opts).fillna("")
        except Exception as e:
            last_err = e
            continue
    try:
        return pd.read_csv(path, low_memory=False).fillna("")
    except Exception:
        raise RuntimeError(f"Failed to read CSV {path}: {last_err}")

# ── cached loaders ────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def load_customer_report() -> List[Dict]:
    return _read_csv(CUSTOMER_REPORT_CSV).to_dict(orient="records")

@lru_cache(maxsize=1)
def load_customer_billing() -> List[Dict]:
    return _read_csv(CUSTOMER_BILLING_CSV).to_dict(orient="records")

# ── id/name heuristics used across both files ────────────────────────────
def _pick_id(row: Dict, fallback_index: int) -> str:
    for key in ("Ship to ID", "Sold to ID", "Ship to ID ", "Sold To BP", "Ship to code"):
        if key in row and str(row[key]).strip():
            return str(row[key]).strip()
    return str(fallback_index)

def _pick_name(row: Dict) -> str:
    for key in ("Ship to Name", "Sold to Name", "CUSTOMER", "Customer", "account_name", "company"):
        if key in row and str(row[key]).strip():
            return str(row[key]).strip()
    # fall back to any non-empty value
    for v in row.values():
        if str(v).strip():
            return str(v).strip()
    return "Unnamed"

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
    return sorted(targets.values(), key=lambda x: x["label"].lower())

def find_inquiry_rows(customer_id: str) -> Dict[str, List[Dict]]:
    report_hits, billing_hits = [], []
    for idx, r in enumerate(load_customer_report()):
        if _pick_id(r, idx) == customer_id:
            report_hits.append(r)
    for idx, r in enumerate(load_customer_billing()):
        if _pick_id(r, idx) == customer_id:
            billing_hits.append(r)
    return {"report": report_hits, "billing": billing_hits}

# ── fuzzy by name → id ───────────────────────────────────────────────────
def find_customer_id_by_name(name: str) -> Optional[str]:
    if not name:
        return None
    labels = make_inquiry_targets()
    # 1) direct substring preference
    name_l = name.lower()
    for item in labels:
        if item["label"] and item["label"].lower() in name_l:
            return item["id"]
    # 2) fuzzy
    options = [it["label"] for it in labels]
    match = difflib.get_close_matches(name, options, n=1, cutoff=0.72)
    if match:
        lbl = match[0]
        for it in labels:
            if it["label"] == lbl:
                return it["id"]
    return None

# ── helpers: numeric/date parsing ────────────────────────────────────────
def _to_float(x) -> float:
    try:
        if isinstance(x, str):
            x = x.replace(",", "").replace("$", "").strip()
        return float(x)
    except Exception:
        return 0.0

def _to_date(x) -> Optional[datetime]:
    if not x:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%Y/%m/%d"):
        try:
            return datetime.strptime(str(x).split(" ")[0], fmt)
        except Exception:
            continue
    return None

# ── department normalization (edit here if your labels differ) ───────────
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
    # unknown: still track, but won’t count as an offering
    return "OTHER"

OFFERING_KEYS = ["SERVICE", "PARTS", "RENTAL", "NEW_EQUIP", "USED_EQUIP"]

# ── segmentation from report R12 and offerings from billing ──────────────
def _aggregate_report_r12(report_rows: List[Dict]) -> Dict[str, float]:
    # best-effort pick of revenue columns (names from your screenshot)
    tot = 0.0
    buckets = {
        "NEW_EQUIP": 0.0,
        "USED_EQUIP": 0.0,
        "PARTS": 0.0,
        "SERVICE": 0.0,
        "RENTAL": 0.0
    }
    for r in report_rows:
        buckets["NEW_EQUIP"] += _to_float(r.get("New Equip R36 Revenue"))
        buckets["USED_EQUIP"] += _to_float(r.get("Used Equip R36 Revenue"))
        buckets["PARTS"]      += _to_float(r.get("Parts Revenue R12"))
        buckets["SERVICE"]    += _to_float(r.get("Service Revenue R12 (Includes GM)"))
        buckets["RENTAL"]     += _to_float(r.get("Rental Revenue R12"))
        # A consolidated R12 Aftermarket is present; we’ll rely on buckets instead
    tot = sum(buckets.values())
    return {"total_r12": tot, **buckets}

def _aggregate_billing(billing_rows: List[Dict]) -> Dict:
    # Sum by normalized department and by month, track recency and frequency
    by_dept = defaultdict(float)
    by_month = defaultdict(float)  # 'YYYY-MM'
    dates = []
    invoices = 0
    last_invoice = None
    for r in billing_rows:
        dep = normalize_department(r.get("Department"))
        amt = _to_float(r.get("REVENUE"))
        by_dept[dep] += amt
        d = _to_date(r.get("Doc. Date"))
        if d:
            dates.append(d)
            key = f"{d.year:04d}-{d.month:02d}"
            by_month[key] += amt
            if (last_invoice is None) or (d > last_invoice):
                last_invoice = d
        invoices += 1

    # frequency (avg days between invoices)
    avg_days = None
    if len(dates) >= 2:
        dates_sorted = sorted(dates)
        deltas = [(dates_sorted[i] - dates_sorted[i-1]).days for i in range(1, len(dates_sorted))]
        if deltas:
            avg_days = sum(deltas) / len(deltas)

    # offerings count (distinct non-zero known offerings)
    offerings_count = sum(1 for k in OFFERING_KEYS if by_dept.get(k, 0.0) > 0)

    # top months and top categories
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
    }

def classify_account_size(total_r12: float) -> str:
    # Platinum (A) = $200k+, Gold (B) = $80k+, Silver (C) = $10k+, Bronze (D) < $10k
    if total_r12 >= 200_000: return "A"
    if total_r12 >= 80_000:  return "B"
    if total_r12 >= 10_000:  return "C"
    return "D"

def classify_relationship(offerings_count: int, had_revenue: bool) -> str:
    # Provided mapping:
    #   1 = 4+, 2 = 3, 3 = 1, P = No Revenue
    # We’ll handle "2 offerings" explicitly as "2" to be transparent.
    if not had_revenue:
        return "P"
    if offerings_count >= 4:
        return "1"
    if offerings_count == 3:
        return "2"
    if offerings_count == 2:
        return "2"  # falls between 2 and 3; adjust if you prefer a custom label
    return "3"      # 0–1 offering

def _recommend_focus(by_dept: Dict[str, float], last_invoice_iso: str) -> List[str]:
    """
    Suggest what to lead with on a visit, based on recent spend patterns.
    """
    recs = []
    # Rank categories by spend
    ranked = sorted([(k, v) for k, v in by_dept.items() if k in OFFERING_KEYS],
                    key=lambda kv: kv[1], reverse=True)
    if ranked:
        leader = ranked[0][0]
        if leader == "RENTAL":
            recs.append("Discuss converting recurring rentals to a new or used purchase with service plan.")
        if leader == "SERVICE":
            recs.append("Offer a Preventive Maintenance agreement and upsell Parts kits.")
        if leader == "PARTS":
            recs.append("Bundle Parts with a Service inspection and explore consignment stock.")
        if leader in ("NEW_EQUIP", "USED_EQUIP"):
            recs.append("Review fleet age/usage and present replacement options with financing and service.")
    # Fill gaps: offerings with $0
    gaps = [k for k in ["SERVICE","PARTS","RENTAL","NEW_EQUIP"] if by_dept.get(k, 0.0) == 0]
    if gaps:
        recs.append("Add a second offering this quarter: " + ", ".join(gaps[:2]) + ".")
    # If very stale
    if not last_invoice_iso:
        recs.append("No recent invoices found — schedule a discovery visit and load survey.")
    return recs

def _actions_to_move(size_letter: str, rel_code: str, by_dept: Dict[str, float]) -> List[str]:
    """
    Concrete steps to improve relationship tier (e.g., B3 → B1).
    We assume size stays the same letter; focus on increasing offerings to 4+.
    """
    actions = []
    # offerings present
    present = {k for k in OFFERING_KEYS if by_dept.get(k, 0.0) > 0}
    missing = [k for k in OFFERING_KEYS if k not in present]

    if rel_code in ("3", "2"):
        needed = 4 - len(present)
        if needed > 0:
            actions.append(f"Add {needed} more distinct offering(s) to reach relationship tier 1 (4+ offerings).")
        if missing:
            nice = {"SERVICE":"Service", "PARTS":"Parts", "RENTAL":"Rental", "NEW_EQUIP":"New Equip", "USED_EQUIP":"Used Equip"}
            actions.append("Target new offering(s): " + ", ".join(nice[m] for m in missing[:max(1, needed)]))
    if rel_code == "P":
        actions.append("Establish first revenue with a low-friction entry: PM service check or short-term rental.")

    # size nudges: optional guidance to climb to next size tier
    size_targets = {"D": 10_000, "C": 80_000, "B": 200_000}
    if size_letter in size_targets:
        actions.append(f"To reach next size tier, annualize pipeline toward ${size_targets[size_letter]:,}+ in R12 revenue.")

    return actions

def build_inquiry_brief(name_or_id: str) -> Optional[Dict]:
    """
    Returns a dict with:
      - context_block (string)
      - size_letter (A/B/C/D)
      - relationship_code (1/2/3/P)
      - inferred_name (for display)
    """
    # identify customer id
    cid = name_or_id
    # if looks like a name, try to resolve to id
    if not cid.isdigit():
        probe = find_customer_id_by_name(name_or_id)
        if probe:
            cid = probe

    rows = find_inquiry_rows(cid)
    report_rows = rows["report"]
    billing_rows = rows["billing"]

    if not report_rows and not billing_rows:
        return None

    # name inference
    disp_name = None
    for r in (report_rows + billing_rows):
        nm = _pick_name(r)
        if nm and nm != "Unnamed":
            disp_name = nm
            break
    if not disp_name:
        disp_name = f"Customer {cid}"

    # aggregates
    rep = _aggregate_report_r12(report_rows)
    bil = _aggregate_billing(billing_rows)

    size_letter = classify_account_size(rep["total_r12"])
    had_rev = (rep["total_r12"] > 0) or any(v > 0 for v in bil["by_dept"].values())
    rel_code = classify_relationship(bil["offerings_count"], had_rev)

    # recommendations
    visit_focus = _recommend_focus(bil["by_dept"], bil["last_invoice"])
    move_actions = _actions_to_move(size_letter, rel_code, bil["by_dept"])

    # format a compact context block for the LLM
    nice = {"SERVICE":"Service", "PARTS":"Parts", "RENTAL":"Rental", "NEW_EQUIP":"New Equipment", "USED_EQUIP":"Used Equipment"}
    lines = []
    lines.append("<CONTEXT:INQUIRY>")
    lines.append(f"Customer: {disp_name} (ID: {cid})")
    lines.append(f"Segmentation: Size={size_letter}  Relationship={rel_code}")
    lines.append(f"R12 Revenue (approx): ${rep['total_r12']:,.0f}")
    lines.append("By Offering (R12-ish): " + ", ".join([f"{nice[k]} ${rep.get(k,0.0):,.0f}" for k in ["SERVICE","PARTS","RENTAL","NEW_EQUIP","USED_EQUIP"]]))
    lines.append(f"Billing: invoices={bil['invoice_count']}  last_invoice={bil['last_invoice'] or 'N/A'}  avg_days_between_invoices={bil['avg_days_between_invoices'] or 'N/A'}")
    if bil["top_months"]:
        lines.append("Top Months: " + ", ".join([f"{m} ${v:,.0f}" for m, v in bil['top_months']]))
    if bil["top_offerings"]:
        lines.append("Top Offerings: " + ", ".join([f"{nice.get(k,k)} ${v:,.0f}" for k, v in bil['top_offerings']]))
    if visit_focus:
        lines.append("Visit Focus Suggestions: " + " | ".join(visit_focus))
    if move_actions:
        lines.append("Actions to Move Up: " + " | ".join(move_actions))
    lines.append("</CONTEXT:INQUIRY>")

    return {
        "context_block": "\n".join(lines),
        "size_letter": size_letter,
        "relationship_code": rel_code,
        "inferred_name": disp_name
    }
