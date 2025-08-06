import os
import pandas as pd
from flask import Blueprint, render_template

# ────────────────────────────────────────────────────────────────────────────
# 1) Blueprint & Config
# ────────────────────────────────────────────────────────────────────────────
targeting_bp = Blueprint("targeting", __name__, template_folder="templates")

CUSTOMER_CSV = os.getenv("TARGETING_CUSTOMER_CSV", "customer_report")
BILLING_CSV  = os.getenv("TARGETING_BILLING_CSV",  "customer_billing")

# ────────────────────────────────────────────────────────────────────────────
# 2) File Reading Helpers
# ────────────────────────────────────────────────────────────────────────────
def _resolve_path(base: str) -> str:
    for ext in ("", ".csv", ".xlsx"):
        p = base + ext
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Cannot find {base}(.csv/.xlsx)")

def _read_loose(base: str) -> pd.DataFrame:
    """
    Try:
      1) UTF-8 CSV (skip bad lines)
      2) CP1252 CSV (skip bad lines)
      3) Excel via openpyxl
      4) Final CSV fallback
    """
    path = _resolve_path(base)
    for enc in ("utf-8", "cp1252"):
        try:
            return pd.read_csv(path, dtype=str, encoding=enc, on_bad_lines="skip")
        except Exception:
            pass
    if path.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(path, dtype=str, engine="openpyxl")
    return pd.read_csv(path, dtype=str, on_bad_lines="skip")

def _to_number(x: any) -> float:
    """Convert '$1,234', '(56)' → float."""
    if pd.isna(x):
        return 0.0
    s = str(x).strip().replace("$","").replace(",","")
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        return float(s)
    except ValueError:
        return 0.0

# ────────────────────────────────────────────────────────────────────────────
# 3) Segmentation Logic
# ────────────────────────────────────────────────────────────────────────────
C_R12 = "Revenue Rolling 12 Months - Aftermarket"
C_R13 = "Revenue Rolling 13 - 24 Months - Aftermarket"
C_R25 = "Revenue Rolling 25 - 36 Months - Aftermarket"

def segment_account(row, momentum_thresh=0.20, recapture_thresh=100000):
    r12 = row.get(C_R12, 0.0)
    r13 = row.get(C_R13, 0.0)
    r25 = row.get(C_R25, 0.0)
    # momentum = (this year – prior) / prior
    eps = 1e-9
    momentum = (r12 - r13) / (abs(r13) + eps)
    # three-year peak + recapture
    peak = max(r12, r13, r25)
    recapture = max(0.0, peak - r12)

    if recapture >= recapture_thresh:
        return "ATTACK"
    if momentum >= momentum_thresh:
        return "GROW"
    if r12 > 0:
        return "MAINTAIN"
    return "TEST/EXPAND"

TACTICS = {
    "ATTACK":      "Win-back/displace: multi-thread, demo, sharp pricing.",
    "GROW":        "Upsell/cross-sell: add PM, attachments, lithium conversion.",
    "MAINTAIN":    "Defend & delight: QBR cadence, SLA adherence.",
    "TEST/EXPAND": "Low-friction pilot: starter order, trial service, quick win."
}

# ────────────────────────────────────────────────────────────────────────────
# 4) Route Handler
# ────────────────────────────────────────────────────────────────────────────
@targeting_bp.route("/targeting")
def targeting_page():
    # Load & clean customer data
    cust = _read_loose(CUSTOMER_CSV)
    cust.columns = cust.columns.str.strip()
    for col in (C_R12, C_R13, C_R25):
        cust[col] = cust.get(col, 0).apply(_to_number)

    # Load & pivot billing for services
    bill = _read_loose(BILLING_CSV)
    bill.columns = bill.columns.str.strip()
    bill["CUSTOMER"] = bill["CUSTOMER"].astype(str)
    bill["Type"]     = bill["Type"].astype(str)
    bill["bought"]   = True

    pivot = (
        bill
        .pivot_table(
            index="CUSTOMER",
            columns="Type",
            values="bought",
            aggfunc="max",
            fill_value=False
        )
    )

    # Master list of services
    SERVICES = ["Parts", "Service", "Rental", "New Equipment", "Used Equipment"]
    for svc in SERVICES:
        if svc not in pivot.columns:
            pivot[svc] = False
    pivot = pivot[SERVICES]

    # Compute segmentation & recommendations
    cust["Sold to Name"]   = cust["Sold to Name"].astype(str)
    cust["Sales Rep Name"] = cust["Sales Rep Name"].astype(str)
    cust["Segment"] = cust.apply(segment_account, axis=1)
    cust["Tactic"]  = cust["Segment"].map(TACTICS)

    # Join in pivot so we know what they've never bought
    df = cust.set_index("Sold to Name").join(pivot, how="left").fillna(False)
    df["Recommended Services"] = df.apply(
        lambda r: [s for s in SERVICES if not r.get(s, False)], axis=1
    )

    # Group by territory & sort by segment
    SEG_ORDER = {"ATTACK": 0, "GROW": 1, "MAINTAIN": 2, "TEST/EXPAND": 3}
    rep_groups = {}
    for rep, group in df.groupby("Sales Rep Name"):
        entries = []
        for name, row in group.iterrows():
            entries.append({
                "company": name,
                "segment": row["Segment"],
                "services": row["Recommended Services"]
            })
        entries.sort(key=lambda e: SEG_ORDER.get(e["segment"], 99))
        rep_groups[rep] = entries

    return render_template("targeting.html", rep_groups=rep_groups)
