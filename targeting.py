# targeting.py

import os
import io
import math
import pandas as pd
from flask import Blueprint, render_template, request, send_file

# ────────────────────────────────────────────────────────────────────────────
# 1) Blueprint & Config
# ────────────────────────────────────────────────────────────────────────────
targeting_bp = Blueprint("targeting", __name__, template_folder="templates")

CUSTOMER_CSV = os.getenv("TARGETING_CUSTOMER_CSV", "customer_report")
BILLING_CSV  = os.getenv("TARGETING_BILLING_CSV",  "customer_billing")

# ────────────────────────────────────────────────────────────────────────────
# 2) Data Loading & Cleaning Helpers
# ────────────────────────────────────────────────────────────────────────────
def _resolve_path(base: str) -> str:
    for ext in ("", ".csv", ".xlsx"):
        p = base + ext
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Cannot find {base}(.csv/.xlsx)")

def _read_table(base: str) -> pd.DataFrame:
    """Try UTF8 CSV → CP1252 CSV → Excel."""
    path = _resolve_path(base)
    for enc in ("utf-8", None, "cp1252"):
        try:
            if path.endswith((".xls", ".xlsx")):
                return pd.read_excel(path, dtype=str)
            return pd.read_csv(path, dtype=str, encoding=enc, on_bad_lines="skip")
        except Exception:
            continue
    # Fallback: pandas autodetect
    return pd.read_csv(path, dtype=str)

def _to_number(x: str) -> float:
    if pd.isna(x):
        return 0.0
    s = (str(x).strip()
           .replace("$","")
           .replace(",","")
           .replace("(","-")
           .replace(")",""))
    try:
        return float(s)
    except ValueError:
        return 0.0

def load_customer() -> pd.DataFrame:
    df = _read_table(CUSTOMER_CSV)
    # ensure all expected columns
    cols = {
        "ship_to_name":"Ship to Name",
        "sold_to_name":"Sold to Name",
        "r12":"Revenue Rolling 12 Months - Aftermarket",
        "r13":"Revenue Rolling 13 - 24 Months - Aftermarket",
        "r25":"Revenue Rolling 25 - 36 Months - Aftermarket",
        "new36":"New Equip R36 Revenue",
        "used36":"Used Equip R36 Revenue",
        "parts":"Parts Revenue R12",
        "service":"Service Revenue R12 (Includes GM)",
        "rental":"Rental Revenue R12",
        "county":"County State",
    }
    for c in cols.values():
        if c not in df.columns:
            df[c] = pd.NA
    # parse numbers
    for key in ("r12","r13","r25","new36","used36","parts","service","rental"):
        df[cols[key]] = df[cols[key]].apply(_to_number)
    # derive extras
    df["Momentum %"] = (
        (df[cols["r12"]] - df[cols["r13"]])
        / (df[cols["r13"]].abs() + 1e-9)
    )
    df["3Yr Peak"]   = df[[cols["r12"],cols["r13"],cols["r25"]]].max(axis=1)
    df["Recapture"]  = (df["3Yr Peak"] - df[cols["r12"]]).clip(lower=0)
    df["Breadth"]    = (
        (df[[cols["parts"],cols["service"],cols["rental"],cols["new36"],cols["used36"]]] > 0)
        .sum(axis=1)
    )
    # state code
    df["State"] = df[cols["county"]].astype(str).str[-2:].str.upper()
    # coerce names to str for joins
    df[cols["sold_to_name"]] = df[cols["sold_to_name"]].astype(str)
    df[cols["ship_to_name"]] = df[cols["ship_to_name"]].astype(str)
    return df

def load_billing() -> pd.DataFrame|None:
    try:
        b = _read_table(BILLING_CSV)
    except FileNotFoundError:
        return None
    for col in ("Date","CUSTOMER","REVENUE"):
        if col not in b.columns:
            b[col] = pd.NA
    b["Date"]    = pd.to_datetime(b["Date"], errors="coerce")
    b["REVENUE"] = b["REVENUE"].apply(_to_number)
    b["CUSTOMER"]= b["CUSTOMER"].astype(str)
    today = pd.Timestamp.now().normalize()
    b["Days Ago"] = (today - b["Date"]).dt.days
    latest = (
        b.groupby("CUSTOMER", dropna=False)["Date"]
         .max()
         .reset_index(name="Last Invoice Date")
    )
    def roll_sum(days):
        return (
            b[b["Days Ago"] <= days]
            .groupby("CUSTOMER")["REVENUE"]
            .sum()
            .rename(f"Rev {days}d")
        )
    out = latest.join(roll_sum(90), on="CUSTOMER") \
                .join(roll_sum(180), on="CUSTOMER") \
                .join(roll_sum(365), on="CUSTOMER")
    out["Days Since Last Invoice"] = (today - out["Last Invoice Date"]).dt.days
    return out

# ────────────────────────────────────────────────────────────────────────────
# 3) Segmentation & Tactics
# ────────────────────────────────────────────────────────────────────────────
def segment_account(row, momentum_thresh=0.20, recapture_thresh=100000):
    m = row["Momentum %"]
    r = row["Recapture"]
    b = row["Breadth"]
    if m < 0 and r >= recapture_thresh:
        return "ATTACK"
    if (row["Revenue Rolling 12 Months - Aftermarket"] > 0 and b <= 1) or m >= momentum_thresh:
        return "GROW"
    if row["Revenue Rolling 12 Months - Aftermarket"] > 0:
        return "MAINTAIN"
    return "TEST/EXPAND"

TACTICS = {
    "ATTACK":     "Win-back/displace…",
    "GROW":       "Upsell/cross-sell…",
    "MAINTAIN":   "Defend & delight…",
    "TEST/EXPAND":"Low-friction pilot…",
}

# ────────────────────────────────────────────────────────────────────────────
# 4) Route Handlers
# ────────────────────────────────────────────────────────────────────────────

@targeting_bp.route("/targeting")
def targeting_page():
    # parse thresholds
    momentum = float(request.args.get("momentum",  0.20))
    recapture = float(request.args.get("recapture",100000))

    cust = load_customer()
    bill = load_billing()

    # join billing by sold-to, then fill from ship-to
    if bill is not None:
        bill_idx = bill.set_index("CUSTOMER")
        cust = (
            cust
            .join(bill_idx, on="Sold to Name", rsuffix="_s")
            .join(bill_idx, on="Ship to Name", rsuffix="_h")
        )
        for fld in ("Last Invoice Date","Rev 90d","Rev 180d","Rev 365d","Days Since Last Invoice"):
            cust[fld] = cust[f"{fld}_s"].fillna(cust[f"{fld}_h"])
        cust.drop(columns=[c for c in cust if c.endswith(("_s","_h"))], inplace=True)

    # segmentation & tactic
    cust["Segment"] = cust.apply(segment_account, axis=1,
                                 args=(momentum, recapture))
    cust["Tactic"]  = cust["Segment"].map(TACTICS)

    # prepare table (just show first 20)
    cols = ["Sold to Name","Ship to Name","Segment","Tactic"]
    rows = cust[cols].head(20).to_dict(orient="records")
    total = len(cust)

    return render_template(
        "targeting.html",
        table_columns=cols,
        table_rows=rows,
        total_accounts=total,
        momentum=momentum,
        recapture=recapture
    )

@targeting_bp.route("/targeting/download")
def targeting_download():
    cust = load_customer()
    bill = load_billing()
    if bill is not None:
        bill_idx = bill.set_index("CUSTOMER")
        cust = (
            cust
            .join(bill_idx, on="Sold to Name", rsuffix="_s")
            .join(bill_idx, on="Ship to Name", rsuffix="_h")
        )
    cust["Segment"] = cust.apply(segment_account, axis=1)
    cust["Tactic"]  = cust["Segment"].map(TACTICS)

    buf = io.StringIO()
    cust.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(
        io.BytesIO(buf.getvalue().encode()),
        as_attachment=True,
        download_name="targeting_full.csv",
        mimetype="text/csv"
    )
