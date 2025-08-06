import os
import pandas as pd
from flask import Blueprint, render_template

targeting_bp = Blueprint("targeting", __name__, template_folder="templates")

CUSTOMER_CSV = os.getenv("TARGETING_CUSTOMER_CSV", "customer_report")
BILLING_CSV  = os.getenv("TARGETING_BILLING_CSV",  "customer_billing")

def _resolve_path(base: str) -> str:
    for ext in ("", ".csv", ".xlsx"):
        p = base + ext
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Cannot find {base}(.csv/.xlsx)")

def _read_loose(base: str) -> pd.DataFrame:
    """Try UTF-8 CSV → CP1252 CSV → Excel (openpyxl) → fallback CSV."""
    path = _resolve_path(base)
    # 1) UTF-8 CSV
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8", on_bad_lines="skip")
    except Exception:
        pass
    # 2) CP1252 CSV
    try:
        return pd.read_csv(path, dtype=str, encoding="cp1252", on_bad_lines="skip")
    except Exception:
        pass
    # 3) Excel if extension matches
    if path.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(path, dtype=str, engine="openpyxl")
    # 4) Final CSV fallback
    return pd.read_csv(path, dtype=str, on_bad_lines="skip")

@targeting_bp.route("/targeting")
def targeting_page():
    # 1) Load data
    cust = _read_loose(CUSTOMER_CSV)
    bill = _read_loose(BILLING_CSV)

    # 2) Pivot billing: CUSTOMER × Type → bool purchased
    bill["CUSTOMER"] = bill["CUSTOMER"].astype(str)
    bill["Type"]     = bill["Type"].astype(str)
    bill["bought"]   = True
    pivot = (
        bill
        .pivot_table(index="CUSTOMER",
                     columns="Type",
                     values="bought",
                     aggfunc="max",
                     fill_value=False)
    )

    # 3) Master list of services
    SERVICES = ["Parts", "Service", "Rental", "New Equipment", "Used Equipment"]
    for svc in SERVICES:
        if svc not in pivot.columns:
            pivot[svc] = False
    pivot = pivot[SERVICES]

    # 4) Join to customer_report on Sold-to Name
    cust["Sold to Name"]   = cust["Sold to Name"].astype(str)
    cust["Sales Rep Name"] = cust["Sales Rep Name"].astype(str)
    df = cust.set_index("Sold to Name").join(pivot, how="left").fillna(False)

    # 5) Compute recommended services
    def recommend(row):
        return [svc for svc in SERVICES if not row.get(svc, False)]
    df["Recommended Services"] = df.apply(recommend, axis=1)

    # 6) Group by territory
    rep_groups = {
        rep: [
            {"company": name, "services": row["Recommended Services"]}
            for name, row in group.iterrows()
        ]
        for rep, group in df.groupby("Sales Rep Name")
    }

    # 7) Render
    return render_template("targeting.html", rep_groups=rep_groups)
