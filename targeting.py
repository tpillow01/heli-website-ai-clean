import os
import pandas as pd
from flask import Blueprint, render_template

targeting_bp = Blueprint("targeting", __name__, template_folder="templates")

CUSTOMER_CSV = os.getenv("TARGETING_CUSTOMER_CSV", "customer_report")
BILLING_CSV  = os.getenv("TARGETING_BILLING_CSV",  "customer_billing")

def _resolve_path(base):
    for ext in ("", ".csv", ".xlsx"):
        p = base + ext
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Cannot find {base}(.csv/.xlsx)")

def _read_loose(base) -> pd.DataFrame:
    path = _resolve_path(base)
    # try CSV first, then Excel
    try:
        return pd.read_csv(path, dtype=str)
    except Exception:
        return pd.read_excel(path, dtype=str)

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

    # 3) Define the master list of services
    SERVICES = ["Parts", "Service", "Rental", "New Equipment", "Used Equipment"]
    # ensure all columns present
    for svc in SERVICES:
        if svc not in pivot.columns:
            pivot[svc] = False
    pivot = pivot[SERVICES]

    # 4) Join to customer_report on Sold-to Name
    cust["Sold to Name"]    = cust["Sold to Name"].astype(str)
    cust["Sales Rep Name"]  = cust["Sales Rep Name"].astype(str)
    df = cust.set_index("Sold to Name").join(pivot, how="left").fillna(False)

    # 5) Compute “Recommended Services” = those they haven’t bought
    def _recommend(row):
        return [svc for svc in SERVICES if not row.get(svc, False)]

    df["Recommended Services"] = df.apply(_recommend, axis=1)

    # 6) Group by territory (Sales Rep)
    rep_groups = {}
    for rep, group in df.groupby("Sales Rep Name"):
        rep_groups[rep] = [
            {
                "company": name,
                "services": row["Recommended Services"]
            }
            for name, row in group.iterrows()
        ]

    # 7) Render one table per territory
    return render_template("targeting.html", rep_groups=rep_groups)
