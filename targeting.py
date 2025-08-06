# targeting.py
from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

@app.route('/targeting')
def targeting_page():
    # Load customer report
    df = pd.read_csv("customer_report.csv", dtype=str)
    df = df.rename(columns=lambda x: x.strip().upper())

    # Load billing data
    billing = pd.read_csv("customer_billing.csv", dtype=str)
    billing = billing.rename(columns=lambda x: x.strip().upper())

    # Ensure matching company names are strings
    df["COMPANY"] = df["COMPANY"].astype(str)
    billing["CUSTOMER"] = billing["CUSTOMER"].astype(str)

    # Normalize invoice types
    billing["INVOICE_TYPE"] = billing["INVOICE_TYPE"].str.upper()

    # Create a pivot to determine if a customer has ever been billed for each service
    service_pivot = billing.pivot_table(
        index="CUSTOMER",
        columns="INVOICE_TYPE",
        values="INVOICE_NO",
        aggfunc="count",
        fill_value=0
    )

    # Determine services customer has NOT used (needs attention)
    services = ["PARTS", "RENTAL", "SERVICE", "NEW", "USED"]
    for svc in services:
        if svc not in service_pivot.columns:
            service_pivot[svc] = 0

    # Merge data
    df = df.merge(service_pivot, how="left", left_on="COMPANY", right_index=True)
    df[services] = df[services].fillna(0)

    # Mark focus targets: if they have no usage in any category
    for svc in services:
        df[f"TARGET_{svc}"] = df[svc] == 0

    # Group by Territory
    df_grouped = {}
    for territory, group in df.groupby("SALES TERRITORY"):
        sorted_group = group.sort_values("SEGMENT")
        df_grouped[territory] = sorted_group

    return render_template("targeting.html", grouped=df_grouped, services=services)
