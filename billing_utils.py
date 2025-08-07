import csv
from collections import defaultdict
from datetime import datetime

CSV_FILE = "customer_billing.csv"

def parse_revenue(value):
    try:
        return float(value.replace("$", "").replace(",", "").strip())
    except:
        return 0.0

def load_billing_data():
    summary = defaultdict(lambda: {
        "total": 0.0,
        "count": 0,
        "categories": defaultdict(float),
        "latest": None
    })

    with open(CSV_FILE, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            customer = (row.get("CUSTOMER") or "").strip()
            revenue = parse_revenue(row.get("REVENUE", "0"))
            category = (row.get("Type") or "Unknown").strip()
            date_str = row.get("Date")

            if not customer:
                continue

            entry = summary[customer]
            entry["total"] += revenue
            entry["count"] += 1
            entry["categories"][category] += revenue

            try:
                date_obj = datetime.strptime(date_str, "%m/%d/%Y")
                if not entry["latest"] or date_obj > entry["latest"]:
                    entry["latest"] = date_obj
            except:
                pass

    return summary

def get_customer_insight(customer_name, summary):
    data = summary.get(customer_name)
    if not data:
        return f"No invoice history found for {customer_name}."

    latest_str = data["latest"].strftime("%B %d, %Y") if data["latest"] else "N/A"
    top_categories = sorted(data["categories"].items(), key=lambda x: -x[1])
    top_lines = [f"- {k}: ${v:,.2f}" for k, v in top_categories[:3]]

    return (
        f"Customer: {customer_name}\n"
        f"Total Revenue: ${data['total']:,.2f}\n"
        f"Number of Invoices: {data['count']}\n"
        f"Last Invoice Date: {latest_str}\n"
        f"Top Categories:\n" + "\n".join(top_lines)
    )
