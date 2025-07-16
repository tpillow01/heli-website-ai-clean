# ai_logic.py

import json
import re
import difflib
from typing import List, Dict, Any

# —————————————————————————————————————————————————————————————————————
# Load JSON data once
# —————————————————————————————————————————————————————————————————————
with open("accounts.json", "r", encoding="utf-8") as f:
    accounts_raw = json.load(f)
accounts_data = {
    acct["Account Name"].lower(): acct
    for acct in accounts_raw
    if "Account Name" in acct
}

with open("models.json", "r", encoding="utf-8") as f:
    models_data = json.load(f)


def get_customer_context(customer_name: str) -> str:
    """
    Pulls the exact profile fields out of accounts.json,
    including SIC Code, Industry, etc.
    """
    acct = accounts_data.get(customer_name.lower()) if customer_name else None
    if not acct:
        return ""
    lines = ["<span class=\"section-label\">Customer Profile:</span>"]
    for field in ("Company", "Industry", "SIC Code",
                  "Total Company Fleet Size", "Truck Types at Location"):
        val = acct.get(field, "N/A")
        lines.append(f"- {field}: {val}")
    lines.append("")  # blank line
    return "\n".join(lines)


def filter_models_for_account(user_input: str) -> List[Dict[str, Any]]:
    """
    Exactly the same 3‑model filter you were using:
      • keyword filters (narrow aisle, rough terrain, electric)
      • capacity hints like “3000 lb”
      • fallback to top 3 by Capacity_lbs
    """
    ui = user_input.lower()
    cands = models_data[:]

    # keyword filters
    if "narrow aisle" in ui:
        cands = [m for m in cands if "narrow" in m.get("Type", "").lower()]
    if "rough terrain" in ui:
        cands = [m for m in cands if "rough" in m.get("Type", "").lower()]
    if "electric" in ui:
        cands = [m for m in cands if "electric" in m.get("Power", "").lower()]

    # capacity hints
    caps = [int(n.replace(",", "")) for n in re.findall(r"(\d{3,5})\s*lbs?", ui)]
    if caps:
        min_cap = max(caps)
        cands = [
            m for m in cands
            if isinstance(m.get("Capacity_lbs"), (int, float))
               and m["Capacity_lbs"] >= min_cap
        ]

    # fallback to top 3 by capacity
    if not cands:
        cands = sorted(
            models_data,
            key=lambda m: m.get("Capacity_lbs", 0),
            reverse=True
        )

    return cands[:3]


def generate_forklift_context(user_input: str, customer_name: str) -> str:
    """
    1) Customer Profile block from accounts.json
    2) 3 matching models (same logic as before)
    3) Raw user question appended
    """
    ctx_lines: List[str] = []

    # 1) Customer Profile
    cust_intro = get_customer_context(customer_name)
    if cust_intro:
        ctx_lines.append(cust_intro)

    # 2) Model recommendations
    matches = filter_models_for_account(user_input)
    if matches:
        for m in matches:
            ctx_lines += [
                "<span class=\"section-label\">Model:</span>",
                f"- {m.get('Model','N/A')}",
                "<span class=\"section-label\">Power:</span>",
                f"- {m.get('Power','N/A')}",
                "<span class=\"section-label\">Capacity:</span>",
                f"- {m.get('Capacity_lbs','N/A')} lbs",
                "<span class=\"section-label\">Type:</span>",
                f"- {m.get('Type','N/A')}",
                ""  # blank line
            ]
    else:
        ctx_lines.append(
            "You are a forklift expert assistant. No models matched the filters; "
            "please provide a professional recommendation based on the user's requirements."
        )
        ctx_lines.append("")

    # 3) Raw question
    ctx_lines.append(user_input)
    return "\n".join(ctx_lines)
