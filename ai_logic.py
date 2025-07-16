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


def filter_models_for_account(question: str, account_name: str) -> List[Dict[str, Any]]:
    """
    Returns up to 3 models based on:
      1) Customer industry or truck types
      2) Question keywords (narrow aisle, rough terrain, electric)
      3) Capacity hints in the question
      4) Fallback: top 3 by Capacity_lbs
    """
    ui = question.lower()
    cands = models_data[:]

    # 1) Customer profile filter
    acct = accounts_data.get(account_name.lower()) if account_name else None
    if acct:
        industry = acct.get("Industry", "").lower()
        trucks = acct.get("Truck Types at Location", "").lower()
        matched = []
        for m in cands:
            inds = [i.lower() for i in m.get("Industries", [])]
            types = [t.lower() for t in m.get("Compatible Truck Types", [])]
            if any(ind in industry for ind in inds) or any(tr in trucks for tr in types):
                matched.append(m)
        if matched:
            cands = matched

    # 2) Keyword filters
    if "narrow aisle" in ui:
        cands = [m for m in cands if "narrow" in m.get("Type", "").lower()]
    if "rough terrain" in ui:
        cands = [m for m in cands if "rough" in m.get("Type", "").lower()]
    if "electric" in ui:
        cands = [m for m in cands if "electric" in m.get("Power", "").lower()]

    # 3) Capacity hints
    caps = [int(n.replace(",", "")) for n in re.findall(r"(\d{3,5})\s*lbs?", ui)]
    if caps:
        min_cap = max(caps)
        cands = [m for m in cands if isinstance(m.get("Capacity_lbs"), (int, float)) and m.get("Capacity_lbs") >= min_cap]

    # 4) Fallback: top 3 by capacity
    if not cands:
        cands = sorted(
            models_data,
            key=lambda m: m.get("Capacity_lbs", 0),
            reverse=True
        )

    return cands[:3]


def generate_forklift_context(user_input: str, customer_name: str) -> str:
    """
    Build the AI context:
      1) Customer Profile (if any)
      2) up to 3 matching models
      3) Raw user question
    """
    lines: List[str] = []

    # Customer Profile
    if customer_name:
        acct = accounts_data.get(customer_name.lower())
        if acct:
            lines.append("<span class=\"section-label\">Customer Profile:</span>")
            for field in ("Company", "Industry", "SIC Code", "Total Company Fleet Size", "Truck Types at Location"):
                val = acct.get(field, "N/A")
                lines.append(f"- {field}: {val}")
            lines.append("")

    # Model recommendations
    matches = filter_models_for_account(user_input, customer_name)
    if matches:
        for m in matches:
            lines += [
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
        lines.append(
            "You are a forklift expert assistant. No models matched the filters, "
            "please provide a professional recommendation based on the user's requirements."
        )

    # Append raw question
    lines.append(user_input)
    return "\n".join(lines)
