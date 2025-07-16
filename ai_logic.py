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
    models_data: List[Dict[str, Any]] = json.load(f)


def get_customer_context(customer: Dict[str, Any]) -> str:
    """Return Customer Profile block from the exact JSON entry."""
    if not customer:
        return ""
    lines = ["<span class=\"section-label\">Customer Profile:</span>"]
    lines.append(f"- Company: {customer.get('Account Name','N/A')}")
    lines.append(f"- Industry: {customer.get('Industry','N/A')}")
    lines.append(f"- SIC Code: {customer.get('SIC Code','N/A')}")
    lines.append(f"- Fleet Size: {customer.get('Total Company Fleet Size','N/A')}")
    lines.append(f"- Truck Types: {customer.get('Truck Types at Location','N/A')}")
    lines.append("")  # blank line
    return "\n".join(lines)


def filter_models(user_input: str) -> List[Dict[str, Any]]:
    """Your original 3‑model filter, pulling from models_data."""
    ui = user_input.lower()
    cands = models_data[:]

    # narrow aisle
    if "narrow aisle" in ui:
        cands = [m for m in cands if "narrow" in m.get("Type","").lower()]

    # rough terrain
    if "rough terrain" in ui:
        cands = [m for m in cands if "rough" in m.get("Type","").lower()]

    # electric
    if "electric" in ui:
        cands = [m for m in cands if "electric" in m.get("Power","").lower()]

    # capacity hints (e.g. "5000 lb")
    nums = [int(n.replace(",","")) for n in re.findall(r"(\d{3,6})\s*lb", ui)]
    if nums:
        min_cap = max(nums)
        def ok(m):
            cap = m.get("Capacity_lbs",0)
            return isinstance(cap,(int,float)) and cap >= min_cap
        cands = [m for m in cands if ok(m)]

    # fallback top 3 by capacity
    if not cands:
        cands = sorted(cands, key=lambda m: m.get("Capacity_lbs",0), reverse=True)

    return cands[:3]


def generate_forklift_context(user_input: str, account_name_or_obj) -> str:
    """
    1) Customer Profile (if account provided)
    2) Up to 3 matching models with full details
    3) Raw user question
    """
    # allow passing either the name or the dict
    customer = (account_name_or_obj 
                if isinstance(account_name_or_obj, dict) 
                else accounts_data.get(str(account_name_or_obj).lower()))

    ctx_parts: List[str] = []

    # 1) Profile
    profile_block = get_customer_context(customer)
    if profile_block:
        ctx_parts.append(profile_block)

    # 2) Models
    hits = filter_models(user_input)
    if hits:
        for m in hits:
            ctx_parts += [
                "<span class=\"section-label\">Model:</span>",
                f"- {m.get('Model','N/A')}",

                "<span class=\"section-label\">Power:</span>",
                f"- {m.get('Power','N/A')}",

                "<span class=\"section-label\">Capacity (lbs):</span>",
                f"- {m.get('Capacity_lbs','N/A')}",

                "<span class=\"section-label\">Dimensions (in):</span>",
                f"- H: {m.get('Height_in','N/A')}",
                f"- W: {m.get('Width_in','N/A')}",
                f"- L: {m.get('Length_in','N/A')}",

                "<span class=\"section-label\">Max Lifting Height (in):</span>",
                f"- {m.get('LiftHeight_in','N/A')}",

                ""  # blank line
            ]
    else:
        ctx_parts.append(
            "No matching models found in the provided data.\n"
        )

    # 3) Raw question
    ctx_parts.append(user_input)
    return "\n".join(ctx_parts)
