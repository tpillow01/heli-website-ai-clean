# ai_logic.py

import json
import re
from typing import List, Dict, Any

# —————————————————————————————————————————————————————————————————————
# Load JSON data once
# —————————————————————————————————————————————————————————————————————
with open("accounts.json", "r", encoding="utf-8") as f:
    accounts_raw = json.load(f)
accounts_data: Dict[str, Dict[str, Any]] = {
    acct["Account Name"].strip().lower().replace(" ", "_"): acct
    for acct in accounts_raw
    if "Account Name" in acct
}

with open("models.json", "r", encoding="utf-8") as f:
    models_data: List[Dict[str, Any]] = json.load(f)


def get_customer_context(customer_name: str) -> str:
    """Return a Customer Profile section, or empty if no match."""
    if not customer_name:
        return ""
    key = customer_name.strip().lower().replace(" ", "_")
    profile = accounts_data.get(key)
    if not profile:
        return ""
    raw_sic = profile.get("SIC Code", "N/A")
    try:
        sic_code = str(int(raw_sic))
    except:
        sic_code = str(raw_sic)
    lines = [
        "<span class=\"section-label\">Customer Profile:</span>",
        f"- Company: {profile['Account Name']}",
        f"- Industry: {profile.get('Industry', 'N/A')}",
        f"- SIC Code: {sic_code}",
        f"- Fleet Size: {profile.get('Total Company Fleet Size', 'N/A')}",
        f"- Truck Types: {profile.get('Truck Types at Location', 'N/A')}",
        ""  # blank line
    ]
    return "\n".join(lines)


def generate_forklift_context(user_input: str, customer_name: str) -> str:
    """
    Build the AI context:
      1) Customer Profile (if any)
      2) Full Models Data block
      3) Raw user question

    GPT will then choose the best model based on profile + requirements.
    """
    cust_ctx = get_customer_context(customer_name)

    # 2) Build a compact Models Data list for GPT
    model_lines = ["<span class=\"section-label\">Available Models:</span>"]
    for m in models_data:
        model_lines.append(
            f"- {m.get('Model','N/A')} | "
            f"Type: {m.get('Type','N/A')} | "
            f"Power: {m.get('Power','N/A')} | "
            f"Capacity_lbs: {m.get('Capacity_lbs','N/A')} | "
            f"Height_in: {m.get('Height_in','N/A')} | "
            f"LiftHeight_in: {m.get('LiftHeight_in','N/A')}"
        )
    model_block = "\n".join(model_lines)

    # 3) Assemble final context
    parts: List[str] = []
    if cust_ctx:
        parts.append(cust_ctx)
    parts.append(model_block)
    parts.append("")           # blank line
    parts.append(user_input)   # the user's actual question

    return "\n".join(parts)
