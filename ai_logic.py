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


def filter_models(user_input: str, models_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Comprehensive filter over all models.json entries."""
    ui = user_input.lower()
    candidates = models_list[:]

    # keyword filters
    if "narrow aisle" in ui:
        candidates = [m for m in candidates if "narrow" in str(m.get("Type", "")).lower()]

    if "rough terrain" in ui:
        candidates = [m for m in candidates if "rough" in str(m.get("Type", "")).lower()]

    if "electric" in ui or "lithium" in ui:
        candidates = [
            m for m in candidates
            if "electric" in str(m.get("Power", "")).lower()
            or "lithium" in str(m.get("Power", "")).lower()
        ]

    # capacity filter for ANY “### lb” or bare numbers
    weights = [int(n.replace(",", "")) for n in re.findall(r"(\d{3,5})\s*(?:lb|lbs)?", ui)]
    if weights:
        min_cap = max(weights)
        candidates = [m for m in candidates if float(m.get("Capacity_lbs", 0)) >= min_cap]

    # exact model‑name mention
    exact_hits = [m for m in models_list if m.get("Model", "").lower() in ui]
    if exact_hits:
        candidates = exact_hits

    # fuzzy match on model names if nothing else matched
    if not candidates:
        all_names = [m.get("Model", "") for m in models_list]
        close = difflib.get_close_matches(user_input, all_names, n=5, cutoff=0.6)
        candidates = [m for m in models_list if m.get("Model", "") in close]

    # finally cap at 5
    return candidates[:5]


def generate_forklift_context(user_input: str, customer_name: str) -> str:
    """
    Build the AI context:
      1) Customer Profile
      2) Recommended Heli Models
      3) Raw user question
    """
    cust_ctx = get_customer_context(customer_name)
    hits = filter_models(user_input, models_data)

    lines: List[str] = []
    if cust_ctx:
        lines.append(cust_ctx)

    if hits:
        lines.append("<span class=\"section-label\">Recommended Heli Models:</span>")
        for m in hits:
            lines += [
                "<span class=\"section-label\">Model:</span>",
                f"- {m.get('Model','N/A')}",
                "<span class=\"section-label\">Type:</span>",
                f"- {m.get('Type','N/A')}",
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
        lines.append("No matching models found in the provided data.\n")

    lines.append(user_input)
    return "\n".join(lines)
