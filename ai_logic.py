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


def _parse_capacity(value: Any) -> float:
    """
    Normalize capacity values into a float.
    Handles ints, floats, and strings like "1700 lbs", "2,500", etc.
    Non-numeric or missing → 0.0
    """
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value)
    m = re.search(r"[\d,.]+", s)
    if not m:
        return 0.0
    num = m.group(0).replace(",", "")
    try:
        return float(num)
    except:
        return 0.0


def filter_models(user_input: str, models_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Comprehensive filter over all models.json entries."""
    ui = user_input.lower()
    candidates = models_list[:]

    # 1) Capacity filter if user mentions “### lb”
    weights = [int(n.replace(",", "")) for n in re.findall(r"(\d{3,5})\s*(?:lb|lbs)?", ui)]
    if weights:
        min_cap = max(weights)
        candidates = [
            m for m in candidates
            if _parse_capacity(m.get("Capacity_lbs", 0)) >= min_cap
        ]

    # 2) Keyword filters
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

    # 3) Exact model-code mention
    exact_hits = [m for m in models_list if m.get("Model", "").lower() in ui]
    if exact_hits:
        return exact_hits[:5]

    # 4) Fuzzy match on model codes if still empty
    if not candidates:
        all_names = [m.get("Model", "") for m in models_list]
        close = difflib.get_close_matches(user_input, all_names, n=5, cutoff=0.6)
        if close:
            return [m for m in models_list if m.get("Model", "") in close]

    # 5) If still nothing (i.e. user gave no hints), default to top 5 by capacity
    if not candidates:
        candidates = sorted(
            models_list,
            key=lambda m: _parse_capacity(m.get("Capacity_lbs", 0)),
            reverse=True
        )

    # Finally, cap at 5
    return candidates[:5]


def generate_forklift_context(user_input: str, customer_name: str) -> str:
    """
    Build the AI context:
      1) Customer Profile (if any)
      2) <span class="section-label">Recommended Heli Models:</span>
         – for each model, list Model, Power, Capacity, Tire Type, Attachments, Comparison
      3) Raw user question
    """
    cust_ctx = get_customer_context(customer_name)
    hits = filter_models(user_input, models_data)

    # Define the exact sections your system prompt expects
    SECTION_FIELDS = [
        ("Model", "Model"),
        ("Power", "Power"),
        ("Capacity", "Capacity_lbs"),
        ("Tire Type", "Tire Type"),
        ("Attachments", "Attachments"),
        ("Comparison", "Comparison"),
    ]

    lines: List[str] = []
    if cust_ctx:
        lines.append(cust_ctx)

    if hits:
        lines.append("<span class=\"section-label\">Recommended Heli Models:</span>")
        for m in hits:
            for label, field in SECTION_FIELDS:
                lines.append(f"<span class=\"section-label\">{label}:</span>")
                if field == "Capacity_lbs":
                    cap = _parse_capacity(m.get(field, 0))
                    lines.append(f"- {cap} lbs")
                else:
                    lines.append(f"- {m.get(field, 'N/A')}")
            lines.append("")  # blank line between models
    else:
        lines.append("No matching models found in the provided data.\n")

    # Always finish with the raw user question
    lines.append(user_input)
    return "\n".join(lines)
