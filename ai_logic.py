# ai_logic.py

import json
from typing import List, Dict, Any

# —————————————————————————————————————————————————————————————————————
# Load customer accounts.json just once
# —————————————————————————————————————————————————————————————————————
with open("accounts.json", "r", encoding="utf-8") as f:
    accounts_raw = json.load(f)
accounts_data = {
    acct["Account Name"].strip().lower().replace(" ", "_"): acct
    for acct in accounts_raw
    if "Account Name" in acct
}

def get_customer_context(customer_name: str) -> str:
    if not customer_name:
        return ""
    key = customer_name.strip().lower().replace(" ", "_")
    profile = accounts_data.get(key)
    if not profile:
        return ""
    lines = [
        "<span class=\"section-label\">Customer Profile:</span>",
        f"- Company: {profile['Account Name']}",
        f"- Industry: {profile.get('Industry','N/A')}",
        f"- Fleet Size: {profile.get('Total Company Fleet Size','N/A')}",
        f"- Truck Types: {profile.get('Truck Types at Location','N/A')}"
    ]
    lines.append("")  # blank line
    return "\n".join(lines)


# —————————————————————————————————————————————————————————————————————
# Keyword-based filter of models_list
# —————————————————————————————————————————————————————————————————————
def filter_models(
    user_input: str,
    customer_name: str = None,
    models_list: List[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    if models_list is None:
        return []
    ui = user_input.lower()
    filtered = models_list[:]

    # Example filters—add/remove as needed
    if "narrow aisle" in ui:
        filtered = [m for m in filtered if "narrow" in str(m.get("Type","")).lower()]
    if "rough terrain" in ui:
        filtered = [m for m in filtered if "rough" in str(m.get("Type","")).lower()]
    if "electric" in ui:
        filtered = [m for m in filtered if "electric" in str(m.get("Power","")).lower()]
    if "3000 lb" in ui or "3,000 lb" in ui:
        def cap_ok(c):
            try:
                return float(str(c).split()[0].replace(",", "")) >= 3000
            except:
                return False
        filtered = [m for m in filtered if cap_ok(m.get("Load Capacity (lbs)", 0))]

    return filtered[:5]


# —————————————————————————————————————————————————————————————————————
# Build the AI prompt context
# —————————————————————————————————————————————————————————————————————
def generate_forklift_context(
    user_input: str,
    customer_name: str = None,
    models_list: List[Dict[str, Any]] = None
) -> str:
    cust_ctx = get_customer_context(customer_name)
    models = filter_models(user_input, customer_name, models_list)

    lines: List[str] = []
    if cust_ctx:
        lines.append(cust_ctx)

    if models:
        for m in models:
            lines.append("<span class=\"section-label\">Model:</span>")
            lines.append(f"- Model: {m.get('Model Name','N/A')}")

            lines.append("<span class=\"section-label\">Power:</span>")
            lines.append(f"- {m.get('Power','N/A')}")

            lines.append("<span class=\"section-label\">Capacity:</span>")
            lines.append(f"- {m.get('Load Capacity (lbs)','N/A')}")

            lines.append("<span class=\"section-label\">Dimensions (in):</span>")
            lines.append(f"- Height: {m.get('Overall Height (in)','N/A')}")
            lines.append(f"- Width: {m.get('Overall Width (in)','N/A')}")
            lines.append(f"- Length: {m.get('Overall Length (in)','N/A')}")

            lines.append("<span class=\"section-label\">Max Lifting Height (in):</span>")
            lines.append(f"- {m.get('Max Lifting Height (in)','N/A')}")

            lines.append("")  # blank line between models
    else:
        lines.append("No matching models found in the provided data.")

    # finally, include the raw user question
    lines.append(user_input)
    return "\n".join(lines)
