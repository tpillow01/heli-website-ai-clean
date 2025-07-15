import json
from typing import List, Dict, Any

# —————————————————————————————————————————————————————————————————————
# Load customer accounts.json once
# —————————————————————————————————————————————————————————————————————
with open("accounts.json", "r", encoding="utf-8") as f:
    accounts_raw = json.load(f)
accounts_data: Dict[str, Dict[str, Any]] = {
    acct["Account Name"].strip().lower().replace(" ", "_"): acct
    for acct in accounts_raw
    if "Account Name" in acct
}


def get_customer_context(customer_name: str) -> str:
    """
    Return a formatted Customer Profile block, or empty if no match.
    """
    if not customer_name:
        return ""
    key = customer_name.strip().lower().replace(" ", "_")
    print(f"[DEBUG] get_customer_context: lookup key = '{key}'")
    profile = accounts_data.get(key)
    if not profile:
        print(f"[DEBUG] get_customer_context: no profile for '{key}'")
        return ""

    print(f"[DEBUG] get_customer_context: found profile for '{profile['Account Name']}'")
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
    """
    Keyword‑based filtering on the exact JSON fields.
    """
    ui = user_input.lower()
    filtered = models_list[:]

    # Narrow aisle
    if "narrow aisle" in ui:
        filtered = [
            m for m in filtered
            if "narrow" in str(m.get("Type", "")).lower()
        ]

    # Rough terrain
    if "rough terrain" in ui:
        filtered = [
            m for m in filtered
            if "rough" in str(m.get("Type", "")).lower()
        ]

    # Electric / Lithium
    if "electric" in ui or "lithium" in ui:
        filtered = [
            m for m in filtered
            if "electric" in str(m.get("Power", "")).lower()
            or "lithium" in str(m.get("Power", "")).lower()
        ]

    # Exact capacity requests (e.g. “5000 lb”)
    if "5000" in ui and "lb" in ui:
        def ok(c):
            try:
                return float(str(c).split()[0].replace(",", "")) >= 5000
            except:
                return False
        filtered = [
            m for m in filtered
            if ok(m.get("Capacity_lbs", 0))
        ]

    print(f"[DEBUG] filter_models: matched {len(filtered)} models")
    return filtered[:5]


def generate_forklift_context(
    user_input: str,
    customer_name: str,
    models_list: List[Dict[str, Any]]
) -> str:
    """
    Build the chunk of context: 
      - Optional Customer Profile 
      - Recommended Heli Models block
      - Finally the raw user question
    """
    print(f"[DEBUG] generate_forklift_context: customer_name = '{customer_name}'")
    cust_ctx = get_customer_context(customer_name)
    hits = filter_models(user_input, models_list)

    lines: List[str] = []
    if cust_ctx:
        lines.append(cust_ctx)

    if hits:
        lines.append("<span class=\"section-label\">Recommended Heli Models:</span>")
        for m in hits:
            lines += [
                "<span class=\"section-label\">Model:</span>",
                f"- {m.get('Model', 'N/A')}",
                "<span class=\"section-label\">Type:</span>",
                f"- {m.get('Type', 'N/A')}",
                "<span class=\"section-label\">Power:</span>",
                f"- {m.get('Power', 'N/A')}",
                "<span class=\"section-label\">Capacity (lbs):</span>",
                f"- {m.get('Capacity_lbs', 'N/A')}",
                "<span class=\"section-label\">Dimensions (in):</span>",
                f"- H: {m.get('Height_in', 'N/A')}",
                f"- W: {m.get('Width_in', 'N/A')}",
                f"- L: {m.get('Length_in', 'N/A')}",
                "<span class=\"section-label\">Max Lifting Height (in):</span>",
                f"- {m.get('LiftHeight_in', 'N/A')}",
                ""  # blank line between models
            ]
    else:
        lines.append("No matching models found in the provided data.\n")

    # Always finish with the raw question
    lines.append(user_input)
    return "\n".join(lines)
