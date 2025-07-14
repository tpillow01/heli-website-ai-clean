# ai_logic.py

from typing import List, Dict, Any

# —————————————————————————————————————————————————————————————————————
# Conversion helpers (unchanged)
# —————————————————————————————————————————————————————————————————————

def convert_to_lbs(capacity_raw: Any) -> str:
    text = str(capacity_raw).lower().strip()
    if not text or text in {"n/a", "na", "none"}:
        return "N/A"
    parts = ''.join(c if c.isdigit() or c == '.' else ' ' for c in text).strip().split()
    if not parts:
        return str(capacity_raw)
    try:
        val = float(parts[0])
    except ValueError:
        return str(capacity_raw)
    if "kg" in text:
        pounds = round(val * 2.20462)
        return f"{pounds:,} lbs (from {val:,} kg)"
    elif "ton" in text or "t" in text:
        pounds = round(val * 2000)
        return f"{pounds:,} lbs (from {val} tons)"
    else:
        return f"{val:,} lbs"

def mm_to_feet_inches(mm_value: Any) -> str:
    try:
        mm = float(mm_value)
        total_inches = mm / 25.4
        feet = int(total_inches // 12)
        inches = round(total_inches % 12)
        return f"{feet} ft {inches} in"
    except Exception:
        return str(mm_value)

def m_to_feet(m_value: Any) -> str:
    try:
        text = str(m_value).lower()
        num_part = ''.join(c if c.isdigit() or c == '.' else ' ' for c in text).strip().split()
        if not num_part:
            return str(m_value)
        meters = float(num_part[0])
        feet = round(meters * 3.28084, 1)
        return f"{feet} ft (from {meters} m)"
    except Exception:
        return str(m_value)

# —————————————————————————————————————————————————————————————————————
# Customer profile context (still uses accounts.json from the app)
# —————————————————————————————————————————————————————————————————————

import json
# We still load accounts inside ai_logic so you don't have to pass it in;
# you can remove this and pass accounts_data from heli_backup_ai.py if you prefer.
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
        f"- SIC Code: {profile.get('SIC Code','N/A')}",
        f"- Fleet Size: {profile.get('Total Company Fleet Size','N/A')}",
    ]
    if profile.get("Indoor Use"):
        lines.append("- Uses forklifts indoors")
    if profile.get("Outdoor Use"):
        lines.append("- Uses forklifts outdoors")
    lines.append(f"- Application: {profile.get('Application','N/A')}")
    lines.append("")  # blank line
    return "\n".join(lines)

# —————————————————————————————————————————————————————————————————————
# Model filtering & context builder (now uses passed-in model list)
# —————————————————————————————————————————————————————————————————————

def filter_models(user_input: str,
                  customer_name: str = None,
                  models_list: List[Dict[str, Any]] = None
                 ) -> List[Dict[str, Any]]:
    """
    Return a filtered sublist of models_list based on keywords in user_input
    and optionally customer profile. Always operate on the passed-in list.
    """
    if models_list is None:
        return []
    filtered = models_list[:]

    ui = user_input.lower()
    # Example keyword filters:
    if "narrow aisle" in ui:
        filtered = [m for m in filtered if "narrow" in str(m.get("Type","")).lower()]
    if "rough terrain" in ui:
        filtered = [m for m in filtered if "rough" in str(m.get("Type","")).lower()]
    if "electric" in ui:
        filtered = [m for m in filtered if "electric" in str(m.get("Power","")).lower()]
    if "3000 lb" in ui or "3,000 lb" in ui:
        def ok(cap):
            try:
                v = float(str(cap).split()[0].replace(",",""))
                return v >= 3000
            except:
                return False
        filtered = [m for m in filtered if ok(m.get("Capacity",0))]

    # You can also apply SIC/industry-based filtering here using get_customer_context logic
    return filtered[:5]

def generate_forklift_context(user_input: str,
                              customer_name: str = None,
                              models_list: List[Dict[str, Any]] = None
                             ) -> str:
    """
    Build the chunk of context (customer profile + formatted model info)
    to prepend to the user's question when calling the AI.
    """
    cust_ctx = get_customer_context(customer_name)
    models = filter_models(user_input, customer_name, models_list)

    lines = []
    if cust_ctx:
        lines.append(cust_ctx)

    if models:
        lines.append("<span class=\"section-label\">Recommended Heli Models:</span>")
        for m in models:
            lift_raw = m.get("Max Lifting Height (mm)", "N/A")
            lift_disp = (
                m_to_feet(lift_raw)
                if isinstance(lift_raw, str) and "m" in lift_raw.lower() and "mm" not in lift_raw.lower()
                else mm_to_feet_inches(lift_raw)
            )
            lines += [
                "<span class=\"section-label\">Model:</span>",
                f"- {m.get('Model Name ','N/A')}",
                f"- Power: {m.get('Power','N/A')}",
                f"- Capacity: {convert_to_lbs(m.get('Load Capacity','N/A'))}",
                "<span class=\"section-label\">Max Lift Height:</span>",
                f"- {lift_disp}",
                "",
            ]
    else:
        # no matches
        lines.append(
            "No models matched your filters—please recommend for the user's stated needs."
        )

    # always append the raw user question at the end
    lines.append(user_input)
    return "\n".join(lines)
