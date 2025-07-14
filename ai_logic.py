# ai_logic.py

# (No top‐level JSON loads here)

# Convert kg/tons to lbs
def convert_to_lbs(capacity_raw):
    text = str(capacity_raw).lower().strip()
    if not text or text in ["n/a", "na", "none"]:
        return "N/A"
    parts = ''.join(c if c.isdigit() or c == '.' else ' ' for c in text).strip().split()
    if not parts:
        return str(capacity_raw)
    try:
        val = float(parts[0])
    except:
        return str(capacity_raw)
    if "kg" in text:
        pounds = round(val * 2.20462)
        return f"{pounds:,} lbs (from {val:,} kg)"
    elif "ton" in text or "t" in text:
        pounds = round(val * 2000)
        return f"{pounds:,} lbs (from {val} tons)"
    else:
        return f"{val:,} lbs"

# Convert mm to ft/in
def mm_to_feet_inches(mm_value):
    try:
        mm = float(mm_value)
        total_inches = mm / 25.4
        feet = int(total_inches // 12)
        inches = round(total_inches % 12)
        return f"{feet} ft {inches} in"
    except:
        return str(mm_value)

# Convert meters to ft
def m_to_feet(m_value):
    try:
        text = str(m_value).lower()
        num_part = ''.join(c if c.isdigit() or c == '.' else ' ' for c in text).strip().split()
        if not num_part:
            return str(m_value)
        meters = float(num_part[0])
        feet = round(meters * 3.28084, 1)
        return f"{feet} ft (from {meters} m)"
    except:
        return str(m_value)

# Get customer profile context
# (This still loads accounts.json at top of this file, you can keep or also refactor similarly)
import json
with open("accounts.json", "r", encoding="utf-8") as f:
    accounts_raw = json.load(f)
accounts_data = {
    acct["Account Name"].strip().lower().replace(" ", "_"): acct
    for acct in accounts_raw if "Account Name" in acct
}

def get_customer_context(customer_name):
    if not customer_name:
        return ""
    key = customer_name.strip().lower().replace(" ", "_")
    profile = accounts_data.get(key)
    if not profile:
        return ""
    lines = [
        "<span class=\"section-label\">Customer Profile:</span>",
        f"- Company: {profile['Account Name']}",
        f"- Industry: {profile.get('Industry', 'N/A')}",
        f"- SIC Code: {profile.get('SIC Code', 'N/A')}",
        f"- Fleet Size: {profile.get('Total Company Fleet Size', 'N/A')}",
    ]
    if profile.get("Indoor Use"):
        lines.append("- Uses forklifts indoors")
    if profile.get("Outdoor Use"):
        lines.append("- Uses forklifts outdoors")
    lines.append(f"- Application: {profile.get('Application', 'N/A')}")
    lines.append("")
    return "\n".join(lines)

# Filter models (now takes models_list as an argument!)
def filter_models(user_input, customer_name=None, models_list=None):
    if models_list is None:
        return []
    filtered = models_list

    ui = user_input.lower()
    if "narrow aisle" in ui:
        filtered = [m for m in filtered if "narrow" in str(m.get("Type", "")).lower()]
    if "rough terrain" in ui:
        filtered = [m for m in filtered if "rough" in str(m.get("Type", "")).lower()]
    if "electric" in ui:
        filtered = [m for m in filtered if "electric" in str(m.get("Power", "")).lower()]
    if "3000 lb" in ui or "3,000 lb" in ui:
        def cap_ok(cap): 
            try:
                v = float(str(cap).split()[0].replace(",", ""))
                return v >= 3000
            except:
                return False
        filtered = [m for m in filtered if cap_ok(m.get("Capacity", 0))]

    # SIC/industry‐based filtering can go here if desired…

    return filtered[:5]

# Final context builder: now takes models_list parameter
def generate_forklift_context(user_input, customer_name=None, models_list=None):
    cust_ctx = get_customer_context(customer_name)
    models = filter_models(user_input, customer_name, models_list)

    if models:
        lines = []
        if cust_ctx:
            lines.append(cust_ctx)

        lines.append("<span class=\"section-label\">Recommended Heli Models:</span>")
        for m in models:
            lift = m.get('Max Lifting Height (mm)', 'N/A')
            lift_str = str(lift).lower()
            lift_disp = (m_to_feet(lift) 
                         if "m" in lift_str and "mm" not in lift_str 
                         else mm_to_feet_inches(lift))
            lines += [
                "<span class=\"section-label\">Model:</span>",
                f"- {m.get('Model Name ', 'N/A')}",
                f"- {m.get('Power', 'N/A')}",
                f"- {convert_to_lbs(m.get('Load Capacity', 'N/A'))}",
                "<span class=\"section-label\">Max Lift Height:</span>",
                f"- {lift_disp}",
                "",
            ]
        return "\n".join(lines)

    # Fallback
    return (
        f"{cust_ctx}\n"
        "No models matched your filters—please provide a recommendation based on user needs."
    )
