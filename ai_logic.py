import json

# Load forklift models
with open("models.json", "r", encoding="utf-8") as f:
    models_data = json.load(f)

# Load customer accounts
with open("accounts.json", "r", encoding="utf-8") as f:
    accounts_data = json.load(f)

print(f"✅ Loaded {len(models_data)} models from JSON")

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

# Convert meters to ft (if stored as "3m" or "3.5 m")
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
def get_customer_context(customer_name):
    if not customer_name:
        return ""
    key = customer_name.strip().lower().replace(" ", "_")
    profile = accounts_data.get(key)
    if not profile:
        return ""

    lines = [
        f"<span class=\"section-label\">Customer Profile:</span>",
        f"- Company: {profile['company_name']}",
        f"- Industry: {profile['industry']}",
        f"- Fleet Size: {profile['fleet_size']}",
    ]
    if profile.get("indoor_use"):
        lines.append("- Uses forklifts indoors")
    if profile.get("outdoor_use"):
        lines.append("- Uses forklifts outdoors")
    lines.append(f"- Application: {profile['application']}")
    lines.append("")  # Add a break before models
    return "\n".join(lines)

# Filter models from user input
def filter_models(user_input, models_list=None):
    if models_list is None:
        models_list = models_data

    filtered = models_list

    if "narrow aisle" in user_input.lower():
        filtered = [m for m in filtered if "narrow" in m.get("Type", "").lower()]

    if "rough terrain" in user_input.lower():
        filtered = [m for m in filtered if "rough" in m.get("Type", "").lower()]

    if "electric" in user_input.lower():
        filtered = [m for m in filtered if "electric" in m.get("Power", "").lower()]

    if "3000 lb" in user_input.lower() or "3,000 lb" in user_input.lower():
        def capacity_ok(cap):
            try:
                cap_val = float(str(cap).split()[0].replace(",", ""))
                return cap_val >= 3000
            except:
                return False
        filtered = [m for m in filtered if capacity_ok(m.get("Capacity", 0))]

    print(f"📌 Filtered models: {filtered}")
    return filtered[:3]  # Limit to 3

# Final context builder
def generate_forklift_context(user_input, customer_name=None):
    customer_context = get_customer_context(customer_name)
    models = filter_models(user_input)

    if models:
        context_lines = []
        if customer_context:
            context_lines.append(customer_context)

        context_lines.append("<span class=\"section-label\">Recommended Heli Models:</span>")
        for m in models:
            lift_raw = m.get('LiftHeight_mm', 'N/A')
            lift_str = str(lift_raw).lower()
            lift_display = m_to_feet(lift_raw) if "m" in lift_str and "mm" not in lift_str else mm_to_feet_inches(lift_raw)

            context_lines += [
                "<span class=\"section-label\">Model:</span>",
                f"- {m.get('Model', 'N/A')}",
                f"- {m.get('Type', 'N/A')} forklift designed for relevant tasks.",

                "<span class=\"section-label\">Power:</span>",
                f"- {m.get('Power', 'N/A')}",

                "<span class=\"section-label\">Capacity:</span>",
                f"- {convert_to_lbs(m.get('Capacity', 'N/A'))}",

                "<span class=\"section-label\">Dimensions:</span>",
                f"- Height: {mm_to_feet_inches(m.get('Height_mm', 'N/A'))}",
                f"- Width: {mm_to_feet_inches(m.get('Width_mm', 'N/A'))}",
                f"- Length: {mm_to_feet_inches(m.get('Length_mm', 'N/A'))}",
                f"- Max Lift Height: {lift_display}",

                "<span class=\"section-label\">Features:</span>",
                f"- {m.get('Features', 'N/A')}",
                ""
            ]
        return "\n".join(context_lines)
    else:
        return (
            f"{customer_context}\n"
            "You are a forklift expert assistant. No models matched the filters, "
            "but please provide a professional recommendation based on the user's input."
        )
