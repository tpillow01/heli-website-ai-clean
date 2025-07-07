import json

# Load models from JSON
with open("models.json", "r", encoding="utf-8") as f:
    models_data = json.load(f)

print(f"✅ Loaded {len(models_data)} models from JSON")


# Convert weight from kg/tons to lbs
def convert_to_lbs(capacity_raw):
    text = str(capacity_raw).lower().strip()

    if not text or text in ["n/a", "na", "none"]:
        return "N/A"

    parts = ''.join(c if c.isdigit() or c == '.' else ' ' for c in text).strip().split()
    if not parts:
        return f"{capacity_raw}"

    try:
        val = float(parts[0])
    except:
        return f"{capacity_raw}"

    if "kg" in text:
        pounds = round(val * 2.20462)
        return f"{pounds:,} lbs (converted from {val:,} kg)"
    elif "ton" in text or "t" in text:
        pounds = round(val * 2000)
        return f"{pounds:,} lbs (converted from {val} tons)"
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


# Convert meters to feet
def m_to_feet(m_value):
    try:
        meters = float(m_value)
        feet = round(meters * 3.28084, 1)
        return f"{feet} ft (converted from {meters} m)"
    except:
        return str(m_value)


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
    return filtered[:3]


def generate_forklift_context(user_input, models=None):
    if models is None:
        models = filter_models(user_input)

    if models:
        context_lines = ["Here are a few matching Heli models:"]
        for m in models:
            # Handle lift height (could be in mm or meters)
            lift_raw = m.get('LiftHeight_mm', 'N/A')
            lift_str = str(lift_raw).lower()
            if "m" in lift_str and "mm" not in lift_str:
                lift_display = m_to_feet(lift_raw)
            else:
                lift_display = mm_to_feet_inches(lift_raw)

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
            "You are a forklift expert assistant. No models matched the filters, "
            "but please provide a professional recommendation based on the user's input."
        )
