import json

# Only load this if user doesn't provide models explicitly
with open("models.json", "r", encoding="utf-8") as f:
    models_data = json.load(f)

print(f"✅ Loaded {len(models_data)} models from JSON")


def convert_to_lbs(capacity_raw):
    text = str(capacity_raw).lower().strip()

    # Extract numeric part
    num = ''.join(c if c.isdigit() or c == '.' else ' ' for c in text).strip().split()[0]
    try:
        val = float(num)
    except:
        return f"{capacity_raw}"  # fallback if invalid

    if "kg" in text:
        pounds = round(val * 2.20462)
        return f"{pounds:,} lbs (converted from {val:,} kg)"
    elif "ton" in text or "t" in text:
        pounds = round(val * 2000)
        return f"{pounds:,} lbs (converted from {val} tons)"
    else:
        return f"{val:,} lbs"  # assume already in pounds


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
            context_lines += [
                "<span class=\"section-label\">Model:</span>",
                f"- {m.get('Model', 'N/A')}",
                f"- {m.get('Type', 'N/A')} forklift designed for relevant tasks.",

                "<span class=\"section-label\">Power:</span>",
                f"- {m.get('Power', 'N/A')}",

                "<span class=\"section-label\">Capacity:</span>",
                f"- {convert_to_lbs(m.get('Capacity', 'N/A'))}",

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
