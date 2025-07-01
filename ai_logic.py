import json

# Load models from JSON
with open("models.json", "r", encoding="utf-8") as f:
    models_data = json.load(f)

print(f"✅ Loaded {len(models_data)} models from JSON")

def filter_models(user_input):
    filtered = models_data

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

def generate_forklift_context(user_input):
    filtered_data = filter_models(user_input)

    if filtered_data:
        context_lines = ["Here are a few matching Heli models:"]
        for m in filtered_data:
            context_lines += [
                "<span class=\"section-label\">Model:</span>",
                f"- {m.get('Model', 'N/A')}",
                f"- {m.get('Type', 'N/A')} forklift designed for relevant tasks.",

                "<span class=\"section-label\">Power:</span>",
                f"- {m.get('Power', 'N/A')}",

                "<span class=\"section-label\">Capacity:</span>",
                f"- {m.get('Capacity', 'N/A')}",

                "<span class=\"section-label\">Features:</span>",
                f"- {m.get('Features', 'N/A')}",

                ""
            ]
        context = "\n".join(context_lines)
    else:
        context = (
            "You are a forklift expert assistant. No models matched the filters, "
            "but please provide a professional recommendation based on the user's input."
        )

    return context
