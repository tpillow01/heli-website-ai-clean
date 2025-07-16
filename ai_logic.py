# ai_logic.py

import json
import re
from typing import List, Dict, Any

# —————————————————————————————————————————————————————————————————————
# Load JSON data once
# —————————————————————————————————————————————————————————————————————
with open("accounts.json", "r", encoding="utf-8") as f:
    accounts_raw = json.load(f)
# Build a lookup if you need to fuzzy-find elsewhere
accounts_lookup = {acct["Account Name"].lower(): acct for acct in accounts_raw}

with open("models.json", "r", encoding="utf-8") as f:
    models_data: List[Dict[str, Any]] = json.load(f)


def filter_models(user_input: str) -> List[Dict[str, Any]]:
    """
    Return up to 3 models from models_data:
      • keyword filters (narrow aisle, rough terrain, electric)
      • capacity hints like “3000 lb”
      • fallback: top 3 by Capacity_lbs
    """
    ui = user_input.lower()
    cands = models_data[:]

    # keyword filters
    if "narrow aisle" in ui:
        cands = [m for m in cands if "narrow" in m.get("Type", "").lower()]
    if "rough terrain" in ui:
        cands = [m for m in cands if "rough" in m.get("Type", "").lower()]
    if "electric" in ui:
        cands = [m for m in cands if "electric" in m.get("Power", "").lower()]

    # capacity hints (lb only)
    caps = [int(n.replace(",", "")) for n in re.findall(r"(\d{3,6})\s*lbs?", ui)]
    if caps:
        min_cap = max(caps)
        cands = [
            m for m in cands
            if isinstance(m.get("Capacity_lbs"), (int, float))
               and m["Capacity_lbs"] >= min_cap
        ]

    # fallback: top 3 by Capacity_lbs
    if not cands:
        cands = sorted(
            models_data,
            key=lambda m: m.get("Capacity_lbs", 0),
            reverse=True
        )

    return cands[:3]


def generate_forklift_context(user_input: str, account: Dict[str, Any] = None) -> str:
    """
    Build the AI context:
      1) Customer Profile from the full account dict
      2) Up to 3 matching models (using models_data)
      3) Append the raw user question
    """
    lines: List[str] = []

    # 1) Customer Profile
    if account:
        lines.append("<span class=\"section-label\">Customer Profile:</span>")
        # Use the exact fields from accounts.json
        for field in ("Account Name", "Industry", "SIC Code",
                      "Total Company Fleet Size", "Truck Types at Location"):
            val = account.get(field, "N/A")
            # Rename "Account Name" to "Company" in output
            label = "Company" if field == "Account Name" else field
            lines.append(f"- {label}: {val}")
        lines.append("")  # blank line

    # 2) Model recommendations
    matches = filter_models(user_input)
    if matches:
        for m in matches:
            lines += [
                "<span class=\"section-label\">Model:</span>",
                f"- {m.get('Model','N/A')}",
                "<span class=\"section-label\">Power:</span>",
                f"- {m.get('Power','N/A')}",
                "<span class=\"section-label\">Capacity:</span>",
                f"- {m.get('Capacity_lbs','N/A')} lbs",
                "<span class=\"section-label\">Type:</span>",
                f"- {m.get('Type','N/A')}",
                ""  # blank line
            ]
    else:
        lines.append(
            "You are a forklift expert assistant. No models matched the filters; "
            "please provide a professional recommendation based on the user's requirements.\n"
        )

    # 3) Raw question at the end
    lines.append(user_input)

    return "\n".join(lines)
