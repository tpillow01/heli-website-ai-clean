import json
import re
from typing import List, Dict, Any

# —————————————————————————————————————————————————————————————————————
# Load JSON data once
# —————————————————————————————————————————————————————————————————————
with open("accounts.json", "r", encoding="utf-8") as f:
    accounts_raw = json.load(f)
accounts_data = {
    acct["Account Name"].lower(): acct
    for acct in accounts_raw
    if "Account Name" in acct
}

with open("models.json", "r", encoding="utf-8") as f:
    models_data: List[Dict[str, Any]] = json.load(f)


def filter_models(user_input: str) -> List[Dict[str, Any]]:
    """
    Return up to 3 models from the full catalog:
      • filter by keywords (narrow aisle, rough terrain, electric)
      • filter by capacity hints (e.g. “5000 lb”)
      • fallback: top 3 by Capacity_lbs
    """
    ui = user_input.lower()
    cands = models_data[:]  # start with entire catalog

    # keyword filters
    if "narrow aisle" in ui:
        cands = [m for m in cands if "narrow" in m.get("Type", "").lower()]
    if "rough terrain" in ui:
        cands = [m for m in cands if "rough" in m.get("Type", "").lower()]
    if "electric" in ui:
        cands = [m for m in cands if "electric" in m.get("Power", "").lower()]

    # capacity hints
    nums = [int(n.replace(",", "")) for n in re.findall(r"(\d{3,6})\s*lbs?", ui)]
    if nums:
        min_cap = max(nums)
        cands = [
            m for m in cands
            if isinstance(m.get("Capacity_lbs"), (int, float))
               and m["Capacity_lbs"] >= min_cap
        ]

    # fallback sorting
    if not cands:
        cands = sorted(
            models_data,
            key=lambda m: m.get("Capacity_lbs", 0),
            reverse=True
        )

    return cands[:3]


def generate_forklift_context(user_input: str, account: Dict[str, Any] = None) -> str:
    """
    1) Customer Profile (from the JSON record)
    2) Up to 3 matching models (with full details)
    3) Raw user question
    """
    lines: List[str] = []

    # 1) Customer Profile block
    if account:
        lines.append("<span class=\"section-label\">Customer Profile:</span>")
        lines.append(f"- Company: {account.get('Account Name','N/A')}")
        lines.append(f"- Industry: {account.get('Industry','N/A')}")
        lines.append(f"- SIC Code: {account.get('SIC Code','N/A')}")
        lines.append(f"- Fleet Size: {account.get('Total Company Fleet Size','N/A')}")
        lines.append(f"- Truck Types: {account.get('Truck Types at Location','N/A')}")
        lines.append("")

    # 2) Model recommendations
    matches = filter_models(user_input)
    if matches:
        for m in matches:
            lines += [
                "<span class=\"section-label\">Model:</span>",
                f"- {m.get('Model','N/A')}",
                "<span class=\"section-label\">Power:</span>",
                f"- {m.get('Power','N/A')}",
                "<span class=\"section-label\">Capacity (lbs):</span>",
                f"- {m.get('Capacity_lbs','N/A')}",
                "<span class=\"section-label\">Type:</span>",
                f"- {m.get('Type','N/A')}",
                ""  # blank line
            ]
    else:
        lines.append("No matching models found in the provided data.\n")

    # 3) Raw user question
    lines.append(user_input)
    return "\n".join(lines)
