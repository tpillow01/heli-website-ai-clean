# ai_logic.py

import json
import re
import difflib
from typing import List, Dict, Any

# —————————————————————————————————————————————————————————————————————
# Load JSON data once
# —————————————————————————————————————————————————————————————————————
with open("accounts.json", "r", encoding="utf-8") as f:
    accounts_raw = json.load(f)
accounts_data: Dict[str, Dict[str, Any]] = {
    acct["Account Name"].strip().lower().replace(" ", "_"): acct
    for acct in accounts_raw
    if "Account Name" in acct
}

with open("models.json", "r", encoding="utf-8") as f:
    models_data: List[Dict[str, Any]] = json.load(f)


def get_customer_context(customer_name: str) -> str:
    if not customer_name:
        return ""
    key = customer_name.strip().lower().replace(" ", "_")
    profile = accounts_data.get(key)
    if not profile:
        return ""
    raw_sic = profile.get("SIC Code", "N/A")
    try:
        sic_code = str(int(raw_sic))
    except:
        sic_code = str(raw_sic)
    lines = [
        "<span class=\"section-label\">Customer Profile:</span>",
        f"- Company: {profile['Account Name']}",
        f"- Industry: {profile.get('Industry','N/A')}",
        f"- SIC Code: {sic_code}",
        f"- Fleet Size: {profile.get('Total Company Fleet Size','N/A')}",
        f"- Truck Types: {profile.get('Truck Types at Location','N/A')}",
        ""
    ]
    return "\n".join(lines)


def _parse_capacity(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value)
    m = re.search(r"[\d,.]+", s)
    if not m:
        return 0.0
    return float(m.group(0).replace(",", ""))


def filter_models(user_input: str, models_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ui = user_input.lower()
    cands = models_list[:]

    # capacity hint
    caps = [int(n.replace(",", "")) for n in re.findall(r"(\d{3,5})\s*(?:lb|lbs)?", ui)]
    if caps:
        min_cap = max(caps)
        cands = [m for m in cands if _parse_capacity(m.get("Capacity_lbs",0))>=min_cap]

    # keywords
    if "narrow aisle" in ui:
        cands = [m for m in cands if "narrow" in m.get("Type","").lower()]
    if "rough terrain" in ui:
        cands = [m for m in cands if "rough" in m.get("Type","").lower()]
    if "electric" in ui or "lithium" in ui:
        cands = [m for m in cands if "electric" in m.get("Power","").lower() or "lithium" in m.get("Power","").lower()]

    # exact model mention
    exact = [m for m in models_list if m.get("Model","").lower() in ui]
    if exact:
        return exact[:10]

    # fuzzy model mention
    if not cands:
        names = [m["Model"] for m in models_list]
        close = difflib.get_close_matches(user_input, names, n=10, cutoff=0.6)
        return [m for m in models_list if m["Model"] in close]

    # fallback top by capacity
    if not cands:
        cands = sorted(models_list, key=lambda m: _parse_capacity(m.get("Capacity_lbs",0)), reverse=True)

    return cands[:10]


def generate_forklift_context(user_input: str, customer_name: str) -> str:
    cust_ctx = get_customer_context(customer_name)
    hits = filter_models(user_input, models_data)

    # Build candidate block
    model_block = ["<span class=\"section-label\">Candidate Models:</span>"]
    for m in hits:
        model_block.append(
            f"- {m['Model']} | Type: {m.get('Type','N/A')} | Power: {m.get('Power','N/A')} | "
            f"Capacity: {m.get('Capacity_lbs','N/A')} lbs"
        )
    block = "\n".join(model_block)

    parts = []
    if cust_ctx:
        parts.append(cust_ctx)
    parts.append(block)
    parts.append("")        # blank line
    parts.append(user_input)

    return "\n".join(parts)
