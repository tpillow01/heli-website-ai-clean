# ai_logic.py
import json, re, difflib
from typing import List, Dict, Any, Optional, Tuple

# ────────────────────────────────────────────────────────────────
# Load JSON once at import
# ────────────────────────────────────────────────────────────────
with open("accounts.json", "r", encoding="utf-8") as f:
    _accounts_raw = json.load(f)

with open("models.json", "r", encoding="utf-8") as f:
    models_data: List[Dict[str, Any]] = json.load(f)

# Account lookup map keyed by normalized company name
_accounts_map = {
    a["Account Name"].strip().lower(): a for a in _accounts_raw if "Account Name" in a
}

# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────
_CAP_RE = re.compile(r"([\d\.,]+)\s*(kg|kilogram|lbs?|pounds?)?", re.I)

def _to_lbs(num: float, unit: str) -> float:
    return num * 2.20462 if unit.lower().startswith("kg") else num

def _capacity_from_str(s: str) -> float:
    """Extract first number from a capacity string & convert to lbs."""
    if not s:
        return 0.0
    m = _CAP_RE.search(str(s))
    if not m:
        return 0.0
    num = float(m.group(1).replace(",", ""))
    unit = m.group(2) or "lbs"
    return _to_lbs(num, unit)

def capacity_of(model: Dict[str, Any]) -> float:
    """Return model capacity in **pounds** (search a few common fields)."""
    for k in ("Capacity_lbs", "Load Capacity", "Capacity"):
        if k in model and model[k]:
            return _capacity_from_str(model[k])
    return 0.0

def _keyword_filter(ui: str, cand: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply keyword filters for narrow‑aisle, rough‑terrain, power, etc."""
    if "narrow aisle" in ui:
        cand = [m for m in cand if "narrow" in str(m.get("Type", "")).lower()]
    if "rough terrain" in ui:
        cand = [m for m in cand if "rough" in str(m.get("Workplace", "")).lower()
                                 or "rough" in str(m.get("Type", "")).lower()]
    if "electric" in ui or "lithium" in ui:
        cand = [m for m in cand if "electric" in str(m.get("Power", "")).lower()
                               or "lithium"  in str(m.get("Power", "")).lower()]
    if "diesel" in ui:
        cand = [m for m in cand if "diesel" in str(m.get("Power", "")).lower()]
    return cand

def _requested_capacity(ui: str) -> Optional[int]:
    """Return largest capacity mentioned by user (lbs), or None."""
    nums = [int(n.replace(",", "")) for n in re.findall(r"(\d{3,6})\s*lbs?", ui)]
    return max(nums) if nums else None

# ────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────
def get_account(company_text: str) -> Optional[Dict[str, Any]]:
    """Best‑effort substring first, then fuzzy match."""
    lc = company_text.lower()
    for name, acc in _accounts_map.items():
        if name in lc:
            return acc

    names = list(_accounts_map.keys())
    fuzzy = difflib.get_close_matches(company_text.lower(), names, n=1, cutoff=0.7)
    return _accounts_map.get(fuzzy[0]) if fuzzy else None

def get_customer_profile_block(acc: Optional[Dict[str, Any]]) -> str:
    if not acc:
        return ""
    sic = acc.get("SIC Code", "N/A")
    return (
        "<span class=\"section-label\">Customer Profile:</span>\n"
        f"- Company: {acc['Account Name']}\n"
        f"- Industry: {acc.get('Industry', 'N/A')}\n"
        f"- SIC Code: {sic}\n"
        f"- Fleet Size: {acc.get('Total Company Fleet Size', 'N/A')}\n"
        f"- Truck Types: {acc.get('Truck Types at Location', 'N/A')}\n\n"
    )

def filter_models(user_input: str) -> List[Dict[str, Any]]:
    ui = user_input.lower()
    cand = models_data[:]  # start with every model

    # keyword trims
    cand = _keyword_filter(ui, cand)

    # capacity trim
    req_cap = _requested_capacity(ui)
    if req_cap:
        cand = [m for m in cand if capacity_of(m) >= req_cap]

    # sort by closeness to requested capacity (if any), else by capacity ascending
    cand.sort(
        key=lambda m: (
            abs(capacity_of(m) - req_cap) if req_cap else capacity_of(m),
            m.get("Model")
        )
    )
    return cand[:5]  # top 5

def generate_forklift_context(user_input: str, acc: Optional[Dict[str, Any]]) -> str:
    profile = get_customer_profile_block(acc)
    models = filter_models(user_input)

    lines: List[str] = []
    if profile:
        lines.append(profile)

    if models:
        lines.append("<span class=\"section-label\">Recommended Heli Models:</span>")
        for m in models:
            lines += [
                "<span class=\"section-label\">Model:</span>",
                f"- {m.get('Model','N/A')}",
                "<span class=\"section-label\">Power:</span>",
                f"- {m.get('Power','N/A')}",
                "<span class=\"section-label\">Capacity:</span>",
                f"- {int(capacity_of(m)):,} lbs",
                ""  # blank line between models
            ]
    else:
        lines.append("No matching models found in the provided data.\n")

    lines.append(user_input)   # always end with raw question
    return "\n".join(lines)
