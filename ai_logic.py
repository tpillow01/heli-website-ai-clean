# ai_logic.py  ── 2025‑07‑16
import json, re, difflib
from typing import List, Dict, Any

# ─── Load JSON once ──────────────────────────────────────────────────────
with open("accounts.json", "r", encoding="utf-8") as f:
    accounts_raw = json.load(f)

accounts_data: Dict[str, Dict[str, Any]] = {
    a["Account Name"].strip().lower().replace(" ", "_"): a
    for a in accounts_raw if "Account Name" in a
}

with open("models.json", "r", encoding="utf-8") as f:
    models_data: List[Dict[str, Any]] = json.load(f)

# ─── Helpers ─────────────────────────────────────────────────────────────
def _parse_capacity(val: Any) -> float:
    """Return numeric lbs value; non‑numeric → 0.0."""
    if isinstance(val, (int, float)):
        return float(val)
    m = re.search(r"[\d,.]+", str(val))
    if not m:
        return 0.0
    try:
        return float(m.group(0).replace(",", ""))
    except Exception:
        return 0.0


def get_customer_context(name: str) -> str:
    """Format a ‘Customer Profile:’ block or empty string."""
    if not name:
        return ""
    key = name.strip().lower().replace(" ", "_")
    info = accounts_data.get(key)
    if not info:
        return ""
    ctx = [
        '<span class="section-label">Customer Profile:</span>',
        f"- Company: {info.get('Account Name','N/A')}",
        f"- Industry: {info.get('Industry','N/A')}",
        f"- SIC Code: {info.get('SIC Code','N/A')}",
        f"- Fleet Size: {info.get('Total Company Fleet Size','N/A')}",
        f"- Truck Types: {info.get('Truck Types at Location','N/A')}",
        ""
    ]
    return "\n".join(ctx)


def filter_models(user_text: str,
                  models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a *candidate* list that meets keywords / capacity."""
    txt = user_text.lower()
    cands = models[:]

    # keyword filters
    if "narrow aisle" in txt:
        cands = [m for m in cands if "narrow" in str(m.get("Type","")).lower()]
    if "rough terrain" in txt:
        cands = [m for m in cands if "rough" in str(m.get("Type","")).lower()]
    if any(k in txt for k in ("electric", "lithium", "li-ion")):
        cands = [m for m in cands if "electric" in str(m.get("Power","")).lower()
                 or "lithium"  in str(m.get("Power","")).lower()]

    # capacity filter
    weights = [int(w.replace(",", "")) for w in re.findall(r"(\d{3,5})\s*(?:lb|lbs)?", txt)]
    if weights:
        need = max(weights)
        cands = [m for m in cands if _parse_capacity(m.get("Capacity_lbs",0)) >= need]

    # exact model mention
    exact = [m for m in models if m.get("Model","").lower() in txt]
    if exact:
        return exact[:5]

    # fuzzy fallback if nothing
    if not cands:
        names = [m.get("Model","") for m in models]
        close = difflib.get_close_matches(user_text, names, 5, 0.7)
        cands = [m for m in models if m.get("Model","") in close]

    return cands[:25]        # keep a bigger pool (we’ll sort later)


# ─── Main builder ────────────────────────────────────────────────────────
def generate_forklift_context(user_input: str,
                              account: Dict[str,Any] | None) -> str:
    """Return the full string fed to GPT as the *user* message."""
    prof_block = get_customer_context(account["Account Name"]) if account else ""
    models = filter_models(user_input, models_data)

    # sort by closeness to requested capacity, if any
    weights = [int(w.replace(",", "")) for w in re.findall(r"(\d{3,5})\s*(?:lb|lbs)?",
                                                           user_input.lower())]
    if weights:
        target = max(weights)
        models.sort(key=lambda m: abs(_parse_capacity(m.get("Capacity_lbs",0)) - target))

    models = models[:5]   # final cap

    lines: List[str] = []
    if prof_block:
        lines.append(prof_block)

    if models:
        # guard‑rail so GPT doesn’t invent names
        lines.append("Choose **only** from the MODEL codes below; do not invent new names.")
        lines.append('<span class="section-label">Recommended Heli Models:</span>')
        for m in models:
            lines += [
                '<span class="section-label">Model:</span>',
                f"- {m.get('Model','N/A')}",
                '<span class="section-label">Power:</span>',
                f"- {m.get('Power','N/A')}",
                '<span class="section-label">Capacity:</span>',
                f"- {m.get('Capacity_lbs','N/A')}",
                '<span class="section-label">Tire Type:</span>',
                f"- {m.get('Tire Type','N/A')}",
                '<span class="section-label">Attachments:</span>',
                f"- {m.get('Attachments','N/A')}",
                '<span class="section-label">Comparison:</span>',
                "- N/A",
                ""
            ]
    else:
        lines.append("No matching models found in the provided data.\n")

    # raw user question at the end
    lines.append(user_input)
    return "\n".join(lines)
