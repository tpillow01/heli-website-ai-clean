"""
Pure helper module: account lookup + model filtering + prompt context builder
"""

import json, re, difflib
from typing import List, Dict, Any

# ── load JSON once -------------------------------------------------------
with open("accounts.json", "r", encoding="utf-8") as f:
    accounts_raw = json.load(f)

with open("models.json", "r", encoding="utf-8") as f:
    models_raw: List[Dict[str, Any]] = json.load(f)

# ── account helpers ------------------------------------------------------
def get_account(text: str) -> Dict[str, Any] | None:
    """
    Find a customer record by substring first, then fuzzy match.
    """
    low = text.lower()
    for acct in accounts_raw:                       # substring pass
        if acct["Account Name"].lower() in low:
            return acct

    names = [a["Account Name"] for a in accounts_raw]
    close = difflib.get_close_matches(text, names, n=1, cutoff=0.7)
    if close:
        return next(a for a in accounts_raw if a["Account Name"] == close[0])
    return None

def customer_block(acct: Dict[str, Any]) -> str:
    """
    Pretty HTML‑tagged customer profile for the LLM prompt.
    """
    sic = acct.get("SIC Code", "N/A")
    try:  # normalise weird text values
        sic = str(int(str(sic).split()[0]))
    except Exception:
        pass

    return (
        "<span class=\"section-label\">Customer Profile:</span>\n"
        f"- Company: {acct.get('Account Name')}\n"
        f"- Industry: {acct.get('Industry', 'N/A')}\n"
        f"- SIC Code: {sic}\n"
        f"- Fleet Size: {acct.get('Total Company Fleet Size', 'N/A')}\n"
        f"- Truck Types: {acct.get('Truck Types at Location', 'N/A')}\n\n"
    )

# ── model helpers --------------------------------------------------------
def _cap_val(value: Any) -> float:
    """
    Parse '5000 lbs', '3000', 3000 → 3000.0
    """
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"[\d,\.]+", str(value))
    if not m:
        return 0.0
    return float(m.group(0).replace(",", ""))

def filter_models(user_q: str,
                  limit: int = 5) -> List[Dict[str, Any]]:
    """
    Very simple rule‑set:
      • if text has '#### lbs' use capacity cutoff
      • electric / diesel / narrow‑aisle / rough‑terrain keywords
    """
    cand = models_raw[:]
    q = user_q.lower()

    # capacity
    lbs = [int(n.replace(",", ""))
           for n in re.findall(r"(\d{3,5})\s*lbs?", q)]
    if lbs:
        need = max(lbs)
        cand = [m for m in cand if _cap_val(m.get("Capacity_lbs", 0)) >= need]

    # power keyword
    if "electric" in q or "lithium" in q:
        cand = [m for m in cand
                if "electric" in str(m.get("Power", "")).lower()
                or "lithium"  in str(m.get("Power", "")).lower()]
    if "diesel" in q:
        cand = [m for m in cand if "diesel" in str(m.get("Power", "")).lower()]

    # use first N deterministic alphabetical (or capacity sorted for realism)
    cand.sort(key=lambda m: (_cap_val(m.get("Capacity_lbs", 0)), m.get("Model")))
    return cand[:limit]

# ── build final prompt chunk --------------------------------------------
def generate_forklift_context(user_q: str,
                              acct: Dict[str, Any] | None) -> str:
    """
    Sections:
      (optional) Customer Profile
      Recommended Heli Models:
         for each → Model, Power, Capacity, Tire Type, Attachments, Comparison
      raw user question (so GPT sees the ask verbatim)
    """
    lines: list[str] = []

    # 1) customer profile
    if acct:
        lines.append(customer_block(acct))

    # 2) recommended models
    hits = filter_models(user_q)
    if hits:
        lines.append("<span class=\"section-label\">Recommended Heli Models:</span>")
        for m in hits:
            lines += [
                "<span class=\"section-label\">Model:</span>",
                f"- {m.get('Model','N/A')}",
                "<span class=\"section-label\">Power:</span>",
                f"- {m.get('Power','N/A')}",
                "<span class=\"section-label\">Capacity:</span>",
                f"- {m.get('Capacity_lbs','N/A')}",
                "<span class=\"section-label\">Tire Type:</span>",
                f"- {m.get('Tire Type','N/A')}",
                "<span class=\"section-label\">Attachments:</span>",
                f"- {m.get('Attachments','N/A')}",
                "<span class=\"section-label\">Comparison:</span>",
                "- Similar capacity models available from Toyota or CAT are typically higher cost.\n"
            ]
    else:
        lines.append("No matching models found in the provided data.\n")

    # 3) raw question last
    lines.append(user_q)

    return "\n".join(lines)
