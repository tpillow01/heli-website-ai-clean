"""
Pure helper module: account lookup + model filtering + prompt context builder
"""

import json, re, difflib
from typing import List, Dict, Any, Optional

# ── load JSON once -------------------------------------------------------
with open("accounts.json", "r", encoding="utf-8") as f:
    accounts_raw = json.load(f)

with open("models.json", "r", encoding="utf-8") as f:
    _raw = json.load(f)
    # accept either a list or {"models":[...]}
    models_raw: List[Dict[str, Any]] = _raw.get("models", _raw) if isinstance(_raw, dict) else _raw

# ── account helpers ------------------------------------------------------
def get_account(text: str) -> Optional[Dict[str, Any]]:
    """
    Find a customer record by substring first, then fuzzy match.
    """
    low = text.lower()
    for acct in accounts_raw:  # substring pass
        name = str(acct.get("Account Name", "")).lower()
        if name and name in low:
            return acct

    names = [a.get("Account Name", "") for a in accounts_raw if a.get("Account Name")]
    close = difflib.get_close_matches(text, names, n=1, cutoff=0.7)
    if close:
        return next(a for a in accounts_raw if a.get("Account Name") == close[0])
    return None

def customer_block(acct: Dict[str, Any]) -> str:
    """
    Pretty HTML-tagged customer profile for the LLM prompt.
    """
    sic = acct.get("SIC Code", "N/A")
    try:
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
    Parse '5000 lbs', '3 ton', '2.5t', '3000', '2000 kg' → pounds (float).
    """
    s = str(value or "").lower().strip()

    # kg → lbs
    kg_match = re.search(r"([\d,\.]+)\s*kg", s)
    if kg_match:
        kg = float(kg_match.group(1).replace(",", ""))
        return kg * 2.20462

    # tons (t / ton / tons) → lbs (1 ton ≈ 2000 lb)
    ton_match = re.search(r"([\d,\.]+)\s*(t|ton|tons)\b", s)
    if ton_match:
        t = float(ton_match.group(1).replace(",", ""))
        return t * 2000.0

    # lbs straight
    lb_match = re.search(r"([\d,\.]+)\s*l(b|bs)\b", s)
    if lb_match:
        return float(lb_match.group(1).replace(",", ""))

    # plain number fallback
    m = re.search(r"[\d,\.]+", s)
    if not m:
        try:
            return float(value) if value is not None else 0.0
        except Exception:
            return 0.0
    return float(m.group(0).replace(",", ""))

def _first(m: Dict[str, Any], *keys, default=None):
    for k in keys:
        if k in m and m[k] not in (None, ""):
            return m[k]
    return default

def _norm_power(v: Any) -> str:
    s = str(v or "").lower()
    if "electric" in s or "lith" in s: return "electric"
    if "diesel" in s: return "diesel"
    if "lpg" in s or "propane" in s or "gas" in s: return "lpg"
    return s or "unknown"

def _has(m: Dict[str, Any], text: str) -> bool:
    return text.lower() in json.dumps(m, ensure_ascii=False).lower()

def _parse_question(q: str) -> Dict[str, Any]:
    ql = (q or "").lower()

    # capacity signals
    cap = None
    # "5000 lb(s)"
    lb = re.search(r"(\d{3,6})\s*l(b|bs)\b", ql)
    if lb:
        cap = float(lb.group(1))
    # "x.x ton / t"
    tmatch = re.search(r"(\d+(\.\d+)?)\s*(t|ton|tons)\b", ql)
    if tmatch:
        cap = max(cap or 0.0, float(tmatch.group(1)) * 2000.0)
    # "#### kg"
    kg = re.search(r"(\d{3,6})\s*kg\b", ql)
    if kg:
        cap = max(cap or 0.0, float(kg.group(1)) * 2.20462)

    # height signals
    lift_in = None
    # feet
    ft = re.search(r"(\d{1,2})(\.\d+)?\s*(ft|foot|feet)\b", ql)
    if ft:
        lift_in = float(ft.group(1)) * 12.0
    # inches
    inch = re.search(r"(\d{2,3})\s*(in|inch|inches)\b", ql)
    if inch:
        lift_in = max(lift_in or 0.0, float(inch.group(1)))

    # aisle width
    aisle_in = None
    a_in = re.search(r"aisle[s]?\s*(of|to|around|about)?\s*(\d{2,3})\s*(in|inch|inches)\b", ql)
    if a_in:
        aisle_in = float(a_in.group(2))

    # power preference
    power = None
    if "electric" in ql or "lithium" in ql or "battery" in ql: power = "electric"
    elif "diesel" in ql: power = "diesel"
    elif "propane" in ql or "lpg" in ql or "gas" in ql: power = "lpg"

    # environment
    indoor = True if "indoor" in ql or "warehouse" in ql else None
    outdoor = True if "outdoor" in ql or "yard" in ql else None
    rough   = True if "rough" in ql or "gravel" in ql or "construction" in ql else None
    narrow  = True if "narrow" in ql or "reach" in ql or "very tight" in ql else None

    return {
        "capacity_min_lb": cap,
        "lift_height_min_in": lift_in,
        "aisle_max_in": aisle_in,
        "power_pref": power,
        "indoor": indoor,
        "outdoor": outdoor,
        "rough": rough,
        "narrow": narrow,
    }

def _model_capacity_lb(m: Dict[str, Any]) -> float:
    return _cap_val(_first(m, "Capacity_lbs", "capacity_lbs", "capacity_lb", "Capacity", "Rated Capacity (lb)", "Rated Capacity", "Load Capacity", default=0))

def _model_power(m: Dict[str, Any]) -> str:
    return _norm_power(_first(m, "Power", "power", "Fuel", "Fuel Type", "Fuel_Type", default=""))

def _model_tire(m: Dict[str, Any]) -> str:
    return str(_first(m, "Tire Type", "Tire", "Tires", "tire_type", default="")).strip()

def _model_attachments(m: Dict[str, Any]) -> str:
    v = _first(m, "Attachments", "attachments", default="")
    if isinstance(v, list):
        return ", ".join(v)
    return str(v or "")

def _model_name(m: Dict[str, Any]) -> str:
    return str(_first(m, "Model", "model", "code", "name", default="N/A")).strip()

def filter_models(user_q: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Robust rule-set:
      • parse capacity (lbs / tons / kg), lift height, aisle width, power hints
      • soft match environment (indoor/outdoor, rough, narrow)
      • never return an empty list — fallback to top by capacity
    """
    cand = list(models_raw)
    if not cand:
        return []

    parsed = _parse_question(user_q)

    # hard-ish filters
    if parsed["capacity_min_lb"]:
        need = parsed["capacity_min_lb"]
        cand = [m for m in cand if _model_capacity_lb(m) >= need]

    # aisle constraint (narrow-aisle / reach)
    if parsed["aisle_max_in"]:
        # If model has explicit aisle_min_in key, honor it. Otherwise, prefer known narrow-aisle types.
        filtered = []
        for m in cand:
            aisle_val = _first(m, "aisle_min_in", "Aisle_Min_In", "Aisle", default=None)
            if aisle_val is not None:
                try:
                    if float(str(aisle_val)) <= parsed["aisle_max_in"]:
                        filtered.append(m)
                    continue
                except Exception:
                    pass
            # heuristic: include reach/order picker/pallet stacker in narrow aisles
            if parsed["aisle_max_in"] <= 96 and (
                _has(m, "reach") or _has(m, "order picker") or _has(m, "very narrow") or _has(m, "vna")
            ):
                filtered.append(m)
        cand = filtered or cand

    # power preferences (soft if not present)
    if parsed["power_pref"]:
        p = parsed["power_pref"]
        pref = [m for m in cand if _model_power(m) == p]
        cand = pref or cand

    # environment hints (soft scoring later)

    # scoring
    scored = []
    for m in cand:
        s = 0.0
        cap = _model_capacity_lb(m)
        if parsed["capacity_min_lb"]:
            margin = max(0.0, cap - parsed["capacity_min_lb"])
            # prefer modest overkill vs massive overkill
            s += 2.0 - (margin / 4000.0)

        if parsed["power_pref"]:
            s += 1.0 if _model_power(m) == parsed["power_pref"] else -0.5

        tires = _model_tire(m).lower()
        if parsed["indoor"] is True:
            if "cushion" in tires: s += 0.5
            if "pneumatic" in tires: s -= 0.2
        if parsed["outdoor"] is True:
            if "pneumatic" in tires: s += 0.5
        if parsed["rough"] is True:
            if "rough" in tires or "pneumatic" in tires: s += 0.5

        if parsed["narrow"] is True:
            if _has(m, "reach") or "reach" in _model_name(m).lower(): s += 0.6

        # small nudge for lift height (if present)
        req_h = parsed["lift_height_min_in"]
        if req_h:
            h = _first(m, "Lift_Height_In", "lift_height_in", "Max Lift Height (in)", "Max Lift Height", default=None)
            try:
                if h and float(str(h)) >= req_h:
                    s += 0.4
            except Exception:
                pass

        scored.append((s, m))

    if not scored:
        # fallback: show something deterministic
        cand.sort(key=lambda m: (_model_capacity_lb(m), _model_name(m)))
        return cand[:limit]

    scored.sort(key=lambda t: t[0], reverse=True)
    return [m for _, m in scored[:limit]]

# ── build final prompt chunk --------------------------------------------
def generate_forklift_context(user_q: str, acct: Optional[Dict[str, Any]]) -> str:
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

    # 2) recommended models (ground the LLM on EXACT items)
    hits = filter_models(user_q)
    if hits:
        lines.append("<span class=\"section-label\">Recommended Heli Models:</span>")
        for m in hits:
            lines += [
                "<span class=\"section-label\">Model:</span>",
                f"- { _model_name(m) }",
                "<span class=\"section-label\">Power:</span>",
                f"- { _model_power(m) }",
                "<span class=\"section-label\">Capacity:</span>",
                f"- { int(_model_capacity_lb(m)) } lbs",
                "<span class=\"section-label\">Tire Type:</span>",
                f"- { _model_tire(m) or 'N/A' }",
                "<span class=\"section-label\">Attachments:</span>",
                f"- { _model_attachments(m) or 'N/A' }",
                "<span class=\"section-label\">Comparison:</span>",
                "- Similar capacity models available from Toyota or CAT are typically higher cost.\n"
            ]
    else:
        lines.append("No matching models found in the provided data.\n")

    # 3) raw question last
    lines.append(user_q)

    return "\n".join(lines)
