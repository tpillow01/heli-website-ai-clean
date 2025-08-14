"""
Pure helper module: account lookup + model filtering + prompt context builder
Grounds model picks strictly on models.json and parses user needs robustly.
"""
from __future__ import annotations
import json, re, difflib
from typing import List, Dict, Any, Tuple

# ── load JSON once -------------------------------------------------------
with open("accounts.json", "r", encoding="utf-8") as f:
    accounts_raw = json.load(f)

with open("models.json", "r", encoding="utf-8") as f:
    models_raw: List[Dict[str, Any]] = json.load(f)

# Common key aliases we’ll look for in models.json
CAPACITY_KEYS = [
    "Capacity_lbs", "capacity_lbs", "Capacity", "Rated Capacity", "Load Capacity",
    "Capacity (lbs)", "capacity", "LoadCapacity", "capacityLbs", "RatedCapacity"
]
HEIGHT_KEYS = [
    "Lift Height_in", "Max Lift Height (in)", "Lift Height", "Max Lift Height",
    "Mast Height", "lift_height_in", "LiftHeight"
]
AISLE_KEYS = [
    "Aisle_min_in", "Aisle Width_min_in", "Aisle Width (in)", "Min Aisle (in)"
]
POWER_KEYS = ["Power", "power", "Fuel", "fuel", "Drive"]
TYPE_KEYS  = ["Type", "Category", "Segment", "Class"]

# ── account helpers ------------------------------------------------------
def get_account(text: str) -> Dict[str, Any] | None:
    low = text.lower()
    for acct in accounts_raw:                       # substring pass
        name = str(acct.get("Account Name","")).lower()
        if name and name in low:
            return acct
    names = [a.get("Account Name","") for a in accounts_raw if a.get("Account Name")]
    close = difflib.get_close_matches(text, names, n=1, cutoff=0.7)
    if close:
        return next(a for a in accounts_raw if a.get("Account Name") == close[0])
    return None

def customer_block(acct: Dict[str, Any]) -> str:
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

# ── parsing helpers ------------------------------------------------------
def _to_lbs(val: float, unit: str) -> float:
    u = unit.lower()
    if "kg" in u:
        return float(val) * 2.20462
    if "ton" in u and "metric" in u:
        return float(val) * 2204.62
    if "ton" in u:
        return float(val) * 2000.0
    return float(val)

def _to_inches(val: float, unit: str) -> float:
    u = unit.lower()
    if "ft" in u or "'" in u:
        return float(val) * 12.0
    return float(val)

def _num(s: Any) -> float | None:
    if s is None:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", str(s))
    return float(m.group(0)) if m else None

def _num_from_keys(row: Dict[str,Any], keys: List[str]) -> float | None:
    for k in keys:
        if k in row and str(row[k]).strip() != "":
            v = _num(row[k])
            if v is not None:
                return v
    return None

def _normalize_capacity_lbs(row: Dict[str,Any]) -> float | None:
    # Try lbs directly
    for k in CAPACITY_KEYS:
        if k in row:
            s = str(row[k])
            # look for units
            if re.search(r"\bkg\b", s, re.I):
                v = _num(s)
                return _to_lbs(v, "kg") if v is not None else None
            if re.search(r"\btons?\b", s, re.I):
                v = _num(s)
                return _to_lbs(v, "ton") if v is not None else None
            v = _num(s)
            return v
    return None

def _normalize_height_in(row: Dict[str,Any]) -> float | None:
    for k in HEIGHT_KEYS:
        if k in row:
            s = str(row[k])
            if re.search(r"\bft\b|'", s, re.I):
                v = _num(s)
                return _to_inches(v, "ft") if v is not None else None
            v = _num(s)
            return v
    return None

def _normalize_aisle_in(row: Dict[str,Any]) -> float | None:
    for k in AISLE_KEYS:
        if k in row:
            s = str(row[k])
            if re.search(r"\bft\b|'", s, re.I):
                v = _num(s)
                return _to_inches(v, "ft") if v is not None else None
            v = _num(s)
            return v
    return None

def _text_from_keys(row: Dict[str,Any], keys: List[str]) -> str:
    for k in keys:
        v = row.get(k)
        if v:
            return str(v)
    return ""

def _is_reach_or_vna(row: Dict[str,Any]) -> bool:
    t = (_text_from_keys(row, TYPE_KEYS) + " " + str(row.get("Model",""))).lower()
    return any(word in t for word in ["reach", "vna", "order picker", "turret"]) or re.search(r"\b(cqd|rq|vna)\b", t)

def _parse_requirements(q: str) -> Dict[str,Any]:
    ql = q.lower()

    # capacity: lbs / tons / kg
    cap_lbs = None
    # "5000 lb", "5,000 lbs"
    m = re.findall(r"(\d[\d,\.]*)\s*(?:lb|lbs)\b", ql)
    if m:
        cap_lbs = _to_lbs(float(m[-1].replace(",","")), "lb")
    # "3 ton", "3.5 tons"
    if cap_lbs is None:
        m = re.findall(r"(\d[\d,\.]*)\s*tons?\b", ql)
        if m:
            cap_lbs = _to_lbs(float(m[-1].replace(",","")), "ton")
    # "2000 kg"
    if cap_lbs is None:
        m = re.findall(r"(\d[\d,\.]*)\s*kg\b", ql)
        if m:
            cap_lbs = _to_lbs(float(m[-1].replace(",","")), "kg")

    # height: ft/in
    height_in = None
    m = re.findall(r"(\d[\d,\.]*)\s*(?:ft|feet|')\b", ql)
    if m:
        height_in = _to_inches(float(m[-1].replace(",","")), "ft")
    if height_in is None:
        m = re.findall(r"(\d[\d,\.]*)\s*(?:in|\"|inches)\b", ql)
        if m and "aisle" not in ql[m[-1].start() if hasattr(m[-1],'start') else 0:]:
            height_in = float(m[-1].replace(",",""))

    # aisle: look for "aisle" nearby
    aisle_in = None
    m = re.search(r"(?:aisle|aisles)[^\d]{0,10}(\d[\d,\.]*)\s*(?:in|\"|inches|ft|')", ql)
    if m:
        raw = m.group(1)
        unit = m.group(0)
        if "ft" in unit or "'" in unit:
            aisle_in = _to_inches(float(raw.replace(",","")), "ft")
        else:
            aisle_in = float(raw.replace(",",""))

    power_pref = None
    if "electric" in ql or "lithium" in ql:
        power_pref = "electric"
    elif "diesel" in ql:
        power_pref = "diesel"
    elif "lpg" in ql or "propane" in ql:
        power_pref = "lpg"

    indoor = "indoor" in ql or "warehouse" in ql
    outdoor = "outdoor" in ql or "yard" in ql or "rough" in ql
    narrow  = "narrow aisle" in ql or "very narrow" in ql or (aisle_in is not None and aisle_in <= 96)

    tire_pref = "cushion" if "cushion" in ql else ("pneumatic" if "pneumatic" in ql else None)

    return dict(
        cap_lbs=cap_lbs, height_in=height_in, aisle_in=aisle_in,
        power_pref=power_pref, indoor=indoor, outdoor=outdoor,
        narrow=narrow, tire_pref=tire_pref
    )

def _capacity_of(row: Dict[str,Any]) -> float | None:
    return _normalize_capacity_lbs(row)

def _height_of(row: Dict[str,Any]) -> float | None:
    return _normalize_height_in(row)

def _aisle_of(row: Dict[str,Any]) -> float | None:
    return _normalize_aisle_in(row)

def _power_of(row: Dict[str,Any]) -> str:
    return _text_from_keys(row, POWER_KEYS).lower()

def _tire_of(row: Dict[str,Any]) -> str:
    return str(row.get("Tire Type","") or row.get("Tires","")).lower()

def _safe_model_name(m: Dict[str, Any]) -> str:
    for k in ("Model","model","code","name","Code"):
        if m.get(k):
            return str(m[k]).strip()
    return "N/A"

# ── model filtering & ranking -------------------------------------------
def filter_models(user_q: str, limit: int = 5) -> List[Dict[str, Any]]:
    want = _parse_requirements(user_q)
    cap_need = want["cap_lbs"]
    aisle_need = want["aisle_in"]
    power_pref = want["power_pref"]
    narrow     = want["narrow"]
    tire_pref  = want["tire_pref"]
    height_need= want["height_in"]

    scored: List[Tuple[float, Dict[str,Any]]] = []

    for m in models_raw:
        cap = _capacity_of(m) or 0.0
        powr = _power_of(m)
        tire = _tire_of(m)
        ais  = _aisle_of(m)
        hgt  = _height_of(m)
        reach_like = _is_reach_or_vna(m)

        # Hard filters
        if cap_need and cap > 0 and cap < cap_need:
            continue
        if aisle_need and ais and ais > aisle_need:
            continue
        if narrow and not reach_like and aisle_need and not ais:
            # if they need narrow and model isn't reach/VNA and lacks aisle spec → lightly penalize later
            pass

        # Score
        s = 0.0

        # capacity closeness: modest overage OK, huge overkill penalized
        if cap_need and cap:
            over = (cap - cap_need) / cap_need
            if over >= 0:
                s += 2.0 - min(2.0, over)   # 0..2 → smaller overkill gets more points
            else:
                s -= 5.0                     # should've been filtered, but just in case

        # power match
        if power_pref:
            s += 1.2 if power_pref in powr else -0.6

        # tire preference
        if tire_pref:
            s += 0.6 if tire_pref in tire else -0.2

        # aisle/narrow
        if aisle_need:
            if ais:
                # the smaller the required aisle, the better if model meets it
                s += 0.8 if ais <= aisle_need else -1.0
            else:
                s += 0.6 if (narrow and reach_like) else 0.0
        elif narrow:
            s += 0.8 if reach_like else -0.2

        # lift height: bonus if model meets it
        if height_need and hgt:
            s += 0.4 if hgt >= height_need else -0.3

        # small tie-breakers
        s += 0.05  # prevent exact ties from dropping

        scored.append((s, m))

    if not scored:
        # fall back: just pick by capacity (descending) to avoid empty list
        fallback = sorted(models_raw, key=lambda r: (_capacity_of(r) or 0.0), reverse=True)
        return fallback[:limit]

    ranked = sorted(scored, key=lambda t: t[0], reverse=True)
    return [m for _, m in ranked[:limit]]

# ── build final prompt chunk --------------------------------------------
def generate_forklift_context(user_q: str, acct: Dict[str, Any] | None) -> str:
    lines: list[str] = []
    if acct:
        lines.append(customer_block(acct))

    hits = filter_models(user_q)
    if hits:
        lines.append("<span class=\"section-label\">Recommended Heli Models:</span>")
        for m in hits:
            lines += [
                "<span class=\"section-label\">Model:</span>",
                f"- {_safe_model_name(m)}",
                "<span class=\"section-label\">Power:</span>",
                f"- {_text_from_keys(m, POWER_KEYS) or 'N/A'}",
                "<span class=\"section-label\">Capacity:</span>",
                f"- {(_normalize_capacity_lbs(m) or 'N/A')}",
                "<span class=\"section-label\">Tire Type:</span>",
                f"- {_tire_of(m) or 'N/A'}",
                "<span class=\"section-label\">Attachments:</span>",
                f"- {str(m.get('Attachments','N/A'))}",
                "<span class=\"section-label\">Comparison:</span>",
                "- Similar capacity models available from Toyota or CAT are typically higher cost.\n"
            ]
    else:
        lines.append("No matching models found in the provided data.\n")

    lines.append(user_q)
    return "\n".join(lines)

# --- NEW: expose selected list + ALLOWED block for strict grounding ------
def select_models_for_question(user_q: str, k: int = 5):
    hits = filter_models(user_q, limit=k)
    allowed = []
    for m in hits:
        nm = _safe_model_name(m)
        if nm != "N/A":
            allowed.append(nm)
    return hits, allowed

def allowed_models_block(allowed: list[str]) -> str:
    if not allowed:
        return "ALLOWED MODELS:\n(none – say 'No exact match from our lineup.')"
    return "ALLOWED MODELS:\n" + "\n".join(f"- {x}" for x in allowed)
