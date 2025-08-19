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

# --- robust capacity intent parser --------------------------------------
def _parse_capacity_lbs_intent(text: str) -> tuple[int | None, int | None]:
    """
    Returns (min_required_lb, max_allowed_lb). We use the min for filtering later.
    Understands:
      - “7000 pounds”, “7,000 lbs”, “7k lb”
      - “load of 5,000 pounds”, “handle 5 ton”, “payload 3.5T/tonne/mt”
      - ranges “3k–5k”, “3000-5000 lbs”, “between 3,000 and 5,000”
      - bounds “up to 6000”, “max 6000”, “at least 5000”, “minimum 5k”
    """
    if not text:
        return (None, None)

    t = text.lower().replace("–", "-").replace("—", "-")

    UNIT_LB     = r'(?:lb|lbs|pound|pounds)'
    UNIT_TONNE  = r'(?:tonne|tonnes|metric\s*ton(?:s)?|(?<!f)t\b)'  # bare 't' but not part of 'ft'
    UNIT_TON    = r'(?:ton|tons)'
    KNUM        = r'(\d+(?:\.\d+)?)\s*k\b'
    NUM         = r'(\d[\d,\.]*)'
    LOAD_WORDS  = r'(?:capacity|load|loads|payload|rating|lift|handle|carry)'

    def _n(s: str) -> float:
        return float(s.replace(",", ""))

    # ranges “3k-5k”
    m = re.search(rf'{KNUM}\s*-\s*{KNUM}', t)
    if m:
        lo = int(round(_n(m.group(1)) * 1000))
        hi = int(round(_n(m.group(2)) * 1000))
        return (min(lo, hi), max(lo, hi))

    # ranges “3000-5000 (lbs optional)”
    m = re.search(rf'{NUM}\s*-\s*{NUM}\s*(?:{UNIT_LB})?', t)
    if m:
        a = int(round(_n(m.group(1))))
        b = int(round(_n(m.group(2))))
        return (min(a, b), max(a, b))

    # “between 3,000 and 5,000”
    m = re.search(rf'between\s+{NUM}\s+and\s+{NUM}', t)
    if m:
        a = int(round(_n(m.group(1))))
        b = int(round(_n(m.group(2))))
        return (min(a, b), max(a, b))

    # bounds
    m = re.search(rf'(?:up to|max(?:imum)?)\s+{NUM}\s*(?:{UNIT_LB})?', t)
    if m:
        return (None, int(round(_n(m.group(1)))))

    m = re.search(rf'(?:at least|minimum|min)\s+{NUM}\s*(?:{UNIT_LB})?', t)
    if m:
        return (int(round(_n(m.group(1)))), None)

    # singles with units
    m = re.search(rf'{KNUM}\s*(?:{UNIT_LB})?\b', t)  # “7k lb(s)”
    if m:
        return (int(round(_n(m.group(1)) * 1000)), None)

    m = re.search(rf'{NUM}\s*{UNIT_LB}\b', t)        # “7000 pounds/lbs”
    if m:
        return (int(round(_n(m.group(1)))), None)

    m = re.search(rf'{NUM}\s*{UNIT_TONNE}\b', t)     # “3.5 tonne / metric ton / 3.5t”
    if m:
        return (int(round(_n(m.group(1)) * 2204.62)), None)

    m = re.search(rf'{NUM}\s*{UNIT_TON}\b', t)       # “5 ton(s)” (US)
    if m:
        return (int(round(_n(m.group(1)) * 2000)), None)

    # “payload/load/capacity … 7k/5000” (units omitted)
    m = re.search(rf'{LOAD_WORDS}[^0-9\-]*{KNUM}', t)
    if m:
        return (int(round(_n(m.group(1)) * 1000)), None)
    m = re.search(rf'{LOAD_WORDS}[^0-9\-]*{NUM}', t)
    if m:
        return (int(round(_n(m.group(1)))), None)

    # bare 4–5 digit number → assume lb
    m = re.search(r'\b(\d{4,5})\b', t)
    if m:
        return (int(m.group(1)), None)

    return (None, None)

def _parse_requirements(q: str) -> Dict[str,Any]:
    ql = q.lower()

    # capacity from robust intent parser
    cap_min, cap_max = _parse_capacity_lbs_intent(ql)
    cap_lbs = cap_min  # we use minimum-required capacity for filtering logic downstream

    # height: ft/in + synonyms (reach/clearance/mast)
    height_in = None
    m = re.findall(r"(\d[\d,\.]*)\s*(?:ft|feet|')\b", ql)
    if m:
        try:
            height_in = _to_inches(float(m[-1].replace(",","")), "ft")
        except Exception:
            height_in = None
    if height_in is None:
        m = re.findall(r"(\d[\d,\.]*)\s*(?:in|\"|inches)\b", ql)
        if m:
            try:
                height_in = float(m[-1].replace(",",""))
            except Exception:
                height_in = None
    if height_in is None:
        m = re.search(r"(?:lift|raise|reach|height|clearance|mast)\D{0,12}(\d[\d,\.]*)\s*(ft|feet|'|in|\"|inches)", ql)
        if m:
            raw, unit = m.group(1), m.group(2)
            try:
                height_in = _to_inches(float(raw.replace(",","")), "ft") if unit in ("ft","feet","'") \
                            else float(raw.replace(",",""))
            except Exception:
                height_in = None

    # aisle: include right-angle aisle / stacking variants
    aisle_in = None
    m = re.search(r"(?:aisle|aisles|aisle width)\D{0,12}(\d[\d,\.]*)\s*(?:in|\"|inches|ft|')", ql)
    if m:
        raw = m.group(1)
        unit = m.group(0)
        try:
            if "ft" in unit or "'" in unit:
                aisle_in = _to_inches(float(raw.replace(",","")), "ft")
            else:
                aisle_in = float(raw.replace(",",""))
        except Exception:
            aisle_in = None
    if aisle_in is None:
        m = re.search(r"(?:right[-\s]?angle(?:\s+aisle|\s+stack(?:ing)?)?|ra\s*aisle|ras)\D{0,12}(\d[\d,\.]*)\s*(in|\"|inches|ft|')", ql)
        if m:
            raw, unit = m.group(1), m.group(2)
            try:
                aisle_in = _to_inches(float(raw.replace(",","")), "ft") if unit in ("ft","'") \
                           else float(raw.replace(",",""))
            except Exception:
                aisle_in = None

    # power preferences (li-ion / battery / EV; LP/LPG/propane; diesel)
    power_pref = None
    if re.search(r"\bli[\-\s]?ion\b|\bbattery\b|\bev\b", ql) or "lithium" in ql or "electric" in ql:
        power_pref = "electric"
    elif re.search(r"\blp\b|\blp[-\s]?gas\b|\blpg\b|\bpropane\b", ql):
        power_pref = "lpg"
    elif "diesel" in ql:
        power_pref = "diesel"

    # environment hints
    indoor = bool(re.search(r"\bindoor\b|\bwarehouse\b|\binside\b", ql))
    outdoor = bool(re.search(r"\boutdoor\b|\brough\b|\bgravel\b|\bconstruction\b|\bunpaved\b|\bdirt\b|\tyard\b", ql))
    narrow  = "narrow aisle" in ql or "very narrow" in ql or (aisle_in is not None and aisle_in <= 96)

    # tires (+ synonyms)
    tire_pref = None
    if "cushion" in ql:
        tire_pref = "cushion"
    elif "pneumatic" in ql:
        tire_pref = "pneumatic"
    if "non-marking" in ql or "nonmarking" in ql:
        tire_pref = tire_pref or "cushion"
    if "solid pneumatic" in ql or "foam filled" in ql or "foam-filled" in ql:
        tire_pref = tire_pref or "pneumatic"

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

# --- expose selected list + ALLOWED block for strict grounding -----------
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
