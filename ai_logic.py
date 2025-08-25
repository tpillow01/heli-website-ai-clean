"""
Pure helper module: account lookup + model filtering + prompt context builder
Grounds model picks strictly on models.json and parses user needs robustly.
"""
from __future__ import annotations
import json, re, difflib
from typing import List, Dict, Any, Tuple

# ── load JSON once -------------------------------------------------------
def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

accounts_raw = _load_json("accounts.json")
models_raw   = _load_json("models.json")

# In case models.json is wrapped like {"models":[...]} or {"data":[...]}
if not isinstance(models_raw, list):
    if isinstance(models_raw, dict):
        if isinstance(models_raw.get("models"), list):
            models_raw = models_raw["models"]
        elif isinstance(models_raw.get("data"), list):
            models_raw = models_raw["data"]
        else:
            models_raw = []
    else:
        models_raw = []

print(f"[ai_logic] Loaded accounts: {len(accounts_raw)} | models: {len(models_raw)}")

# Common key aliases we’ll look for in models.json
CAPACITY_KEYS = [
    "Capacity_lbs", "capacity_lbs", "Capacity", "Rated Capacity", "Load Capacity",
    "Capacity (lbs)", "capacity", "LoadCapacity", "capacityLbs", "RatedCapacity",
    "Load Capacity (lbs)", "Rated Capacity (lbs)"
]
HEIGHT_KEYS = [
    "Lift Height_in", "Max Lift Height (in)", "Lift Height", "Max Lift Height",
    "Mast Height", "lift_height_in", "LiftHeight", "Lift Height (in)", "Mast Height (in)"
]
AISLE_KEYS = [
    "Aisle_min_in", "Aisle Width_min_in", "Aisle Width (in)", "Min Aisle (in)",
    "Right Angle Aisle (in)", "Right-Angle Aisle (in)", "RA Aisle (in)"
]
POWER_KEYS = ["Power", "power", "Fuel", "fuel", "Drive", "Power Type", "PowerType"]
TYPE_KEYS  = ["Type", "Category", "Segment", "Class", "Class/Type", "Truck Type"]

# ── small formatters ------------------------------------------------------
def _fmt_int(n):
    try:
        return f"{int(round(float(n))):,}"
    except Exception:
        return None

def _fmt_lb(n):
    v = _fmt_int(n)
    return f"{v} lb" if v is not None else None

def _fmt_in(n):
    v = _fmt_int(n)
    return f"{v} in" if v is not None else None

def _fmt_v(n):
    v = _fmt_int(n)
    return f"{v} V" if v is not None else None

# ── account helpers (no more false matches on short names like "ATI") ----
def _norm_words(s: str) -> str:
    # Normalize to space-separated alphanumerics
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (s or "").lower())).strip()

def get_account(text: str) -> Dict[str, Any] | None:
    """
    Only match whole-word names. Avoid substring matches so 'ATI' doesn't match
    'pneumATIc' or 'capACIty'. For very short names (<=3 chars) require exact token.
    """
    norm_text = f" {_norm_words(text)} "
    # exact whole-word pass
    for acct in accounts_raw:
        raw_name = (acct.get("Account Name") or "").strip()
        if not raw_name:
            continue
        nm = _norm_words(raw_name)
        if not nm:
            continue
        # short names must match as a whole token
        if len(nm) <= 3:
            if f" {nm} " in norm_text:
                return acct
        else:
            # multiword or longer names: whole-token containment
            if f" {nm} " in norm_text:
                return acct
    # fuzzy match only for longer names to avoid ATI-type collisions
    names = [a.get("Account Name","") for a in accounts_raw if a.get("Account Name")]
    # higher cutoff to reduce spurious matches
    close = difflib.get_close_matches(text, names, n=1, cutoff=0.9)
    if close and len(_norm_words(close[0])) > 3:
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
    # Try lbs directly with unit detection
    for k in CAPACITY_KEYS:
        if k in row:
            s = str(row[k])
            # units
            if re.search(r"\bkg\b", s, re.I):
                v = _num(s)
                return _to_lbs(v, "kg") if v is not None else None
            if re.search(r"\btons?\b|\btonne\b|\bmetric\s*ton\b|\b(?<!f)\bt\b", s, re.I):
                v = _num(s)
                if re.search(r"metric|tonne|\b(?<!f)\bt\b", s, re.I):
                    return _to_lbs(v, "metric ton") if v is not None else None
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
    # check both "Model" and "Model Name"
    t = (
        _text_from_keys(row, TYPE_KEYS) + " " +
        str(row.get("Model","")) + " " + str(row.get("Model Name",""))
    ).lower()
    return any(word in t for word in ["reach", "vna", "order picker", "turret"]) or re.search(r"\b(cqd|rq|vna)\b", t)

# --- robust capacity intent parser --------------------------------------
def _parse_capacity_lbs_intent(text: str) -> tuple[int | None, int | None]:
    """
    Returns (min_required_lb, max_allowed_lb). We use min for filtering.
    Understands: “7000 lb/lbs/lb.”, “7k”, “payload 5,000”, “3.5 ton/tonne/t”,
    ranges “3k–5k / 3000-5000 / between 3,000 and 5,000”, bounds “up to … / min …”.
    """
    if not text:
        return (None, None)

    t = text.lower().replace("–", "-").replace("—", "-")

    UNIT_LB     = r'(?:lb\.?|lbs\.?|pound(?:s)?)'
    UNIT_KG     = r'(?:kg|kgs?|kilogram(?:s)?)'
    UNIT_TONNE  = r'(?:tonne|tonnes|metric\s*ton(?:s)?|(?<!f)\bt\b)'
    UNIT_TON    = r'(?:ton|tons)'
    KNUM        = r'(\d+(?:\.\d+)?)\s*k\b'
    NUM         = r'(\d[\d,\.]*)'
    LOAD_WORDS  = r'(?:capacity|load|loads|payload|rating|lift|handle|carry|weight|weigh|weighs|wt)'

    def _n(s: str) -> float: return float(s.replace(",", ""))

    # ranges “3k-5k”
    m = re.search(rf'{KNUM}\s*-\s*{KNUM}', t)
    if m:
        lo, hi = int(round(_n(m.group(1))*1000)), int(round(_n(m.group(2))*1000))
        return (min(lo, hi), max(lo, hi))

    # ranges “3000-5000 (lbs optional)”
    m = re.search(rf'{NUM}\s*-\s*{NUM}\s*(?:{UNIT_LB})?', t)
    if m:
        a, b = int(round(_n(m.group(1)))), int(round(_n(m.group(2))))
        return (min(a, b), max(a, b))

    # “between 3,000 and 5,000”
    m = re.search(rf'between\s+{NUM}\s+and\s+{NUM}', t)
    if m:
        a, b = int(round(_n(m.group(1)))), int(round(_n(m.group(2))))
        return (min(a, b), max(a, b))

    # bounds
    m = re.search(rf'(?:up to|max(?:imum)?)\s+{NUM}\s*(?:{UNIT_LB})?', t)
    if m: return (None, int(round(_n(m.group(1)))))
    m = re.search(rf'(?:at least|minimum|min)\s+{NUM}\s*(?:{UNIT_LB})?', t)
    if m: return (int(round(_n(m.group(1)))), None)

    # “payload/load/weight … 7k/5000”
    m = re.search(rf'{LOAD_WORDS}[^0-9k\-]*{KNUM}', t)
    if m: return (int(round(_n(m.group(1))*1000)), None)
    m = re.search(rf'{LOAD_WORDS}[^0-9\-]*{NUM}', t)
    if m: return (int(round(_n(m.group(1)))), None)

    # SAFER fallback: bare 4–5 digit number only if near load words
    near = re.search(rf'{LOAD_WORDS}\D{{0,12}}(\d{{4,5}})\b', t)
    if near:
        return (int(near.group(1)), None)

    return (None, None)

# --- parse requirements ---------------------------------------------------
def _parse_requirements(q: str) -> Dict[str,Any]:
    ql = q.lower()

    # capacity
    cap_min, cap_max = _parse_capacity_lbs_intent(ql)
    cap_lbs = cap_min

    # height: avoid matching “90 in” from an aisle phrase
    height_in = None
    for m in re.finditer(r'(\d[\d,\.]*)\s*(ft|feet|\'|in|\"|inches)\b', ql):
        raw, unit = m.group(1), m.group(2)
        ctx = ql[max(0, m.start()-18): m.end()+18]
        if re.search(r'\b(aisle|ra\s*aisle|right[-\s]?angle)\b', ctx):
            continue
        try:
            height_in = _to_inches(float(raw.replace(",","")), "ft") if unit in ("ft","feet","'") \
                        else float(raw.replace(",",""))
            break
        except:
            pass
    if height_in is None:
        m = re.search(r'(?:lift|raise|reach|height|clearance|mast)\D{0,12}(\d[\d,\.]*)\s*(ft|feet|\'|in|\"|inches)', ql)
        if m:
            raw, unit = m.group(1), m.group(2)
            try:
                height_in = _to_inches(float(raw.replace(",","")), "ft") if unit in ("ft","feet","'") \
                            else float(raw.replace(",",""))
            except:
                height_in = None

    # aisle width (includes right-angle aisle variants)
    aisle_in = None
    m = re.search(r'(?:aisle|aisles|aisle width)\D{0,12}(\d[\d,\.]*)\s*(?:in|\"|inches|ft|\')', ql)
    if m:
        raw, unitblob = m.group(1), m.group(0)
        try:
            aisle_in = _to_inches(float(raw.replace(",","")), "ft") if ("ft" in unitblob or "'" in unitblob) \
                       else float(raw.replace(",",""))
        except: pass
    if aisle_in is None:
        m = re.search(r'(?:right[-\s]?angle(?:\s+aisle|\s+stack(?:ing)?)?|ra\s*aisle|ras)\D{0,12}(\d[\d,\.]*)\s*(in|\"|inches|ft|\')', ql)
        if m:
            raw, unit = m.group(1), m.group(2)
            try:
                aisle_in = _to_inches(float(raw.replace(",","")), "ft") if unit in ("ft","'") \
                           else float(raw.replace(",",""))
            except: pass

    # power preference (broad synonyms)
    power_pref = None
    if any(w in ql for w in ["zero emission","zero-emission","emissions free","emissions-free","eco friendly",
                             "eco-friendly","green","battery powered","battery-powered","battery","lithium",
                             "li-ion","li ion","lead acid","lead-acid","electric"]):
        power_pref = "electric"
    if "diesel" in ql: power_pref = "diesel"
    if any(w in ql for w in ["lpg","propane","lp gas","gas (lpg)","gas-powered","gas powered"]): power_pref = "lpg"

    # environment
    indoor  = any(w in ql for w in ["indoor","warehouse","inside","factory floor","distribution center","dc"])
    outdoor = any(w in ql for w in ["outdoor","yard","dock yard","construction","lumber yard","gravel","dirt",
                                    "uneven","rough","pavement","parking lot","rough terrain","rough-terrain"])
    narrow  = ("narrow aisle" in ql) or ("very narrow" in ql) or ("vna" in ql) or ("turret" in ql) \
              or ("reach truck" in ql) or ("stand-up reach" in ql) \
              or (aisle_in is not None and aisle_in <= 96)

    # tires
    tire_pref = None
    if any(w in ql for w in ["cushion","press-on","press on","non-marking","nonmarking"]): tire_pref = "cushion"
    if any(w in ql for w in ["pneumatic","air filled","air-filled","rough terrain tires","rt tires","knobby",
                             "off-road","outdoor tires","solid pneumatic","super elastic","foam filled","foam-filled"]):
        tire_pref = tire_pref or "pneumatic"

    return dict(
        cap_lbs=cap_lbs, height_in=height_in, aisle_in=aisle_in,
        power_pref=power_pref, indoor=indoor, outdoor=outdoor,
        narrow=narrow, tire_pref=tire_pref
    )

# ── accessors ------------------------------------------------------------
def _capacity_of(row: Dict[str,Any]) -> float | None:
    return _normalize_capacity_lbs(row)

def _height_of(row: Dict[str,Any]) -> float | None:
    return _normalize_height_in(row)

def _aisle_of(row: Dict[str,Any]) -> float | None:
    return _normalize_aisle_in(row)

def _power_of(row: Dict[str,Any]) -> str:
    """
    Normalize power so 'lithium' counts as electric, LPG synonyms collapse, etc.
    """
    txt = (_text_from_keys(row, POWER_KEYS) or "").lower().strip()
    if any(w in txt for w in ["lithium", "li-ion", "li ion", "battery", "lead acid", "lead-acid", "electric"]):
        if "electric" not in txt:
            txt += " electric"
    if any(w in txt for w in ["lpg", "lp ", "lp-gas", "propane"]):
        if "lpg" not in txt:
            txt += " lpg"
    return txt

def _tire_of(row: Dict[str,Any]) -> str:
    t = str(row.get("Tire Type","") or row.get("Tires","") or row.get("Tire","")).lower()
    if t:
        return t
    # Heuristic: many 3-wheel electrics are cushion by default for indoor use
    drive = str(row.get("Drive Type","") or row.get("Drive","")).lower()
    power = _power_of(row)
    if "three wheel" in drive and "electric" in power:
        return "cushion"
    return ""

def _safe_model_name(m: Dict[str, Any]) -> str:
    # Include "Model Name" because your JSON uses it
    for k in ("Model","model","Model Name","code","name","Code"):
        if m.get(k):
            return str(m[k]).strip()
    return "N/A"

# Canonicalize / de-duplicate model display variants
def _canon_model_code(s: str) -> str:
    t = (s or "").upper()
    t = t.replace("_", "-")
    t = re.sub(r"\s+", "", t)
    t = t.replace("(", "").replace(")", "")
    t = re.sub(r"-{2,}", "-", t)
    return t

def _display_model_name(m: Dict[str, Any]) -> str:
    nm = _safe_model_name(m)
    return re.sub(r"\s*-\s*", "-", nm).strip()

# --- suggestions based on needs -----------------------------------------
def _suggestions_from_needs(want: Dict[str, Any]) -> Dict[str, List[str] | str]:
    sugg_tire = None
    attach = []

    # Tire suggestions
    if want["tire_pref"]:
        sugg_tire = "non-marking cushion" if want["tire_pref"] == "cushion" and want["indoor"] else want["tire_pref"]
    else:
        if want["outdoor"]:
            sugg_tire = "solid-pneumatic"  # docks/pavement friendly, puncture resistant
        elif want["indoor"]:
            sugg_tire = "cushion"  # typical indoor

    # Attachment suggestions (generic, based on use)
    if want["outdoor"]:
        attach += ["Side shifter (standard on most builds)", "LED work lights", "Rear grab handle w/ horn"]
    if want["indoor"]:
        attach += ["Fork positioner (varied pallets)"]
    if want["cap_lbs"] and want["cap_lbs"] >= 7000:
        attach += ["4th function hydraulics (future clamp/attachments)"]
    if want["height_in"] and want["height_in"] >= 180:
        attach += ["Lift height safety (overhead guard lights/alarms)"]
    if want["power_pref"] == "electric":
        attach += ["Battery telemetry/charger (opportunity charging)"]
    # Deduplicate while preserving order
    seen = set()
    attach = [a for a in attach if not (a in seen or seen.add(a))]

    return {"tire_suggest": sugg_tire or "N/A", "attachments_suggest": attach or ["N/A"]}

# --- model filtering & ranking ------------------------------------------
def filter_models(user_q: str, limit: int = 5) -> List[Dict[str, Any]]:
    want = _parse_requirements(user_q)
    cap_need   = want["cap_lbs"]
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

        # hard filters
        if cap_need and cap > 0 and cap < cap_need:
            continue
        if aisle_need and ais and ais > aisle_need:
            continue

        s = 0.0
        if cap_need and cap:
            over = (cap - cap_need) / cap_need
            s += (2.0 - min(2.0, max(0.0, over))) if over >= 0 else -5.0
        if power_pref: s += 1.2 if power_pref in powr else -0.6
        if tire_pref:  s += 0.6 if tire_pref in tire   else -0.2
        if aisle_need:
            if ais: s += 0.8 if ais <= aisle_need else -1.0
            else:   s += 0.6 if (narrow and reach_like) else 0.0
        elif narrow:
            s += 0.8 if reach_like else -0.2
        if height_need and hgt: s += 0.4 if hgt >= height_need else -0.3

        s += 0.05  # small prior to avoid ties
        scored.append((s, m))

    if not scored:
        return []

    ranked = sorted(scored, key=lambda t: t[0], reverse=True)
    return [m for _, m in ranked[:limit]]

# --- spec helpers for Top Pick ------------------------------------------
_SPEC_KEYS = {
    "capacity": CAPACITY_KEYS,
    "turning": ["Min. Outside Turning Radius (in)", "Outside Turning Radius (in)", "Turning Radius (in)", "turning_in"],
    "load_center": ["Load Center (in)", "LC (in)", "Load Center", "load_center_in"],
    "battery_v": ["Battery Voltage", "Battery (V)", "battery_v", "Voltage"],
    "controller": ["Controller", "controller"],
    "power": POWER_KEYS,
    "wheel_base": ["Wheel Base", "Wheelbase (in)", "wheel_base_in", "Wheelbase"],
    "overall_height": ["Overall Height (in)", "Height_in", "Overall Height"],
    "overall_length": ["Overall Length (in)", "Length_in", "Overall Length"],
    "overall_width":  ["Overall Width (in)", "Width_in", "Overall Width"],
    "max_lift_height": ["Max Lifting Height (in)", "Lift Height_in", "Lift Height", "Max Lift Height"],
    "drive_type": ["Drive Type", "Drive"],
    "series": ["Series", "Family"],
    "workplace": ["Workplace", "Environment", "Application"]
}

def _get_first(row: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in row and str(row[k]).strip() != "":
            return row[k]
    return None

def _top_pick_specs(row: Dict[str, Any]) -> Dict[str, Any]:
    cap = _normalize_capacity_lbs(row)
    turning = _num(_get_first(row, _SPEC_KEYS["turning"]))
    lc = _num(_get_first(row, _SPEC_KEYS["load_center"]))
    batt = _num(_get_first(row, _SPEC_KEYS["battery_v"]))
    specs = {
        "Model": _safe_model_name(row),
        "Series": _get_first(row, _SPEC_KEYS["series"]) or "N/A",
        "Power": _text_from_keys(row, POWER_KEYS) or "N/A",
        "Drive Type": _get_first(row, _SPEC_KEYS["drive_type"]) or "N/A",
        "Controller": _get_first(row, _SPEC_KEYS["controller"]) or "N/A",
        "Capacity": _fmt_lb(cap) or (_get_first(row, _SPEC_KEYS["capacity"]) or "N/A"),
        "Load Center": _fmt_in(lc) or (_get_first(row, _SPEC_KEYS["load_center"]) or "N/A"),
        "Turning Radius": _fmt_in(turning) or (_get_first(row, _SPEC_KEYS["turning"]) or "N/A"),
        "Overall Width":  _fmt_in(_num(_get_first(row, _SPEC_KEYS["overall_width"])))  or (_get_first(row, _SPEC_KEYS["overall_width"]) or "N/A"),
        "Overall Length": _fmt_in(_num(_get_first(row, _SPEC_KEYS["overall_length"]))) or (_get_first(row, _SPEC_KEYS["overall_length"]) or "N/A"),
        "Overall Height": _fmt_in(_num(_get_first(row, _SPEC_KEYS["overall_height"]))) or (_get_first(row, _SPEC_KEYS["overall_height"]) or "N/A"),
        "Wheel Base":     _fmt_in(_num(_get_first(row, _SPEC_KEYS["wheel_base"])))     or (_get_first(row, _SPEC_KEYS["wheel_base"]) or "N/A"),
        "Max Lift Height":_fmt_in(_num(_get_first(row, _SPEC_KEYS["max_lift_height"])))or (_get_first(row, _SPEC_KEYS["max_lift_height"]) or "N/A"),
        "Workplace": _get_first(row, _SPEC_KEYS["workplace"]) or "N/A",
        "Tire Type (from data)": (_tire_of(row) or "N/A")
    }
    return specs

# ── build final prompt chunk --------------------------------------------
def generate_forklift_context(user_q: str, acct: Dict[str, Any] | None) -> str:
    """
    IMPORTANT: Pass ONLY the raw user question to this function from app.py.
    The account block is added here (not mixed into user_q), so the parser
    isn't poisoned by SIC/ZIP/fleet-size numbers.
    """
    lines: list[str] = []
    if acct:
        lines.append(customer_block(acct))

    want = _parse_requirements(user_q)
    sugg = _suggestions_from_needs(want)

    # Needs summary & suggestions (assist the LLM to fill Tires/Attachments)
    lines += [
        "<span class=\"section-label\">Needs Summary:</span>",
        f"- Min Capacity: {(_fmt_lb(want['cap_lbs']) if want['cap_lbs'] else 'N/A')}",
        f"- Lift Height Need: {(_fmt_in(want['height_in']) if want['height_in'] else 'N/A')}",
        f"- Aisle Limit: {(_fmt_in(want['aisle_in']) if want['aisle_in'] else 'N/A')}",
        f"- Power Preference: {want['power_pref'] or 'N/A'}",
        f"- Environment: {'indoor' if want['indoor'] else ''}{'/outdoor' if want['outdoor'] else ''}".replace("//","/") or "- Environment: N/A",
        f"- Suggested Tire: {sugg['tire_suggest']}",
        f"- Suggested Attachments: {', '.join(sugg['attachments_suggest'])}",
        ""
    ]

    hits = filter_models(user_q)
    if hits:
        # Recommended list
        lines.append("<span class=\"section-label\">Recommended Heli Models:</span>")
        allowed_names = []
        for m in hits:
            model_name = _display_model_name(m)
            allowed_names.append(model_name)
            lines += [
                "<span class=\"section-label\">Model:</span>",
                f"- {model_name}",
                "<span class=\"section-label\">Power:</span>",
                f"- {_text_from_keys(m, POWER_KEYS) or 'N/A'}",
                "<span class=\"section-label\">Capacity:</span>",
                f"- {(_normalize_capacity_lbs(m) or 'N/A')}",
                "<span class=\"section-label\">Tire Type:</span>",
                f"- {_tire_of(m) or 'N/A'}",
                ""
            ]

        # Top pick (first in ranked list) with full specs
        top = hits[0]
        specs = _top_pick_specs(top)
        lines += [
            "<span class=\"section-label\">Top Pick Details (from data):</span>",
            f"- Model: {specs['Model']}",
            f"- Series: {specs['Series']}",
            f"- Power: {specs['Power']}",
            f"- Drive Type: {specs['Drive Type']}",
            f"- Controller: {specs['Controller']}",
            f"- Capacity: {specs['Capacity']}",
            f"- Load Center: {specs['Load Center']}",
            f"- Turning Radius: {specs['Turning Radius']}",
            f"- Overall Width: {specs['Overall Width']}",
            f"- Overall Length: {specs['Overall Length']}",
            f"- Overall Height: {specs['Overall Height']}",
            f"- Wheel Base: {specs['Wheel Base']}",
            f"- Max Lift Height: {specs['Max Lift Height']}",
            f"- Workplace: {specs['Workplace']}",
            f"- Tire Type (from data): {specs['Tire Type (from data)']}",
            ""
        ]

        lines += [
            "<span class=\"section-label\">Comparison:</span>",
            "- Similar capacity models available from Toyota or CAT are typically higher cost.\n"
        ]
    else:
        lines.append("No matching models found in the provided data.\n")

    lines.append(user_q)
    return "\n".join(lines)

# --- expose selected list + ALLOWED block for strict grounding -----------
def select_models_for_question(user_q: str, k: int = 5):
    hits = filter_models(user_q, limit=k*2)  # pull a few extra before de-dupe
    allowed, seen = [], set()
    deduped_hits = []
    for m in hits:
        nm = _display_model_name(m)
        if nm == "N/A":
            continue
        key = _canon_model_code(nm)
        if key in seen:
            continue
        seen.add(key)
        allowed.append(nm)
        deduped_hits.append(m)
        if len(allowed) >= k:
            break
    # ensure we return the deduped list in the same order (top pick first)
    return deduped_hits[:k], allowed

def allowed_models_block(allowed: list[str]) -> str:
    if not allowed:
        return "ALLOWED MODELS:\n(none – say 'No exact match from our lineup.')"
    return "ALLOWED MODELS:\n" + "\n".join(f"- {x}" for x in allowed)

# --- debug: show parsed needs & ranking ---------------------------------
def debug_parse_and_rank(user_q: str, limit: int = 10):
    want = _parse_requirements(user_q)
    rows = []
    for m in models_raw:
        cap = _capacity_of(m) or 0.0
        powr = _power_of(m)
        tire = _tire_of(m)
        ais  = _aisle_of(m)
        hgt  = _height_of(m)
        reach_like = _is_reach_or_vna(m)

        # same scoring as filter_models
        s = 0.0
        if want["cap_lbs"] and cap:
            over = (cap - want["cap_lbs"]) / want["cap_lbs"]
            s += (2.0 - min(2.0, max(0.0, over))) if over >= 0 else -5.0
        if want["power_pref"]: s += 1.2 if want["power_pref"] in powr else -0.6
        if want["tire_pref"]:  s += 0.6 if want["tire_pref"]  in tire else -0.2
        if want["aisle_in"]:
            if ais: s += 0.8 if ais <= want["aisle_in"] else -1.0
            else:   s += 0.6 if (want["narrow"] and reach_like) else 0.0
        elif want["narrow"]:
            s += 0.8 if reach_like else -0.2
        if want["height_in"] and hgt: s += 0.4 if hgt >= want["height_in"] else -0.3
        s += 0.05

        rows.append({
            "model": _display_model_name(m),
            "score": round(s, 3),
            "cap_lbs": cap, "power": powr, "tire": tire,
            "aisle_in": ais, "height_in": hgt
        })
    rows.sort(key=lambda r: r["score"], reverse=True)
    return {"parsed": want, "top": rows[:limit]}
