"""
ai_logic.py
Pure helper module: account lookup + model filtering + prompt context builder.
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
TYPE_KEYS  = ["Type", "Category", "Segment", "Class", "Class/Type", "Truck Type", "Description"]

# ── account helpers ------------------------------------------------------
def get_account(text: str) -> Dict[str, Any] | None:
    low = text.lower()
    for acct in accounts_raw:  # substring pass
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

# ── numeric/unit helpers -------------------------------------------------
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

def _text_from_keys(row: Dict[str,Any], keys: List[str]) -> str:
    for k in keys:
        v = row.get(k)
        if v:
            return str(v)
    return ""

# ── raw field normalizers -----------------------------------------------
def _normalize_capacity_lbs(row: Dict[str,Any]) -> float | None:
    # Try lbs directly with unit detection
    for k in CAPACITY_KEYS:
        if k in row:
            s = str(row[k])
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
    # sometimes JSON already has numeric capacity under a plain key
    v = row.get("capacity_lb") or row.get("capacityLb") or row.get("RatedCapacity")
    return float(v) if isinstance(v, (int, float)) else _num(v)

def _normalize_height_in(row: Dict[str,Any]) -> float | None:
    for k in HEIGHT_KEYS:
        if k in row:
            s = str(row[k])
            if re.search(r"\bft\b|'", s, re.I):
                v = _num(s)
                return _to_inches(v, "ft") if v is not None else None
            v = _num(s)
            return v
    return _num(row.get("MaxLiftHeight_in") or row.get("max_lift_height_in"))

def _normalize_aisle_in(row: Dict[str,Any]) -> float | None:
    for k in AISLE_KEYS:
        if k in row:
            s = str(row[k])
            if re.search(r"\bft\b|'", s, re.I):
                v = _num(s)
                return _to_inches(v, "ft") if v is not None else None
            v = _num(s)
            return v
    return _num(row.get("RA_Aisle_in") or row.get("RightAngleAisle_in"))

def _power_of(row: Dict[str,Any]) -> str:
    return _text_from_keys(row, POWER_KEYS).lower()

def _tire_of(row: Dict[str,Any]) -> str:
    return str(row.get("Tire Type","") or row.get("Tires","") or row.get("Tire","")).lower()

def _safe_model_name(m: Dict[str, Any]) -> str:
    for k in ("Model","model","Model Name","ModelName","code","name","Code"):
        if m.get(k):
            return str(m[k]).strip()
    return "N/A"

def _is_reach_or_vna(row: Dict[str,Any]) -> bool:
    t = (_text_from_keys(row, TYPE_KEYS) + " " + str(row.get("Model","")) + " " + str(row.get("Model Name",""))).lower()
    return any(w in t for w in ["reach", "vna", "order picker", "turret"]) or re.search(r"\b(cqd|rq|vna)\b", t)

# --- family / product-type detection ------------------------------------
WALKIE_HINTS = [
    "cbd", "cbt", "cbx", "cbv", "cbdl", "pt", "pallet",
    "walkie", "walk-behind", "stacker", "reach stacker"
]

def _is_walkie_or_stacker(row: Dict[str, Any]) -> bool:
    name = (_safe_model_name(row) or "").lower()
    text = (
        name + " " +
        _text_from_keys(row, TYPE_KEYS).lower() + " " +
        _text_from_keys(row, ["Truck Category", "Truck Type", "Description"]).lower()
    )
    return any(tok in text for tok in WALKIE_HINTS)

# --- matching utilities --------------------------------------------------
def _power_matches(pref: str | None, powr_text: str) -> bool:
    if not pref:
        return True
    t = (powr_text or "").lower()
    if pref == "electric":
        return ("electric" in t) or ("lith" in t) or ("li-ion" in t) or ("lead" in t)
    return pref in t

def _tire_matches(pref: str | None, tire_text: str, ql: str) -> bool:
    if not pref:
        return True
    t = (tire_text or "").lower()
    if pref == "cushion":
        ok = ("cushion" in t) or ("press-on" in t) or ("press on" in t)
        if "non-marking" in ql or "non marking" in ql:
            if t and ("non-mark" not in t):
                return False
        return ok
    if pref == "pneumatic":
        return ("pneumatic" in t) or ("super elastic" in t) or ("solid pneumatic" in t)
    return pref in t

# --- attachment extraction / suggestions --------------------------------
ATTACHMENT_MAP: Dict[str, List[str]] = {
    "side-shifter": ["side shifter", "sideshifter", "side-shifter"],
    "fork positioner": ["fork positioner", "fp", "integral fp", "integral fork positioner"],
    "blue pedestrian light": ["blue light", "blue pedestrian light", "pedestrian light"],
    "led work lights": ["led lights", "work lights", "led work lights"],
    "fork extensions": ["fork extension", "extensions"],
    "paper roll clamp": ["paper roll", "roll clamp"],
    "carton clamp": ["carton clamp"],
    "push/pull": ["push pull", "push/pull"],
    "rotator": ["rotator"],
}

def _extract_requested_attachments(text: str) -> list[str]:
    ql = (text or "").lower()
    found: List[str] = []
    for key, terms in ATTACHMENT_MAP.items():
        if any(term in ql for term in terms):
            found.append(key)
    # de-dup preserving order
    out, seen = [], set()
    for x in found:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def _suggest_attachments(want: Dict[str, Any], q: str) -> list[str]:
    out = _extract_requested_attachments(q)
    ql = (q or "").lower()
    if want.get("indoor"):
        if "side-shifter" not in out:
            out.append("side-shifter")
        if "fork positioner" not in out and want.get("narrow"):
            out.append("fork positioner")
        if ("dock" in ql or "ramp" in ql) and "blue pedestrian light" not in out:
            out.append("blue pedestrian light")
        if "led work lights" not in out:
            out.append("led work lights")
    if want.get("outdoor") and "led work lights" not in out:
        out.append("led work lights")
    # lithium hint
    if want.get("power_pref") == "electric" and "charger/telemetry" not in out:
        # represent as one suggestion
        out.append("charger/telemetry (opportunity charging)")
    return out[:5]

# --- capacity intent parser ---------------------------------------------
def _parse_capacity_lbs_intent(text: str) -> tuple[int | None, int | None]:
    """
    Returns (min_required_lb, max_allowed_lb). We use min for filtering.
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

    # SAFER fallback: only treat a bare 4–5 digit number as lb if near load words
    near = re.search(rf'{LOAD_WORDS}\D{{0,12}}(\d{{4,5}})\b', t)
    if near:
        return (int(near.group(1)), None)

    return (None, None)

# --- parse requirements ---------------------------------------------------
def _parse_requirements(q: str) -> Dict[str,Any]:
    ql = (q or "").lower()

    # capacity
    cap_min, cap_max = _parse_capacity_lbs_intent(ql)
    cap_lbs = cap_min

    # height (avoid matching aisle numbers)
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

    # aisle width (incl. right-angle aisle)
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

    # power preference
    power_pref = None
    if any(w in ql for w in ["zero emission","zero-emission","emissions free","emissions-free","eco friendly",
                             "eco-friendly","battery powered","battery-powered","battery","lithium",
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

    # attachments & dock work
    attachments = _extract_requested_attachments(ql)
    dock_work   = ("dock" in ql) or ("ramp" in ql)

    return dict(
        cap_lbs=cap_lbs, height_in=height_in, aisle_in=aisle_in,
        power_pref=power_pref, indoor=indoor, outdoor=outdoor,
        narrow=narrow, tire_pref=tire_pref,
        attachments=attachments, dock_work=dock_work
    )

# --- accessors -----------------------------------------------------------
def _capacity_of(row: Dict[str,Any]) -> float | None:
    return _normalize_capacity_lbs(row)

def _height_of(row: Dict[str,Any]) -> float | None:
    return _normalize_height_in(row)

def _aisle_of(row: Dict[str,Any]) -> float | None:
    return _normalize_aisle_in(row)

# --- model filtering & ranking ------------------------------------------
def filter_models(user_q: str, limit: int = 5) -> List[Dict[str, Any]]:
    want = _parse_requirements(user_q)
    ql = (user_q or "").lower()

    cap_need    = want["cap_lbs"]
    aisle_need  = want["aisle_in"]
    power_pref  = want["power_pref"]
    narrow      = want["narrow"]
    tire_pref   = want["tire_pref"]
    height_need = want["height_in"]

    allow_walkies = any(w in ql for w in ["pallet", "walkie", "stacker", "cbd", "cbt"])

    # track if we parsed anything meaningful
    parsed_any = any([cap_need, aisle_need, power_pref, tire_pref, height_need,
                      want["indoor"], want["outdoor"], want["narrow"]])

    def _score_pass(strict_height_aisle: bool) -> List[Tuple[float, Dict[str,Any]]]:
        scored: List[Tuple[float, Dict[str,Any]]] = []
        for m in models_raw:
            # exclude walkies unless explicitly asked
            if not allow_walkies and _is_walkie_or_stacker(m):
                continue

            cap = _capacity_of(m) or 0.0
            powr = _power_of(m)
            tire = _tire_of(m)
            ais  = _aisle_of(m)
            hgt  = _height_of(m)
            reach_like = _is_reach_or_vna(m)

            # HARD filters
            if cap_need and cap > 0 and cap < cap_need:
                continue
            if strict_height_aisle and height_need and hgt and hgt < height_need:
                continue
            if strict_height_aisle and aisle_need and ais and ais > aisle_need:
                continue

            s = 0.0
            # capacity closeness
            if cap_need and cap:
                over = (cap - cap_need) / cap_need
                s += (2.0 - min(2.0, max(0.0, over))) if over >= 0 else -5.0

            # power & tires
            s += 1.2 if _power_matches(power_pref, powr) else (-0.5 if power_pref else 0.0)
            s += 0.7 if _tire_matches(tire_pref, tire, ql) else (-0.3 if tire_pref else 0.0)

            # aisle / narrow
            if aisle_need:
                if ais: s += 0.9 if ais <= aisle_need else -1.0
                else:   s += 0.6 if (narrow and reach_like) else 0.0
            elif narrow:
                s += 0.8 if reach_like else -0.2

            # height
            if height_need:
                if hgt: s += 0.5 if hgt >= height_need else -1.5
                else:   s += -0.2  # unknown height is slightly penalized

            # indoor/outdoor very light nudge via tire/power
            if want["indoor"] and "cushion" in tire: s += 0.2
            if want["outdoor"] and ("pneumatic" in tire or "super elastic" in tire or "solid" in tire): s += 0.2

            s += 0.03  # small prior to avoid ties
            scored.append((s, m))
        return scored

    # First pass: strict on height/aisle when provided
    scored = _score_pass(strict_height_aisle=True)

    # Fallback pass: relax height/aisle strictness if no results
    if not scored:
        scored = _score_pass(strict_height_aisle=False)

    if not scored or not parsed_any:
        return []

    ranked = sorted(scored, key=lambda t: t[0], reverse=True)
    return [m for _, m in ranked[:limit]]

# ── build final prompt chunk --------------------------------------------
def _needs_summary(want: Dict[str, Any], q: str) -> str:
    """Compact needs summary the model can use without poisoning parsing."""
    lines = ["<span class=\"section-label\">Needs Summary:</span>"]
    env = []
    if want.get("indoor"): env.append("Indoor")
    if want.get("outdoor"): env.append("Outdoor")
    env_txt = ", ".join(env) if env else "Not specified"
    lines.append(f"- Environment: {env_txt}")

    lines.append(f"- Aisle Limit: {int(want['aisle_in'])} in" if want.get("aisle_in") else "- Aisle Limit: Not specified")
    if want.get("power_pref"):
        p = "Electric" if want["power_pref"]=="electric" else want["power_pref"].upper()
        lines.append(f"- Power Preference: {p.capitalize()}")
    else:
        lines.append("- Power Preference: Not specified")

    if want.get("cap_lbs"):
        lines.append(f"- Minimum Capacity: {int(want['cap_lbs']):,} lb")
    else:
        lines.append("- Minimum Capacity: Not specified")

    if want.get("tire_pref"):
        pref = "Cushion" if want["tire_pref"]=="cushion" else "Pneumatic"
        lines.append(f"- Tire Preference: {pref}")
    else:
        lines.append("- Tire Preference: Not specified")

    atts = _suggest_attachments(want, q)
    if atts:
        lines.append(f"- Attachments: {', '.join(atts)}")
    else:
        lines.append("- Attachments: Not specified")
    return "\n".join(lines) + "\n"

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
    lines.append(_needs_summary(want, user_q))

    hits = filter_models(user_q)
    if hits:
        lines.append("<span class=\"section-label\">Recommended Heli Models:</span>")
        # keep a short, consistent block per model
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
                f"- {', '.join(_suggest_attachments(want, user_q)) or 'N/A'}",
                "<span class=\"section-label\">Comparison:</span>",
                "- Similar capacity models available from Toyota or CAT are typically higher cost.\n"
            ]
    else:
        lines.append("No matching models found in the provided data.\n")

    # include the raw question last
    lines.append(user_q)
    return "\n".join(lines)

# --- expose selected list + ALLOWED block for strict grounding -----------
def select_models_for_question(user_q: str, k: int = 5):
    hits = filter_models(user_q, limit=k)
    allowed = []
    for m in hits:
        nm = _safe_model_name(m)
        if nm and nm != "N/A":
            allowed.append(nm)
    return hits, allowed

def allowed_models_block(allowed: list[str]) -> str:
    if not allowed:
        return "ALLOWED MODELS:\n(none – say 'No exact match from our lineup.')"
    return "ALLOWED MODELS:\n" + "\n".join(f"- {x}" for x in allowed)

# --- debug helper --------------------------------------------------------
def debug_parse_and_rank(user_q: str, limit: int = 10):
    want = _parse_requirements(user_q)
    rows = []
    ql = (user_q or "").lower()
    allow_walkies = any(w in ql for w in ["pallet", "walkie", "stacker", "cbd", "cbt"])

    for m in models_raw:
        if not allow_walkies and _is_walkie_or_stacker(m):
            continue

        cap = _capacity_of(m) or 0.0
        powr = _power_of(m)
        tire = _tire_of(m)
        ais  = _aisle_of(m)
        hgt  = _height_of(m)
        reach_like = _is_reach_or_vna(m)

        s = 0.0
        if want["cap_lbs"] and cap:
            over = (cap - want["cap_lbs"]) / want["cap_lbs"]
            s += (2.0 - min(2.0, max(0.0, over))) if over >= 0 else -5.0
        s += 1.2 if _power_matches(want["power_pref"], powr) else (-0.5 if want["power_pref"] else 0.0)
        s += 0.7 if _tire_matches(want["tire_pref"], tire, ql) else (-0.3 if want["tire_pref"] else 0.0)
        if want["aisle_in"]:
            if ais: s += 0.9 if ais <= want["aisle_in"] else -1.0
            else:   s += 0.6 if (want["narrow"] and reach_like) else 0.0
        elif want["narrow"]:
            s += 0.8 if reach_like else -0.2
        if want["height_in"]:
            if hgt: s += 0.5 if hgt >= want["height_in"] else -1.5
            else:   s += -0.2
        if want["indoor"] and "cushion" in tire: s += 0.2
        if want["outdoor"] and ("pneumatic" in tire or "super elastic" in tire or "solid" in tire): s += 0.2
        s += 0.03

        rows.append({
            "model": _safe_model_name(m),
            "score": round(s, 3),
            "cap_lbs": cap, "power": powr, "tire": tire,
            "aisle_in": ais, "height_in": hgt
        })
    rows.sort(key=lambda r: r["score"], reverse=True)
    return {"parsed": want, "top": rows[:limit]}
