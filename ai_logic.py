# ai_logic.py
"""
Pure helper module: account lookup + model filtering + prompt context builder.
Grounds model picks strictly on models.json and parses user needs robustly.
"""

from __future__ import annotations
import json, re, difflib
from typing import List, Dict, Any, Tuple

# ── load JSON once (robust to {"models":[...]} wrappers) -----------------
def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

accounts_raw = _load_json("accounts.json")
models_raw   = _load_json("models.json")
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

# ── Common key aliases we’ll tolerate in models.json ----------------------
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

# ── small fmt helpers (imported by app.py fallback) -----------------------
def _fmt_int(n):
    try:
        return f"{int(round(float(n))):,}"
    except Exception:
        return None

# ── account helpers -------------------------------------------------------
def get_account(text: str) -> Dict[str, Any] | None:
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

# ── parsing helpers -------------------------------------------------------
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
        if v not in (None, ""):
            return str(v)
    return ""

# ── normalize values from model rows -------------------------------------
def _normalize_capacity_lbs(row: Dict[str,Any]) -> float | None:
    for k in CAPACITY_KEYS:
        if k in row and str(row[k]).strip() != "":
            s = str(row[k])
            # detect units in free text
            if re.search(r"\bkg\b", s, re.I):
                v = _num(s);  return _to_lbs(v, "kg") if v is not None else None
            if re.search(r"\btonne\b|\bmetric\s*ton\b|\b(?<!f)\bt\b", s, re.I):
                v = _num(s);  return _to_lbs(v, "metric ton") if v is not None else None
            if re.search(r"\btons?\b", s, re.I):
                v = _num(s);  return _to_lbs(v, "ton") if v is not None else None
            v = _num(s);      return v
    return None

def _normalize_height_in(row: Dict[str,Any]) -> float | None:
    for k in HEIGHT_KEYS:
        if k in row and str(row[k]).strip() != "":
            s = str(row[k])
            if re.search(r"\bft\b|'", s, re.I):
                v = _num(s);  return _to_inches(v, "ft") if v is not None else None
            v = _num(s);      return v
    return None

def _normalize_aisle_in(row: Dict[str,Any]) -> float | None:
    for k in AISLE_KEYS:
        if k in row and str(row[k]).strip() != "":
            s = str(row[k])
            if re.search(r"\bft\b|'", s, re.I):
                v = _num(s);  return _to_inches(v, "ft") if v is not None else None
            v = _num(s);      return v
    return None

def _is_reach_or_vna(row: Dict[str,Any]) -> bool:
    t = (
        _text_from_keys(row, TYPE_KEYS) + " " +
        str(row.get("Model", "") or row.get("Model Name", ""))
    ).lower()
    return any(w in t for w in ["reach", "vna", "order picker", "turret"]) or \
           re.search(r"\b(cqd|rq|vna)\b", t)

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

    # “5,000 lb capacity” (number first, then unit, then load word)
    m = re.search(rf'{NUM}\s*{UNIT_LB}\s*(?:{LOAD_WORDS})\b', t)
    if m: return (int(round(_n(m.group(1)))), None)

    # “payload/load/weight … 7k/5000”
    m = re.search(rf'{LOAD_WORDS}[^0-9k\-]*{KNUM}', t)
    if m: return (int(round(_n(m.group(1))*1000)), None)
    m = re.search(rf'{LOAD_WORDS}[^0-9\-]*{NUM}', t)
    if m: return (int(round(_n(m.group(1)))), None)

    # singles with units
    m = re.search(rf'{KNUM}\s*(?:{UNIT_LB})?\b', t)         # “7k (lb)”
    if m: return (int(round(_n(m.group(1))*1000)), None)
    m = re.search(rf'{NUM}\s*{UNIT_LB}\b', t)               # “7000 lb/lbs/lb.”
    if m: return (int(round(_n(m.group(1)))), None)
    m = re.search(rf'{NUM}\s*{UNIT_KG}\b', t)               # “2000 kg”
    if m: return (int(round(_n(m.group(1))*2.20462)), None)
    m = re.search(rf'{NUM}\s*{UNIT_TONNE}\b', t)            # “3.5 tonne / 3.5t”
    if m: return (int(round(_n(m.group(1))*2204.62)), None)
    m = re.search(rf'{NUM}\s*{UNIT_TON}\b', t)              # “5 ton(s)”
    if m: return (int(round(_n(m.group(1))*2000)), None)

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
        except Exception:
            pass
    if height_in is None:
        m = re.search(r'(?:lift|raise|reach|height|clearance|mast)\D{0,12}(\d[\d,\.]*)\s*(ft|feet|\'|in|\"|inches)', ql)
        if m:
            raw, unit = m.group(1), m.group(2)
            try:
                height_in = _to_inches(float(raw.replace(",","")), "ft") if unit in ("ft","feet","'") \
                            else float(raw.replace(",",""))
            except Exception:
                height_in = None

    # aisle width (includes right-angle aisle variants)
    aisle_in = None
    m = re.search(r'(?:aisle|aisles|aisle width)\D{0,12}(\d[\d,\.]*)\s*(?:in|\"|inches|ft|\')', ql)
    if m:
        raw, unitblob = m.group(1), m.group(0)
        try:
            aisle_in = _to_inches(float(raw.replace(",","")), "ft") if ("ft" in unitblob or "'" in unitblob) \
                       else float(raw.replace(",",""))
        except Exception:
            pass
    if aisle_in is None:
        m = re.search(r'(?:right[-\s]?angle(?:\s+aisle|\s+stack(?:ing)?)?|ra\s*aisle|ras)\D{0,12}(\d[\d,\.]*)\s*(in|\"|inches|ft|\')', ql)
        if m:
            raw, unit = m.group(1), m.group(2)
            try:
                aisle_in = _to_inches(float(raw.replace(",","")), "ft") if unit in ("ft","'") \
                           else float(raw.replace(",",""))
            except Exception:
                pass

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
    if any(w in ql for w in ["non-marking","nonmarking"]): tire_pref = "non-marking cushion"
    if any(w in ql for w in ["cushion","press-on","press on"]): tire_pref = tire_pref or "cushion"
    if any(w in ql for w in ["pneumatic","air filled","air-filled","rough terrain tires","rt tires","knobby",
                             "off-road","outdoor tires","solid pneumatic","solid-pneumatic","super elastic","foam filled","foam-filled"]):
        # prefer explicit "solid-pneumatic" if seen
        if "solid" in ql and "pneumatic" in ql:
            tire_pref = "solid-pneumatic"
        else:
            tire_pref = tire_pref or "pneumatic"

    return dict(
        cap_lbs=cap_lbs, height_in=height_in, aisle_in=aisle_in,
        power_pref=power_pref, indoor=indoor, outdoor=outdoor,
        narrow=narrow, tire_pref=tire_pref
    )

# ── suggestions based on parsed needs (for context only) -----------------
def _suggest_tire(want: Dict[str, Any]) -> str | None:
    if want.get("tire_pref"):
        return want["tire_pref"]
    if want.get("indoor") and not want.get("outdoor"):
        return "non-marking cushion"
    if want.get("outdoor"):
        return "solid-pneumatic"
    return None

def _suggest_attachments(want: Dict[str, Any]) -> List[str]:
    atts: List[str] = []
    if want.get("outdoor"):
        atts += ["Side shifter", "LED work lights", "Rear grab handle w/ horn"]
    if want.get("indoor"):
        atts += ["Fork positioner"]
    if want.get("cap_lbs") and want["cap_lbs"] >= 7000:
        atts += ["4th function (future clamp-ready)"]
    if want.get("power_pref") == "electric":
        atts += ["Charger/telemetry (opportunity charging)"]
    return list(dict.fromkeys(atts))

# ── model accessors ------------------------------------------------------
def _capacity_of(row: Dict[str,Any]) -> float | None:
    return _normalize_capacity_lbs(row)

def _height_of(row: Dict[str,Any]) -> float | None:
    return _normalize_height_in(row)

def _aisle_of(row: Dict[str,Any]) -> float | None:
    return _normalize_aisle_in(row)

def _power_of(row: Dict[str,Any]) -> str:
    return _text_from_keys(row, POWER_KEYS).lower()

def _tire_of(row: Dict[str,Any]) -> str:
    return str(row.get("Tire Type","") or row.get("Tires","") or row.get("Tire","")).lower()

def _safe_model_name(m: Dict[str, Any]) -> str:
    for k in ("Model","model","Model Name","model name","code","name","Code"):
        if m.get(k):
            return str(m[k]).strip()
    return "N/A"

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

        s += 0.05  # small prior
        scored.append((s, m))

    if not scored:
        return []

    ranked = sorted(scored, key=lambda t: t[0], reverse=True)
    return [m for _, m in ranked[:limit]]

# ── build final prompt chunk (context for the LLM) -----------------------
def generate_forklift_context(user_q: str, acct: Dict[str, Any] | None) -> str:
    """
    IMPORTANT: Pass ONLY the raw user question to this function from app.py.
    The account block is added here (not mixed into user_q), so the capacity
    parser isn't poisoned by SIC/ZIP/fleet-size numbers.
    """
    want = _parse_requirements(user_q)
    tire_suggest = _suggest_tire(want)
    atts_suggest = _suggest_attachments(want)

    # Needs Summary (clean labels; no broken lines)
    needs_lines = [
        "Needs Summary:",
        f"- Environment: {'Indoor' if want['indoor'] and not want['outdoor'] else ('Outdoor' if want['outdoor'] and not want['indoor'] else 'Mixed/Not specified')}",
        f"- Aisle Limit: {(_fmt_int(want['aisle_in']) + ' in') if want['aisle_in'] else 'Not specified'}",
        f"- Power Preference: {want['power_pref'] or 'Not specified'}",
        f"- Minimum Capacity: {(_fmt_int(want['cap_lbs']) + ' lb') if want['cap_lbs'] else 'Not specified'}",
        f"- Suggested Tire: {tire_suggest or 'Not specified'}",
        f"- Suggested Attachments: {', '.join(atts_suggest) if atts_suggest else 'N/A'}",
        ""
    ]

    lines: list[str] = []
    if acct:
        lines.append(customer_block(acct))
    lines.extend(needs_lines)

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

    # Extra guard: drop any model whose normalized capacity < requested min
    cap_need, _ = _parse_capacity_lbs_intent(user_q.lower())
    if cap_need:
        hits = [m for m in hits if (_normalize_capacity_lbs(m) or 0) >= cap_need]

    allowed: List[str] = []
    seen = set()
    for m in hits:
        nm = _safe_model_name(m)
        if nm != "N/A" and nm not in seen:
            allowed.append(nm)
            seen.add(nm)
    return hits, allowed

def allowed_models_block(allowed: list[str]) -> str:
    if not allowed:
        return "ALLOWED MODELS:\n(none – say 'No exact match from our lineup.')"
    return "ALLOWED MODELS:\n" + "\n".join(f"- {x}" for x in allowed)

# --- debugging helper for an admin route ---------------------------------
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

        # same scoring as filter_models (without hard filters here to inspect)
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
            "model": _safe_model_name(m),
            "score": round(s, 3),
            "cap_lbs": cap, "power": powr, "tire": tire,
            "aisle_in": ais, "height_in": hgt
        })
    rows.sort(key=lambda r: r["score"], reverse=True)
    return {"parsed": want, "top": rows[:limit]}
