"""
ai_logic.py
Pure helper module: account lookup + model filtering + prompt context builder.
Grounds model picks strictly on models.json and parses user needs robustly.
"""

from __future__ import annotations
import json, re, difflib
from typing import List, Dict, Any, Tuple, Optional

# --- Options loader (Step 2: read /data/forklift_options_benefits.xlsx) ---
import os
from functools import lru_cache

try:
    import pandas as _pd  # uses your existing pandas from requirements.txt
except Exception:
    _pd = None

_OPTIONS_XLSX = os.path.join(os.path.dirname(__file__), "data", "forklift_options_benefits.xlsx")

def _make_code(name: str) -> str:
    s = (name or "").upper()
    s = re.sub(r"[^\w\+\s-]", " ", s)      # remove odd chars
    s = s.replace("+", " PLUS ")
    s = re.sub(r"[\s/-]+", "_", s)         # normalize separators
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:64] or "UNKNOWN_OPTION"

@lru_cache(maxsize=1)
def load_options() -> list[dict]:
    """
    Returns a list of dicts: [{code, name, benefit}], read from Excel.
    Requires columns: Option, Benefit (case-insensitive is OK).
    """
    if _pd is None:
        return []
    if not os.path.exists(_OPTIONS_XLSX):
        return []
    df = _pd.read_excel(_OPTIONS_XLSX)
    # normalize headers
    cols = {c.lower().strip(): c for c in df.columns}
    opt_col = cols.get("option")
    ben_col = cols.get("benefit")
    if not opt_col or not ben_col:
        return []
    out = []
    for _, r in df.iterrows():
        name = str(r.get(opt_col, "")).strip()
        if not name:
            continue
        benefit = str(r.get(ben_col, "")).strip()
        out.append({"code": _make_code(name), "name": name, "benefit": benefit})
    return out

@lru_cache(maxsize=1)
def options_lookup_by_name() -> dict:
    """Lowercase name -> row dict, for quick lookups."""
    return {o["name"].lower(): o for o in load_options()}

def option_benefit(name: str) -> str | None:
    """Convenience: get the benefit sentence for an exact option name."""
    row = options_lookup_by_name().get((name or "").lower())
    return row["benefit"] if row else None

# --- Step 3: recommend best options directly from the Excel sheet --------
def _infer_category_from_name(n: str) -> str:
    n = (n or "").lower()
    if "tire" in n: return "Tires"
    if "valve" in n or "finger control" in n: return "Hydraulics / Controls"
    if any(k in n for k in ["light", "beacon", "radar", "ops", "blue spot", "red side"]):
        return "Lighting / Safety"
    if any(k in n for k in ["seat", "cab", "windshield", "wiper", "heater", "air conditioner", "rain-proof"]):
        return "Cab / Comfort"
    if any(k in n for k in ["radiator", "screen", "belly pan", "protection bar", "fan"]):
        return "Protection / Cooling"
    if "brake" in n: return "Braking"
    if "fics" in n or "fleet" in n: return "Telematics"
    if "cold storage" in n: return "Environment"
    if "lpg" in n or "fuel" in n: return "Fuel / LPG"
    if any(k in n for k in ["overhead guard", "lifting eyes"]): return "Chassis / Structure"
    return "Other"

def _options_iter():
    """Yield normalized option rows from the Excel loader with inferred category."""
    for o in load_options():
        name = o["name"]
        yield {
            "code": o["code"],
            "name": name,
            "benefit": o.get("benefit",""),
            "category": _infer_category_from_name(name),
            "lname": name.lower()
        }

def _score_option_for_needs(opt: dict, want: dict) -> float:
    """
    Heuristic score: higher = more relevant for stated needs.
    We only use info available in the option name/benefit + parsed needs.
    """
    s = 0.0
    name = opt["lname"]
    indoor, outdoor = want.get("indoor"), want.get("outdoor")
    cap = (want.get("cap_lbs") or 0)
    power_pref = (want.get("power_pref") or "")

    # Tires: pick ONE best later, but still score to rank within tires
    if opt["category"] == "Tires":
        # Default logic
        if indoor and outdoor:
            if "solid" in name and "dual" not in name and "non-mark" not in name: s += 5.0
            if "dual" in name: s += 2.0 if cap >= 8000 else -0.5
            if "non-mark" in name: s -= 0.8  # mixed use penalizes non-marking
        elif outdoor and not indoor:
            if "solid" in name: s += 4.0
            if "dual" in name and cap >= 8000: s += 2.5
            if "non-mark" in name: s -= 1.5
        elif indoor and not outdoor:
            if "non-mark" in name: s += 4.0
            if "cushion" in name or "press-on" in name: s += 2.0
            if "solid pneumatic" in name or "pneumatic" in name: s -= 1.0
        else:
            if "solid" in name and "non-mark" not in name: s += 2.0

    # Hydraulics / Controls: more functions for productivity & heavy loads
    if opt["category"] == "Hydraulics / Controls":
        if any(k in name for k in ["4valve","4-valve","4 valve","5 valve","5-valve"]): s += 3.5
        if any(k in name for k in ["3valve","3-valve","3 valve"]): s += 2.0
        if "finger control" in name: s += 1.0  # ergonomics
        if "msg65" in name: s += 0.8          # seat requirement encoded in name

    # Lighting / Safety: valuable in mixed/indoor aisles & reversing
    if opt["category"] == "Lighting / Safety":
        if "blue spot" in name or "red side" in name: s += 2.5 if indoor or (indoor and outdoor) else 1.0
        if "rotating" in name or "beacon" in name: s += 1.5
        if "rear working light" in name: s += 1.5 if outdoor or (indoor and outdoor) else 0.8
        if "radar" in name or "ops" in name: s += 1.2

    # Cab / Comfort: useful for long shifts & temperature extremes
    if opt["category"] == "Cab / Comfort":
        if "msg65" in name or "suspension seat" in name: s += 2.0 if cap >= 6000 else 1.0
        if "heater" in name or "air conditioner" in name: s += 1.5 if outdoor or (indoor and outdoor) else 0.5
        if "cab" in name or "windshield" in name or "rain-proof" in name: s += 1.2 if outdoor or (indoor and outdoor) else 0.3

    # Protection / Cooling: better outdoors/heavy-duty
    if opt["category"] == "Protection / Cooling":
        if outdoor or (indoor and outdoor): s += 1.8
        if "radiator" in name or "screen" in name or "fan" in name: s += 0.6
        if "belly pan" in name or "protection bar" in name: s += 0.6

    # Braking: heavy capacities benefit
    if opt["category"] == "Braking":
        if cap >= 8000: s += 2.0

    # Telematics note: present but deprioritize if market suspended
    if opt["category"] == "Telematics":
        s += 0.4  # useful generally
        if "suspend" in opt["benefit"].lower() or "suspend" in name: s -= 0.6

    # Environment: cold storage when explicitly mentioned
    qtext = ""  # only using want for now; you can pass the raw text if needed
    # (If you later pass raw user_q, you can boost for "freezer", "cold", etc.)

    # Diesel constraint example (speed control)
    if "speed control" in name and power_pref == "diesel":
        s -= 5.0  # not for diesel engines per your sheet

    return s

def recommend_options_from_sheet(user_q: str, max_total: int = 6) -> dict:
    """
    Returns:
      {
        "tire": {"code","name","benefit","why"} | None,
        "attachments": [{"code","name","benefit"}, ...],
        "others": [{"code","name","benefit"}, ...]
      }
    """
    text = (user_q or "").lower()
    rows = load_options()  # you already have this loader

    # Helper to find an option row by (partial) name
    def find(name_fragment: str):
        frag = name_fragment.strip().lower()
        for r in rows:
            nm = str(r.get("name", "")).lower()
            if frag == nm or frag in nm:
                return r
        return None

    # -------------------- Tire (smart) --------------------
    tire = None
    try:
        t_name, t_why = pick_tire_advanced(user_q)  # added earlier in ai_logic.py
    except Exception:
        t_name, t_why = ("Dual Tires", "Mixed/unspecified environment — dual provides added stability and versatility.")

    t_row = find(t_name)
    if t_row:
        tire = {
            "code": t_row.get("code", ""),
            "name": t_row.get("name", t_name),
            "benefit": t_row.get("benefit", ""),
            "why": t_why
        }
    else:
        tire = {"code": "", "name": t_name, "benefit": "", "why": t_why}

    # -------------------- Attachments (rule-based) --------------------
    # Core attachments you asked to include when relevant
    attach_targets = [
        ("Sideshifter", ["tight aisles", "line up", "frequent pallet", "dock", "staging", "align", "precision"]),
        ("Fork Positioner", ["mixed pallet", "varied pallet", "different pallet", "varying pallet", "multiple widths", "rolls", "coils"]),
        ("Paper Roll Clamp", ["paper", "roll", "tissue", "newsprint"]),
        ("Push/ Pull", ["slip-sheet", "slipsheet", "slip sheet", "bagged goods", "bulk goods", "carton flow"]),
        ("Carpet Pole", ["carpet", "textile", "fabric rolls"]),
        ("Fork Extensions", ["long", "oversize", "wide", "deep pallet", "overhang", "extra reach"])
    ]

    attachments = []
    for name, cues in attach_targets:
        if any(k in text for k in cues):
            row = find(name)
            if row:
                attachments.append({"code": row.get("code",""), "name": row.get("name", name), "benefit": row.get("benefit","")})
            else:
                attachments.append({"code": "", "name": name, "benefit": ""})

    # Gentle defaults if nothing matched but they clearly handle pallets:
    if not attachments and any(k in text for k in ["pallet", "pallets", "dock", "warehouse", "racking"]):
        for default_name in ("Sideshifter", "Fork Positioner"):
            row = find(default_name)
            if row:
                attachments.append({"code": row.get("code",""), "name": row.get("name", default_name), "benefit": row.get("benefit","")})

    # Limit attachments
    attachments = attachments[:min(6, max_total)]

    # -------------------- Options (safety/comfort based on cues) --------------------
    others = []

    def maybe_add(opt_name_fragment: str, when: bool):
        if not when:
            return
        row = find(opt_name_fragment)
        if row:
            item = {"code": row.get("code",""), "name": row.get("name", opt_name_fragment), "benefit": row.get("benefit","")}
            # prevent duplicates if already included via another path
            if all(item["name"] != x.get("name") for x in others):
                others.append(item)

    # Lighting / awareness
    need_ped = any(k in text for k in ["pedestrian", "foot traffic", "busy aisles", "congested", "blind corner", "walkway"])
    dark = any(k in text for k in ["low light", "dim", "night", "second shift", "poor lighting"])
    reversing = any(k in text for k in ["reverse", "backing", "back up", "back-up"])

    maybe_add("Blue spot Light", need_ped or dark)
    maybe_add("Red side line Light", need_ped)
    maybe_add("LED Rotating Light", need_ped)
    maybe_add("Rear Working Light", dark)
    maybe_add("LED Rear Working Light", dark)
    maybe_add("Visible backward radar", reversing)

    # Controls / hydraulics (if multiple attachments or precision)
    precision = any(k in text for k in ["precise", "fine control", "ergonomic", "fatigue", "long shifts"])
    many_funcs = len(attachments) >= 2 or any(k in text for k in ["4th function", "fourth function", "multiple clamps"])
    if many_funcs:
        for name in ["Finger control system(4valve)", "5 Valve with Handle"]:
            maybe_add(name, True)
    elif precision:
        for name in ["Finger control system(3valve)", "3 Valve with Handle"]:
            maybe_add(name, True)

    # Comfort / cab
    hot = any(k in text for k in ["hot", "heat", "summer", "high ambient", "foundry"])
    cold = any(k in text for k in ["cold", "freezer", "cold storage", "refrigerated", "winter"])
    dusty = any(k in text for k in ["dust", "fines", "powder", "sawdust", "grind"])
    wet = any(k in text for k in ["rain", "outdoor", "weather", "snow", "sleet"])

    maybe_add("Panel mounted Cab", wet or cold)
    maybe_add("Air conditioner", hot)
    maybe_add("Heater", cold)
    maybe_add("Glass Windshield with Wiper", wet)
    maybe_add("Top Rain-proof Glass", wet)
    maybe_add("Rear Windshield Glass", wet)
    maybe_add("Dual Air Filter", dusty)
    maybe_add("Pre air cleaner", dusty)

    # Protection / durability
    debris = any(k in text for k in ["debris", "scrap", "metal", "glass", "shavings"])
    rough = any(k in text for k in ["rough", "curb", "dock plate", "pothole", "rail"])
    maybe_add("Radiator protection bar", debris or rough)
    maybe_add("Steel Belly Pan", debris)
    maybe_add("Removable radiator screen", dusty or debris)

    # Fuel / operations
    maybe_add("Low Fuel Indicator Light", any(k in text for k in ["lpg", "diesel", "ic"]))
    maybe_add("Swing Out Drop LPG Bracket", "lpg" in text or "propane" in text or "lp " in text)
    maybe_add("LPG Tank", "lpg" in text or "propane" in text)
    maybe_add("Cooling Fan", hot)
    maybe_add("Speed Control system", any(k in text for k in ["limit speed", "pedestrian safety", "speeding"]))

    # Telematics / ops policy
    maybe_add("Full OPS", need_ped or reversing or any(k in text for k in ["safety policy", "osha", "audit", "insurance"]))

    # Cap others to remaining budget after attachments (but <= max_total)
    remaining = max_total - len(attachments)
    if remaining > 0:
        others = others[:remaining]
    else:
        others = []

    return {"tire": tire, "attachments": attachments, "others": others}

# ── load JSON once -------------------------------------------------------
def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

try:
    accounts_raw = _load_json("accounts.json")
except Exception:
    accounts_raw = []
try:
    models_raw = _load_json("models.json")
except Exception:
    models_raw = []

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
POWER_KEYS = ["Power", "power", "Fuel", "fuel", "Drive", "Drive Type", "Power Type", "PowerType"]
TYPE_KEYS  = ["Type", "Category", "Segment", "Class", "Class/Type", "Truck Type"]

# ── account helpers ------------------------------------------------------
def get_account(text: str) -> Optional[Dict[str, Any]]:
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

# ── parsing & normalization helpers -------------------------------------
def _to_lbs(val: float, unit: str) -> float:
    u = unit.lower()
    if "kg" in u: return float(val) * 2.20462
    if "ton" in u and "metric" in u: return float(val) * 2204.62
    if "ton" in u: return float(val) * 2000.0
    return float(val)

def _to_inches(val: float, unit: str) -> float:
    u = unit.lower()
    if "ft" in u or "'" in u: return float(val) * 12.0
    return float(val)

def _num(s: Any) -> Optional[float]:
    if s is None: return None
    m = re.search(r"-?\d+(?:\.\d+)?", str(s))
    return float(m.group(0)) if m else None

def _num_from_keys(row: Dict[str,Any], keys: List[str]) -> Optional[float]:
    for k in keys:
        if k in row and str(row[k]).strip() != "":
            v = _num(row[k])
            if v is not None:
                return v
    return None

def _normalize_capacity_lbs(row: Dict[str,Any]) -> Optional[float]:
    """Support values like '5000', '5,000 lb', '3.5t', '3500 kg', '7000 lbs'."""
    for k in CAPACITY_KEYS:
        if k in row:
            s = str(row[k]).strip()
            if not s: continue
            # units
            if re.search(r"\bkg\b", s, re.I):
                v = _num(s); return _to_lbs(v, "kg") if v is not None else None
            if re.search(r"\btonne\b|\bmetric\s*ton\b|\b(?<!f)\bt\b", s, re.I):
                v = _num(s); return _to_lbs(v, "metric ton") if v is not None else None
            if re.search(r"\btons?\b", s, re.I):
                v = _num(s); return _to_lbs(v, "ton") if v is not None else None
            # plain number (assume lb)
            v = _num(s)
            return float(v) if v is not None else None
    return None

def _normalize_height_in(row: Dict[str,Any]) -> Optional[float]:
    for k in HEIGHT_KEYS:
        if k in row:
            s = str(row[k])
            if re.search(r"\bft\b|'", s, re.I):
                v = _num(s); return _to_inches(v, "ft") if v is not None else None
            v = _num(s); return v
    return None

def _normalize_aisle_in(row: Dict[str,Any]) -> Optional[float]:
    for k in AISLE_KEYS:
        if k in row:
            s = str(row[k])
            if re.search(r"\bft\b|'", s, re.I):
                v = _num(s); return _to_inches(v, "ft") if v is not None else None
            v = _num(s); return v
    return None

def _text_from_keys(row: Dict[str,Any], keys: List[str]) -> str:
    for k in keys:
        v = row.get(k)
        if v:
            return str(v)
    return ""

def _is_reach_or_vna(row: Dict[str,Any]) -> bool:
    t = (_text_from_keys(row, TYPE_KEYS) + " " + str(row.get("Model","")) + " " + str(row.get("Model Name",""))).lower()
    if any(word in t for word in ["reach", "vna", "order picker", "turret"]):
        return True
    return bool(re.search(r"\b(cqd|rq|vna)\b", t))

def _is_three_wheel(row: Dict[str, Any]) -> bool:
    drive = (str(row.get("Drive Type", "")) + " " + str(row.get("Drive", ""))).lower()
    model = (str(row.get("Model","")) + " " + str(row.get("Model Name",""))).lower()
    return ("three wheel" in drive) or ("3-wheel" in drive) or (" 3 wheel" in drive) or ("sq" in model)

def _power_matches(pref: Optional[str], powr_text: str) -> bool:
    if not pref:
        return True
    p = pref.lower()
    t = (powr_text or "").lower()
    if p == "electric":
        return any(x in t for x in ("electric", "lithium", "li-ion", "lead", "battery"))
    if p == "lpg":
        return any(x in t for x in ("lpg", "propane", "lp gas", "gas"))
    if p == "diesel":
        return "diesel" in t
    return p in t

# --- robust capacity intent parser (improved) -----------------------------
def _parse_capacity_lbs_intent(text: str) -> tuple[Optional[int], Optional[int]]:
    """
    Returns (min_required_lb, max_allowed_lb). We use min for filtering.
    Understands: ~5,000 lb / ≈5000 lb / about 5k / 3.5t / 3.5 tonne / 5k+ /
    3000-5000 / between 3,000 and 5,000 / up to 6000 / min 5000 / etc.
    """
    if not text:
        return (None, None)

    t = text.lower().replace("–", "-").replace("—", "-")
    t = re.sub(r"[~≈≃∼]", "", t)  # remove approx markers
    t = re.sub(r"\bapproximately\b|\bapprox\.?\b|\baround\b|\babout\b", "", t)

    UNIT_LB     = r'(?:lb\.?|lbs\.?|pound(?:s)?)'
    UNIT_KG     = r'(?:kg|kgs?|kilogram(?:s)?)'
    UNIT_TONNE  = r'(?:tonne|tonnes|metric\s*ton(?:s)?|(?<!f)\bt\b)'
    UNIT_TON    = r'(?:ton|tons)'
    KNUM        = r'(\d+(?:\.\d+)?)\s*k\b'
    NUM         = r'(\d[\d,\.]*)'
    LOAD_WORDS  = r'(?:capacity|load|payload|rating|lift|handle|carry|weight|wt)'

    def _n(s: str) -> float: return float(s.replace(",", ""))

    # 5k+ / 5000+ lb => min
    m = re.search(rf'(?:{KNUM}|{NUM})\s*\+\s*(?:{UNIT_LB})?', t)
    if m:
        val = m.group(1) or m.group(2)
        v = float(val.replace(",", ""))
        if m.group(1) is not None:  # KNUM matched
            return (int(round(v*1000)), None)
        return (int(round(v)), None)

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

    # “payload/load/capacity … 7k/5000”
    m = re.search(rf'{LOAD_WORDS}[^0-9k\-]*{KNUM}', t)
    if m: return (int(round(_n(m.group(1))*1000)), None)
    m = re.search(rf'{LOAD_WORDS}[^0-9\-]*{NUM}\s*(?:{UNIT_LB})?', t)
    if m: return (int(round(_n(m.group(1)))), None)

    # explicit units (kg/ton/t/tonne/lb)
    m = re.search(rf'{KNUM}\s*(?:{UNIT_LB})?\b', t)         # “7k (lb)”
    if m: return (int(round(_n(m.group(1))*1000)), None)
    m = re.search(rf'{NUM}\s*{UNIT_LB}\b', t)               # “7000 lb”
    if m: return (int(round(_n(m.group(1)))), None)
    m = re.search(rf'{NUM}\s*{UNIT_KG}\b', t)               # “2000 kg”
    if m: return (int(round(_n(m.group(1))*2.20462)), None)
    m = re.search(rf'{NUM}\s*{UNIT_TONNE}\b', t)            # “3.5 tonne / 3.5t”
    if m: return (int(round(_n(m.group(1))*2204.62)), None)
    m = re.search(rf'{NUM}\s*{UNIT_TON}\b', t)              # “5 ton(s)”
    if m: return (int(round(_n(m.group(1))*2000)), None)

    # safer fallback: a 4–5 digit number followed by lb(s)
    m = re.search(rf'\b(\d[\d,]{{3,5}})\s*(?:{UNIT_LB})\b', t)
    if m: return (int(m.group(1).replace(",", "")), None)

    # last resort: bare 4–5 digit near load words
    near = re.search(rf'(?:{LOAD_WORDS})\D{{0,12}}(\d{{4,5}})\b', t)
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

    # tires — direct mentions
    tire_pref = None
    if any(w in ql for w in ["non-marking","non marking","nonmarking"]): tire_pref = "non-marking cushion"
    if tire_pref is None and any(w in ql for w in ["cushion","press-on","press on"]): tire_pref = "cushion"
    if any(w in ql for w in ["pneumatic","air filled","air-filled","rough terrain tires","rt tires","knobby",
                             "off-road","outdoor tires","solid pneumatic","super elastic","foam filled","foam-filled"]):
        tire_pref = "pneumatic"

    # --- NEW: default tire based on environment if not specified
    if tire_pref is None:
        if indoor and not outdoor:
            tire_pref = "non-marking cushion" if re.search(r"non[-\s]?mark", ql) else "cushion"
        elif outdoor and not indoor:
            tire_pref = "pneumatic"

    return dict(
        cap_lbs=cap_lbs, height_in=height_in, aisle_in=aisle_in,
        power_pref=power_pref, indoor=indoor, outdoor=outdoor,
        narrow=narrow, tire_pref=tire_pref
    )

# ── accessors to raw row fields -----------------------------------------
def _capacity_of(row: Dict[str,Any]) -> Optional[float]:
    return _normalize_capacity_lbs(row)

def _height_of(row: Dict[str,Any]) -> Optional[float]:
    return _normalize_height_in(row)

def _aisle_of(row: Dict[str,Any]) -> Optional[float]:
    return _normalize_aisle_in(row)

def _power_of(row: Dict[str,Any]) -> str:
    return _text_from_keys(row, POWER_KEYS).lower()

def _tire_of(row: Dict[str,Any]) -> str:
    t = str(row.get("Tire Type","") or row.get("Tires","") or row.get("Tire","") or "").lower()
    # normalize common mentions
    if "non-mark" in t: return "non-marking cushion"
    if "cushion" in t or "press" in t: return "cushion"
    if "pneumatic" in t or "super elastic" in t or "solid" in t: return "pneumatic"
    return t  # unknown stays blank

def _safe_model_name(m: Dict[str, Any]) -> str:
    for k in ("Model","Model Name","model","code","name","Code"):
        if m.get(k):
            return str(m[k]).strip()
    return "N/A"

# --- small helper: tire suggestion reasoning -----------------------------
def _tire_guidance(want: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """Return (suggested_tire, rationale) based on environment if not explicitly set by user."""
    t = want.get("tire_pref")
    indoor, outdoor = want.get("indoor"), want.get("outdoor")
    if t:
        # Provide rationale even when user said it
        if "cushion" in t and indoor and not outdoor:
            return (t, "Indoor floors favor low rolling resistance and protect finished surfaces; non-marking avoids floor scuffs.")
        if "pneumatic" in t and outdoor and not indoor:
            return (t, "Outdoor/uneven surfaces need shock absorption and traction; (solid) pneumatic handles debris and rough pavement.")
        return (t, None)
    # Default suggestions
    if indoor and not outdoor:
        return ("cushion", "Indoor warehouse use → cushion tires (non-marking where floor care matters).")
    if outdoor and not indoor:
        return ("pneumatic", "Outdoor/yard work → (solid) pneumatic for stability and grip on uneven ground.")
    return (None, None)

# ---- Advanced tire chooser: maps user text -> your option names ----------
def pick_tire_advanced(user_q: str) -> tuple[str, str]:
    """
    Returns (option_name, rationale), where option_name is EXACTLY one of:
      - "Non-Marking Tires"
      - "Non-Marking Dual Tires"
      - "Solid Tires"
      - "Dual Solid Tires"
      - "Dual Tires"

    Logic blends environment, floor requirements, debris/roughness, capacity,
    and stability cues (ramps/high mast/long loads) to step up to 'dual' when needed.
    """
    t = (user_q or "").lower()

    # Signals
    indoorish = any(k in t for k in [
        "indoor", "inside", "warehouse", "production line", "manufacturing floor",
        "painted", "epoxy", "sealed", "polished", "smooth concrete", "food", "pharma",
        "clean floor", "cleanroom"
    ])
    outdoorish = any(k in t for k in [
        "outdoor", "yard", "dock", "dock yard", "lumber yard", "construction",
        "pavement", "asphalt", "gravel", "dirt", "parking lot"
    ])
    rough = any(k in t for k in [
        "rough terrain", "rough-terrain", "uneven", "broken", "pothole",
        "ruts", "curbs", "thresholds", "rails", "speed bumps"
    ])
    debris = any(k in t for k in [
        "debris", "nails", "screws", "scrap", "metal", "glass", "shavings", "chip"
    ])
    mixed = any(k in t for k in [
        "indoor/outdoor", "inside and outside", "both indoor and outdoor",
        "dock to yard", "mixed environment", "go outside and inside"
    ])
    nonmark_need = any(k in t for k in [
        "non-mark", "non mark", "no marks", "avoid marks", "black marks", "scuff", "no scuffs"
    ])

    # Stability: ramps, high mast, long/wide loads -> favor dual
    stability = any(k in t for k in [
        "ramp", "ramps", "slope", "incline", "grade", "dock plate",
        "high mast", "elevated", "tall stacks", "top heavy",
        "wide loads", "long loads", "coil", "paper", "rolls"
    ])

    # Capacity hint (reuse your existing parser)
    heavy = False
    try:
        cap_min, _ = _parse_capacity_lbs_intent(t)  # existing helper above in this file
        heavy = bool(cap_min and cap_min >= 7000)
    except Exception:
        pass

    # Decision tree
    # 1) Explicit non-marking requirement overrides (indoor bias)
    if nonmark_need or (indoorish and not outdoorish and any(k in t for k in ["concrete", "painted", "polished", "epoxy", "clean"])):
        if heavy or stability or mixed:
            return ("Non-Marking Dual Tires", "Clean/painted floors with heavier or mixed-duty usage — non-marking duals add stability and reduce scuffing.")
        return ("Non-Marking Tires", "Clean indoor floors — non-marking prevents black marks and scuffs.")

    # 2) Rough/debris -> Solid; step to Dual Solid if heavy/stability/mixed
    if rough or debris or (outdoorish and not indoorish and any(k in t for k in ["gravel", "dirt", "pothole", "broken"])):
        if heavy or stability or mixed:
            return ("Dual Solid Tires", "Rough/debris-prone surfaces — dual solid improves stability and is puncture-resistant.")
        return ("Solid Tires", "Rough/debris-prone surfaces — solid is puncture-proof and low maintenance.")

    # 3) Mixed indoors/outdoors (no explicit rough or non-mark need) -> Dual (plain)
    if mixed or (indoorish and outdoorish):
        return ("Dual Tires", "Mixed indoor/outdoor travel — dual improves stability and footprint across surfaces.")

    # 4) Pure outdoor (pavement, light duty) without rough/debris -> Solid
    if outdoorish and not indoorish:
        if heavy or stability:
            return ("Dual Solid Tires", "Outdoor duty with heavier loads or ramps — dual solid adds footprint and stability.")
        return ("Solid Tires", "Outdoor pavement — solid reduces flats and maintenance.")

    # 5) Pure indoor (no non-mark flag): default to non-marking; dual if heavy/stability
    if indoorish and not outdoorish:
        if heavy or stability:
            return ("Non-Marking Dual Tires", "Indoor with higher stability needs — dual non-marking reduces scuffs and adds stability.")
        return ("Non-Marking Tires", "Indoor warehouse floors — non-marking avoids scuffing on concrete/epoxy.")

    # 6) Ambiguous fallback
    if any(k in t for k in ["electric", "battery", "lithium"]) and any(k in t for k in ["concrete", "smooth", "painted"]):
        return ("Non-Marking Tires", "Likely indoor on smooth concrete with electric — non-marking prevents scuffs.")
    return ("Dual Tires", "Mixed/unspecified environment — dual provides added stability and versatility.")

# --- model filtering & ranking ------------------------------------------
def filter_models(user_q: str, limit: int = 5) -> List[Dict[str, Any]]:
    want = _parse_requirements(user_q)
    cap_need    = want["cap_lbs"]
    aisle_need  = want["aisle_in"]
    power_pref  = want["power_pref"]
    narrow      = want["narrow"]
    tire_pref   = want["tire_pref"]
    height_need = want["height_in"]

    scored: List[Tuple[float, Dict[str,Any]]] = []

    for m in models_raw:
        cap = _capacity_of(m) or 0.0
        powr = _power_of(m)
        tire = _tire_of(m)
        ais  = _aisle_of(m)
        hgt  = _height_of(m)
        reach_like = _is_reach_or_vna(m)
        three_wheel = _is_three_wheel(m)

        # HARD FILTERS
        if cap_need:
            if cap <= 0:   # unknown capacity rows are not acceptable when a minimum is set
                continue
            if cap < cap_need:
                continue
        if aisle_need and ais and ais > aisle_need:
            continue

        # SCORING
        s = 0.0
        if cap_need and cap:
            over = (cap - cap_need) / cap_need
            s += (2.0 - min(2.0, max(0.0, over))) if over >= 0 else -5.0

        if power_pref:
            s += 1.0 if _power_matches(power_pref, powr) else -0.8

        # tire preference scoring (never hard-block unknown tires)
        if tire_pref:
            s += 0.6 if (tire_pref in (tire or "")) else -0.2

        # environment-informed nudges
        if want["indoor"] and not want["outdoor"]:
            if tire == "cushion" or tire == "non-marking cushion": s += 0.4
            if "pneumatic" in (tire or ""): s -= 0.4
        if want["outdoor"] and not want["indoor"]:
            if "pneumatic" in (tire or ""): s += 0.5
            if "cushion" in (tire or ""): s -= 0.7

        if aisle_need:
            if ais: s += 0.8 if ais <= aisle_need else -1.0
            else:   s += 0.6 if (narrow and reach_like) else 0.0
        elif narrow:
            s += 0.8 if reach_like else -0.2

        if height_need and hgt:
            s += 0.5 if hgt >= height_need else -0.4

        # steer away from 3-wheel on heavier asks (≥4,500 lb)
        if cap_need and cap_need >= 4500 and three_wheel:
            s -= 0.8

        # small prior to avoid ties
        s += 0.05
        scored.append((s, m))

    if not scored:
        return []

    ranked = sorted(scored, key=lambda t: t[0], reverse=True)
    return [m for _, m in ranked[:limit]]

# ── build final prompt chunk --------------------------------------------
def generate_forklift_context(user_q: str, acct: Optional[Dict[str, Any]]) -> str:
    """
    Builds the final context block the AI returns to your UI.
    Now fills Tire Type + Attachments from the Excel-driven recommender so it only
    recommends items that actually exist in data/forklift_options_benefits.xlsx.
    """
    lines: List[str] = []
    if acct:
        lines.append(customer_block(acct))

    # Parse needs (env, capacity, etc.)
    want = _parse_requirements(user_q)
    env = "Indoor" if (want["indoor"] and not want["outdoor"]) else ("Outdoor" if (want["outdoor"] and not want["indoor"]) else "Mixed/Not specified")

    # Top models from your models.json filter
    hits = filter_models(user_q)

    # ---- Select tire + options from the Excel sheet
    rec = recommend_options_from_sheet(user_q, max_total=6)  # one tire + top options
    chosen_tire = rec.get("tire")
    other_opts = rec.get("others", [])

    # Split “others” into attachments vs non-attachments by keywords
    ATTACH_KEYS = [
        "sideshift", "side shift", "fork positioner", "positioner", "clamp",
        "rotator", "push/pull", "push pull", "bale", "carton", "appliance",
        "drum", "jib", "boom", "fork extension", "extensions", "spreader",
        "multi-pallet", "multi pallet", "double pallet", "triple pallet",
        "roll clamp", "paper roll", "coil ram", "carpet pole", "layer picker"
    ]
    attachments: List[Dict[str, str]] = []
    non_attachments: List[Dict[str, str]] = []
    for o in other_opts:
        name_l = o["name"].lower()
        if any(k in name_l for k in ATTACH_KEYS):
            attachments.append(o)
        else:
            non_attachments.append(o)

    # ------------- FORMAT OUTPUT (keeps your existing headings/flow) ------
    lines.append("Customer Profile:")
    lines.append(f"- Environment: {env}")
    if want.get("cap_lbs"):
        lines.append(f"- Capacity Min: {int(round(want['cap_lbs'])):,} lb")
    else:
        lines.append("- Capacity Min: Not specified")

    # Model block: top pick + alternates from hits
    lines.append("\nModel:")
    if hits:
        top = hits[0]
        top_name = _safe_model_name(top)
        lines.append(f"- Top Pick: {top_name}")
        if len(hits) > 1:
            alts = [ _safe_model_name(m) for m in hits[1:5] ]
            lines.append(f"- Alternates: {', '.join(alts)}")
        else:
            lines.append("- Alternates: None")
    else:
        lines.append("- Top Pick: N/A")
        lines.append("- Alternates: N/A")

    # Power & capacity fields
    lines.append("\nPower:")
    if want.get("power_pref"):
        lines.append(f"- {want['power_pref']}")
    else:
        # try to surface top model power if available
        lines.append(f"- {(_text_from_keys(hits[0], POWER_KEYS) if hits else 'Not specified') or 'Not specified'}")

    lines.append("\nCapacity:")
    lines.append(f"- {int(round(want['cap_lbs'])) if want.get('cap_lbs') else 'Not specified'}")

    # Tire Type from Excel recommender
    lines.append("\nTire Type:")
    if chosen_tire:
        lines.append(f"- {chosen_tire['name']} — {chosen_tire.get('benefit','').strip() or ''}".rstrip(" —"))
    else:
        lines.append("- Not specified")

    # Attachments from Excel recommender (only if present in your sheet)
    lines.append("\nAttachments:")
    if attachments:
        for a in attachments:
            benefit = (a.get("benefit","") or "").strip()
            lines.append(f"- {a['name']}" + (f" — {benefit}" if benefit else ""))
    else:
        lines.append("- Not specified")

    # Comparison block (kept simple and generic; you can customize later)
    lines.append("\nComparison:")
    if hits:
        lines.append("- Top pick vs peers: HELI advantages typically include tight turning (102 in).")
        lines.append("- We can demo against peers on your dock to validate turning, lift, and cycle times.")
    else:
        lines.append("- No model comparison available for the current filters.")

    # Sales Pitch & Objections (same content you already had downstream)
    lines.append("Sales Pitch Techniques:")
    lines.append("- Highlight low emissions of lithium models.")
    lines.append("- Emphasize versatility in mixed environments.")
    lines.append("- Discuss cost-effectiveness compared to competitors.")
    lines.append("- Share customer testimonials on performance and reliability.")

    lines.append("Common Objections:")
    lines.append("- I need better all-terrain capability.' — Ask: 'What specific terrains do you operate on?' | Reframe: 'This model excels in diverse conditions.' | Proof: 'Proven performance in various environments.' | Next: 'Shall we schedule a demo?.")
    lines.append("- Are lithium batteries reliable?' — Ask: 'What concerns do you have about battery performance?' | Reframe: 'Lithium offers longer life and less maintenance.' | Proof: 'Industry-leading warranty on batteries.' | Next: 'Would you like to see the specs?.")
    lines.append("- How does this compare to diesel?' — Ask: 'What are your priorities, emissions or power?' | Reframe: 'Lithium is cleaner and quieter.' | Proof: 'Lower operational costs over time.' | Next: 'Can I provide a cost analysis?.")
    lines.append("- What about service and support?' — Ask: 'What level of support do you expect?' | Reframe: 'We offer comprehensive service plans.' | Proof: 'Dedicated support team available.' | Next: 'Shall we discuss service options?.")
    lines.append("- Is it suitable for heavy-duty tasks?' — Ask: 'What tasks will you be performing?' | Reframe: 'Designed for robust applications.' | Proof: 'Tested under heavy loads.' | Next: 'Would you like to see a demonstration?.")
    lines.append("- I'm concerned about the upfront cost.' — Ask: 'What budget constraints are you working with?' | Reframe: 'Consider total cost of ownership.' | Proof: 'Lower energy and maintenance costs.' | Next: 'Can I help with financing options?.")

    # Pass through the original user question at the end (as your original did)
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

def allowed_models_block(allowed: List[str]) -> str:
    if not allowed:
        return "ALLOWED MODELS:\n(none – say 'No exact match from our lineup.')"
    return "ALLOWED MODELS:\n" + "\n".join(f"- {x}" for x in allowed)

# --- promo helpers: expose top-pick code / class / power -----------------
def _class_of(row: Dict[str, Any]) -> str:
    t = _text_from_keys(row, TYPE_KEYS)
    # Try to extract a forklift class like "I", "II", "III" from text
    m = re.search(r'\bclass\s*([ivx]+)\b', (t or ""), re.I)
    if m:
        roman = m.group(1).upper()
        # Map common roman numerals to I…V
        roman = roman.replace("V","V").replace("X","X")
        return roman
    # Fallback: sometimes stored directly as "I"/"II"/"III"
    t = (t or "").strip().upper()
    if t in {"I","II","III","IV","V"}:
        return t
    return ""

def model_meta_for(row: Dict[str, Any]) -> tuple[str, str, str]:
    """Return (model_code, class, power) for a models.json row."""
    code = _safe_model_name(row)  # already prefers Model/Code/Name fields
    cls = _class_of(row)
    pwr = _power_of(row) or ""
    return (code, cls, pwr)

def top_pick_meta(user_q: str) -> Optional[tuple[str, str, str]]:
    hits = filter_models(user_q, limit=1)
    if not hits:
        return None
    return model_meta_for(hits[0])

# --- debug helper --------------------------------------------------------
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
        three_wheel = _is_three_wheel(m)

        # scoring mirror
        s = 0.0
        if want["cap_lbs"] and cap:
            over = (cap - want["cap_lbs"]) / want["cap_lbs"]
            s += (2.0 - min(2.0, max(0.0, over))) if over >= 0 else -5.0
        if want["power_pref"]:
            s += 1.0 if _power_matches(want["power_pref"], powr) else -0.8
        if want["tire_pref"]:
            s += 0.6 if want["tire_pref"] in (tire or "") else -0.2
        if want["indoor"] and not want["outdoor"]:
            if tire == "cushion" or tire == "non-marking cushion": s += 0.4
            if "pneumatic" in (tire or ""): s -= 0.4
        if want["outdoor"] and not want["indoor"]:
            if "pneumatic" in (tire or ""): s += 0.5
            if "cushion" in (tire or ""): s -= 0.7
        if want["aisle_in"]:
            if ais: s += 0.8 if ais <= want["aisle_in"] else -1.0
            else:   s += 0.6 if (want["narrow"] and reach_like) else 0.0
        elif want["narrow"]:
            s += 0.8 if reach_like else -0.2
        if want["height_in"] and hgt:
            s += 0.5 if hgt >= want["height_in"] else -0.4
        if want["cap_lbs"] and want["cap_lbs"] >= 4500 and three_wheel:
            s -= 0.8
        s += 0.05

        rows.append({
            "model": _safe_model_name(m),
            "score": round(s, 3),
            "cap_lbs": cap, "power": powr, "tire": tire,
            "aisle_in": ais, "height_in": hgt,
            "three_wheel": three_wheel
        })
    rows.sort(key=lambda r: r["score"], reverse=True)
    return {"parsed": want, "top": rows[:limit]}
