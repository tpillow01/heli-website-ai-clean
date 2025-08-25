# ai_logic.py
"""
Helper module: account lookup + model parsing/filtering + prompt context builder
- Grounds picks strictly on models.json
- Enforces capacity/tire/height constraints
- Adds environment-aware attachments
- Provides optional competitor peer lines if data present (heli_comp_models.json)
"""
from __future__ import annotations
import json, re, difflib, os
from typing import List, Dict, Any, Tuple, Optional

# ── load JSON once -------------------------------------------------------
def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

accounts_raw = _load_json("accounts.json")

_models_raw = _load_json("models.json")
models_raw: List[Dict[str, Any]]
if isinstance(_models_raw, list):
    models_raw = _models_raw
elif isinstance(_models_raw, dict) and isinstance(_models_raw.get("models"), list):
    models_raw = _models_raw["models"]
elif isinstance(_models_raw, dict) and isinstance(_models_raw.get("data"), list):
    models_raw = _models_raw["data"]
else:
    models_raw = []

print(f"[ai_logic] Loaded accounts: {len(accounts_raw)} | models: {len(models_raw)}")

# Optional competitor data (used to seed comparison lines)
COMP_PEERS_PATH = os.path.join("data", "heli_comp_models.json")
try:
    with open(COMP_PEERS_PATH, "r", encoding="utf-8") as f:
        comp_rows = json.load(f)
    if not isinstance(comp_rows, list):
        comp_rows = []
except Exception:
    comp_rows = []

# ── key aliases ----------------------------------------------------------
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
TIRE_KEYS  = ["Tire Type", "Tires", "Tire", "TireType"]

# ── account helpers ------------------------------------------------------
def get_account(text: str) -> Dict[str, Any] | None:
    low = text.lower()
    for acct in accounts_raw:
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

# ── parsing & format helpers --------------------------------------------
def _to_lbs(val: float, unit: str) -> float:
    u = unit.lower()
    if "kg" in u:
        return float(val) * 2.20462
    if "ton" in u and "metric" in u:
        return float(val) * 2204.62
    if re.search(r"\bton(s)?\b", u):
        return float(val) * 2000.0
    return float(val)

def _to_inches(val: float, unit: str) -> float:
    u = unit.lower()
    if "ft" in u or "'" in u:
        return float(val) * 12.0
    return float(val)

def _num(s: Any) -> Optional[float]:
    if s is None:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", str(s))
    return float(m.group(0)) if m else None

def _first_text(row: Dict[str,Any], keys: List[str]) -> str:
    for k in keys:
        v = row.get(k)
        if v not in (None, ""):
            return str(v)
    return ""

def _normalize_capacity_lbs(row: Dict[str,Any]) -> Optional[float]:
    for k in CAPACITY_KEYS:
        if k in row and str(row[k]).strip() != "":
            s = str(row[k])
            if re.search(r"\bkg\b", s, re.I):
                v = _num(s);  return _to_lbs(v, "kg") if v is not None else None
            if re.search(r"\btonne|\bmetric\s*ton|\b(?<!f)\bt\b", s, re.I):
                v = _num(s);  return _to_lbs(v, "metric ton") if v is not None else None
            if re.search(r"\btons?\b", s, re.I):
                v = _num(s);  return _to_lbs(v, "ton") if v is not None else None
            v = _num(s);      return v
    return None

def _normalize_height_in(row: Dict[str,Any]) -> Optional[float]:
    for k in HEIGHT_KEYS:
        if k in row and str(row[k]).strip() != "":
            s = str(row[k])
            if re.search(r"\bft\b|'", s, re.I):
                v = _num(s);  return _to_inches(v, "ft") if v is not None else None
            v = _num(s);      return v
    return None

def _normalize_aisle_in(row: Dict[str,Any]) -> Optional[float]:
    for k in AISLE_KEYS:
        if k in row and str(row[k]).strip() != "":
            s = str(row[k])
            if re.search(r"\bft\b|'", s, re.I):
                v = _num(s);  return _to_inches(v, "ft") if v is not None else None
            v = _num(s);      return v
    return None

def _power_of(row: Dict[str,Any]) -> str:
    return _first_text(row, POWER_KEYS).strip().lower()

def _tire_of(row: Dict[str,Any]) -> str:
    return _first_text(row, TIRE_KEYS).strip().lower()

def _safe_model_name(m: Dict[str, Any]) -> str:
    for k in ("Model","Model Name","model","code","name","Code"):
        if m.get(k):
            return str(m[k]).strip()
    return "N/A"

def _fmt_lb(n: Optional[float]) -> str:
    if n is None: return "Not specified"
    try:
        return f"{int(round(float(n))):,} lb"
    except Exception:
        return "Not specified"

def _fmt_in(n: Optional[float]) -> str:
    if n is None: return "Not specified"
    try:
        return f"{int(round(float(n))):,} in"
    except Exception:
        return "Not specified"

# ── capacity intent parser ----------------------------------------------
def _parse_capacity_lbs_intent(text: str) -> tuple[Optional[int], Optional[int]]:
    """Return (min_required_lb, max_allowed_lb)."""
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

    # ranges
    m = re.search(rf'{KNUM}\s*-\s*{KNUM}', t)
    if m: return (int(round(_n(m.group(1))*1000)), int(round(_n(m.group(2))*1000)))
    m = re.search(rf'{NUM}\s*-\s*{NUM}\s*(?:{UNIT_LB})?', t)
    if m:
        a, b = int(round(_n(m.group(1)))), int(round(_n(m.group(2))))
        return (min(a,b), max(a,b))
    m = re.search(rf'between\s+{NUM}\s+and\s+{NUM}', t)
    if m:
        a, b = int(round(_n(m.group(1)))), int(round(_n(m.group(2))))
        return (min(a,b), max(a,b))

    # bounds
    m = re.search(rf'(?:up to|max(?:imum)?)\s+{NUM}\s*(?:{UNIT_LB})?', t)
    if m: return (None, int(round(_n(m.group(1)))))
    m = re.search(rf'(?:at least|minimum|min)\s+{NUM}\s*(?:{UNIT_LB})?', t)
    if m: return (int(round(_n(m.group(1)))), None)

    # within load words
    m = re.search(rf'{LOAD_WORDS}[^0-9k\-]*{KNUM}', t)
    if m: return (int(round(_n(m.group(1))*1000)), None)
    m = re.search(rf'{LOAD_WORDS}[^0-9\-]*{NUM}', t)
    if m: return (int(round(_n(m.group(1)))), None)

    # safe fallback: only bare 4–5 digits near load words
    near = re.search(rf'{LOAD_WORDS}\D{{0,12}}(\d{{4,5}})\b', t)
    if near: return (int(near.group(1)), None)

    return (None, None)

# ── parse requirements ---------------------------------------------------
def _parse_requirements(q: str) -> Dict[str,Any]:
    ql = (q or "").lower()

    cap_min, cap_max = _parse_capacity_lbs_intent(ql)
    cap_lbs = cap_min

    # height (avoid aisle captures)
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
        except: pass
    if height_in is None:
        m = re.search(r'(?:lift|raise|reach|height|clearance|mast)\D{0,12}(\d[\d,\.]*)\s*(ft|feet|\'|in|\"|inches)', ql)
        if m:
            raw, unit = m.group(1), m.group(2)
            try:
                height_in = _to_inches(float(raw.replace(",","")), "ft") if unit in ("ft","feet","'") \
                            else float(raw.replace(",",""))
            except: pass

    # aisle width (incl. RA)
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

    # environment keywords
    outdoor = any(w in ql for w in [
        "outdoor","yard","dock yard","construction","lumber yard","gravel","dirt","uneven","rough",
        "pavement","parking lot","rough terrain","rough-terrain","all-terrain","all terrain"
    ])
    indoor  = any(w in ql for w in ["indoor","warehouse","inside","factory floor","distribution center","dc"])
    dock    = any(w in ql for w in ["dock","loading dock","dock work","dock-use","dock use"])

    # power preference (preference, not hard block)
    power_pref = None
    if any(w in ql for w in ["zero emission","zero-emission","emissions free","emissions-free","eco friendly",
                             "eco-friendly","battery powered","battery-powered","battery","lithium",
                             "li-ion","li ion","lead acid","lead-acid","electric"]):
        power_pref = "electric"
    if "diesel" in ql: power_pref = "diesel"
    if any(w in ql for w in ["lpg","propane","lp gas","gas (lpg)","gas-powered","gas powered"]): power_pref = "lpg"

    # tires (hard requirement if outdoor/rough is present)
    tire_pref = None
    if any(w in ql for w in ["pneumatic","air filled","air-filled","rough terrain tires","rt tires","knobby",
                             "off-road","outdoor tires","solid pneumatic","solid-pneumatic","super elastic",
                             "super-elastic","foam filled","foam-filled","all-terrain","all terrain"]):
        tire_pref = "pneumatic"
    if any(w in ql for w in ["cushion","press-on","press on","non-marking","nonmarking","non marking"]):
        # store 'cushion', but we'll special-case 'non-marking' for advice
        tire_pref = tire_pref or "cushion"

    # narrow aisle heuristic
    narrow  = ("narrow aisle" in ql) or ("very narrow" in ql) or ("vna" in ql) or ("turret" in ql) \
              or ("reach truck" in ql) or ("stand-up reach" in ql) \
              or (aisle_in is not None and aisle_in <= 96)

    return dict(
        cap_lbs=cap_lbs, height_in=height_in, aisle_in=aisle_in,
        power_pref=power_pref, indoor=indoor, outdoor=outdoor, dock=dock,
        narrow=narrow, tire_pref=tire_pref, raw=ql
    )

# ── environment rules ----------------------------------------------------
_PNEU_WORDS = ("pneumatic","solid pneumatic","solid-pneumatic","super elastic","super-elastic","foam filled","foam-filled","air")
_CUSH_WORDS = ("cushion","press-on","press on")

def _tires_match_outdoor(tire_text: str) -> bool:
    t = tire_text.lower()
    return any(w in t for w in _PNEU_WORDS)

def _tires_match_cushion(tire_text: str) -> bool:
    t = tire_text.lower()
    return any(w in t for w in _CUSH_WORDS)

def _attachments_for(want: Dict[str,Any]) -> List[str]:
    out: List[str] = []
    if want["dock"]:
        out += ["Fork positioner", "Blue pedestrian light", "LED work lights"]
    if want["outdoor"]:
        out += ["Side-shifter", "Cab or weather package", "Rear work light"]
    if want["indoor"]:
        out += ["Charger/telemetry (opportunity charging)"]
    # common safety
    out += ["Backup alarm"]
    # dedupe preserve order
    seen = set()
    kept = []
    for a in out:
        key = a.lower()
        if key not in seen:
            kept.append(a)
            seen.add(key)
    return kept

# ── feature extractors for rows -----------------------------------------
def _capacity_of(row: Dict[str,Any]) -> Optional[float]:
    return _normalize_capacity_lbs(row)

def _height_of(row: Dict[str,Any]) -> Optional[float]:
    return _normalize_height_in(row)

def _aisle_of(row: Dict[str,Any]) -> Optional[float]:
    return _normalize_aisle_in(row)

# ── competitor peers (optional) -----------------------------------------
def _heli_family_from_row(row: Dict[str,Any]) -> str:
    p = _power_of(row)
    tires = _tire_of(row)
    # extremely light family buckets for peer matching
    if "diesel" in p or "lpg" in p or "ic" in p:
        return "IC Cushion" if _tires_match_cushion(tires) else "IC Pneumatic"
    if "electric" in p or "lith" in p or "lead" in p:
        return "Electric Cushion" if _tires_match_cushion(tires) else "Electric Pneumatic"
    return "Other/Unknown"

def _find_peer_lines(top_row: Dict[str,Any], K: int = 4) -> List[str]:
    if not comp_rows:
        return []
    fam = _heli_family_from_row(top_row)
    cap = _capacity_of(top_row)
    turn = _aisle_of(top_row) or None  # if you stored turning in aisle keys
    width = _num(top_row.get("Overall Width (in)") or top_row.get("Overall Width") or "")

    scored = []
    for r in comp_rows:
        score = 0.0
        score += 0 if r.get("family") == fam else 30.0
        try:
            # penalize distance in capacity (per 100 lb)
            c = float(r.get("capacity_lb")) if r.get("capacity_lb") is not None else None
        except Exception:
            c = None
        if cap is not None and c is not None:
            score += abs(cap - c) / 100.0
        # light nudges
        if width and r.get("width_in"):
            try: score += abs(width - float(r["width_in"])) / 50.0
            except Exception: pass
        if turn and r.get("turning_in"):
            try: score += abs(turn - float(r["turning_in"])) / 50.0
            except Exception: pass
        scored.append((score, r))

    scored.sort(key=lambda x: x[0])
    peers = []
    for _, r in scored[:K]:
        brand = r.get("brand") or ""
        mdl   = r.get("model") or ""
        cc    = r.get("capacity_lb")
        trn   = r.get("turning_in")
        wid   = r.get("width_in")
        fuel  = r.get("fuel") or ""
        parts = []
        if cc is not None:
            try: parts.append(f"{int(round(float(cc))):,} lb")
            except Exception: pass
        if trn is not None:
            try: parts.append(f"turn {int(round(float(trn))):,} in")
            except Exception: pass
        if wid is not None:
            try: parts.append(f"width {int(round(float(wid))):,} in")
            except Exception: pass
        if fuel:
            parts.append(fuel if isinstance(fuel, str) else str(fuel))
        peers.append(f"- {brand} {mdl} — " + "; ".join(parts) if parts else f"- {brand} {mdl}")
    return peers

# ── model filtering & ranking -------------------------------------------
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
        tires= _tire_of(m)
        ais  = _aisle_of(m)
        hgt  = _height_of(m)
        # === HARD FILTERS =================================================
        # required capacity
        if cap_need and (cap <= 0 or cap < cap_need): 
            continue
        # aisle clearance (if model advertises)
        if aisle_need and ais and ais > aisle_need:
            continue
        # height (if asked and model advertises)
        if height_need and hgt and hgt < height_need:
            continue
        # outdoor / pneumatic rule
        if want["outdoor"]:
            if not _tires_match_outdoor(tires):
                continue
        # requested tire type
        if tire_pref == "pneumatic" and not _tires_match_outdoor(tires):
            continue
        if tire_pref == "cushion" and not _tires_match_cushion(tires):
            continue
        # ==================================================================

        s = 0.0
        # soft scores
        if cap_need and cap:
            over = (cap - cap_need) / cap_need
            s += (2.0 - min(2.0, max(0.0, over))) if over >= 0 else -5.0

        if power_pref:
            s += 1.0 if power_pref in powr else -0.5

        if tire_pref:
            if tire_pref == "pneumatic":
                s += 0.8 if _tires_match_outdoor(tires) else -0.8
            elif tire_pref == "cushion":
                s += 0.6 if _tires_match_cushion(tires) else -0.6

        if aisle_need:
            if ais: s += 0.6 if ais <= aisle_need else -0.8
            elif narrow: s += 0.3  # tiny benefit if the truck is reach/VNA pattern (not used here)

        if height_need and hgt: 
            s += 0.4 if hgt >= height_need else -0.4

        # tiny prior to avoid tie churn
        s += 0.03
        scored.append((s, m))

    if not scored:
        return []

    ranked = sorted(scored, key=lambda t: t[0], reverse=True)
    return [m for _, m in ranked[:limit]]

# ── build final prompt chunk --------------------------------------------
def _best_and_alts(models: List[Dict[str,Any]]) -> tuple[Optional[Dict[str,Any]], List[Dict[str,Any]]]:
    if not models: return None, []
    return models[0], models[1:5]

def _capacity_line(m: Dict[str,Any]) -> str:
    cap = _capacity_of(m)
    return _fmt_lb(cap)

def _tires_line(m: Dict[str,Any]) -> str:
    t = _tire_of(m)
    if not t: return "Not specified"
    # normalize a bit
    if _tires_match_outdoor(t): return "Pneumatic / solid-pneumatic"
    if _tires_match_cushion(t): return "Cushion"
    return t

def generate_forklift_context(user_q: str, acct: Dict[str, Any] | None) -> str:
    """
    NOTE: pass ONLY the raw user question from app.py so parsers aren't
    poisoned by account numbers (SIC/ZIP/etc.). We add the account block here.
    """
    want = _parse_requirements(user_q)
    hits = filter_models(user_q, limit=5)
    top, alts = _best_and_alts(hits)

    lines: List[str] = []
    if acct:
        lines.append(customer_block(acct))

    # Seed a compact, LLM-friendly context block (not rendered directly)
    lines.append("<span class=\"section-label\">Parsed Need:</span>")
    lines.append(f"- Environment: {'Outdoor' if want['outdoor'] else ('Indoor' if want['indoor'] else 'Mixed/Unspecified')}")
    lines.append(f"- Aisle Limit: { _fmt_in(want['aisle_in']) if want['aisle_in'] else 'Not specified' }")
    lines.append(f"- Power Preference: { (want['power_pref'] or 'Not specified').title() if want['power_pref'] else 'Not specified' }")
    lines.append(f"- Capacity Min: { _fmt_lb(want['cap_lbs']) if want['cap_lbs'] else 'Not specified' }")
    lines.append(f"- Tire Preference: { 'Pneumatic' if want['tire_pref']=='pneumatic' else ('Cushion' if want['tire_pref']=='cushion' else 'Not specified') }")
    if want["dock"]:
        lines.append(f"- Dock Work: Yes")

    if hits:
        lines.append("<span class=\"section-label\">Candidate Heli Models (ranked):</span>")
        for i, m in enumerate(hits, 1):
            lines.append(f"- {i}. {_safe_model_name(m)} | Power: {_first_text(m, POWER_KEYS) or 'N/A'} | Capacity: {_capacity_line(m)} | Tires: {_tires_line(m)}")
    else:
        lines.append("No matching models found in the provided data.\n")

    # Suggested attachments (used by the LLM in Attachments section)
    sugg = _attachments_for(want)
    if sugg:
        lines.append("<span class=\"section-label\">Suggested Attachments:</span>")
        lines.append("- " + "; ".join(sugg))

    # Seed peer lines for the *top pick* (if any) – helps the model write 'Comparison:'
    if top and comp_rows:
        peer_lines = _find_peer_lines(top, K=4)
        if peer_lines:
            lines.append("<span class=\"section-label\">Peer Models (for Comparison):</span>")
            lines += peer_lines

    # Finally append the raw user ask last
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
    # Keep order (best first) so your app can present top pick first if it preserves ordering
    return "ALLOWED MODELS:\n" + "\n".join(f"- {x}" for x in allowed)

# --- debug utility for your /api/debug_recommend route -------------------
def debug_parse_and_rank(user_q: str, limit: int = 10):
    want = _parse_requirements(user_q)
    rows = []
    for m in models_raw:
        cap = _capacity_of(m) or 0.0
        powr = _power_of(m)
        tire = _tire_of(m)
        ais  = _aisle_of(m)
        hgt  = _height_of(m)

        # replicate scoring (without hard filters) for transparency
        s = -9999.0
        # mirror actual filter decision first (if filtered out, show why)
        filtered_out = False
        reasons = []
        if want["cap_lbs"] and (cap <= 0 or cap < want["cap_lbs"]):
            filtered_out = True; reasons.append("cap below need")
        if want["aisle_in"] and ais and ais > want["aisle_in"]:
            filtered_out = True; reasons.append("aisle too large")
        if want["height_in"] and hgt and hgt < want["height_in"]:
            filtered_out = True; reasons.append("height short")
        if want["outdoor"] and not _tires_match_outdoor(tire):
            filtered_out = True; reasons.append("needs pneumatic/outdoor")
        if want["tire_pref"] == "pneumatic" and not _tires_match_outdoor(tire):
            filtered_out = True; reasons.append("requested pneumatic")
        if want["tire_pref"] == "cushion" and not _tires_match_cushion(tire):
            filtered_out = True; reasons.append("requested cushion")

        if not filtered_out:
            s = 0.0
            if want["cap_lbs"] and cap:
                over = (cap - want["cap_lbs"]) / want["cap_lbs"]
                s += (2.0 - min(2.0, max(0.0, over))) if over >= 0 else -5.0
            if want["power_pref"]: 
                s += 1.0 if want["power_pref"] in powr else -0.5
            if want["tire_pref"]:
                if want["tire_pref"] == "pneumatic":
                    s += 0.8 if _tires_match_outdoor(tire) else -0.8
                elif want["tire_pref"] == "cushion":
                    s += 0.6 if _tires_match_cushion(tire) else -0.6
            if want["aisle_in"]:
                if ais: s += 0.6 if ais <= want["aisle_in"] else -0.8
            if want["height_in"] and hgt: s += 0.4 if hgt >= want["height_in"] else -0.4
            s += 0.03

        rows.append({
            "model": _safe_model_name(m),
            "filtered_out": filtered_out,
            "filter_reasons": reasons,
            "score": round(s, 3) if not filtered_out else None,
            "cap_lbs": cap, "power": powr, "tire": tire,
            "aisle_in": ais, "height_in": hgt
        })
    # show passing rows ranked first, then filtered
    rows_pass = [r for r in rows if not r["filtered_out"]]
    rows_fail = [r for r in rows if r["filtered_out"]]
    rows_pass.sort(key=lambda r: r["score"], reverse=True)
    return {"parsed": want, "top": rows_pass[:limit], "filtered": rows_fail[:limit]}
