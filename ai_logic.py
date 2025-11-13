"""
ai_logic.py
Catalog helpers (tires / attachments / options), intent -> picks,
and model recommendation utilities used across the Heli AI app.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Imports / logging
# ─────────────────────────────────────────────────────────────────────────────
import os, json, re, logging, hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("ai_logic")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

try:
    import pandas as _pd
except Exception:
    _pd = None

# ─────────────────────────────────────────────────────────────────────────────
# Paths / constants
# ─────────────────────────────────────────────────────────────────────────────
_BASEDIR = os.path.dirname(__file__)
_OPTIONS_XLSX = os.environ.get(
    "HELI_CATALOG_XLSX",
    os.path.join(_BASEDIR, "data", "forklift_options_benefits.xlsx")
)
_MODELS_JSON = os.environ.get("HELI_MODELS_JSON", os.path.join(_BASEDIR, "data", "models.json"))
_ACCOUNTS_JSON = os.environ.get("HELI_ACCOUNTS_JSON", os.path.join(_BASEDIR, "data", "accounts.json"))

# Keys used to read models.json (your flexible sheets)
CAPACITY_KEYS = [
    "Capacity_lbs","capacity_lbs","Capacity","Rated Capacity","Load Capacity",
    "Capacity (lbs)","capacity","LoadCapacity","capacityLbs","RatedCapacity",
    "Load Capacity (lbs)","Rated Capacity (lbs)"
]
HEIGHT_KEYS = [
    "Lift Height_in","Max Lift Height (in)","Lift Height","Max Lift Height",
    "Mast Height","lift_height_in","LiftHeight","Lift Height (in)","Mast Height (in)"
]
AISLE_KEYS = [
    "Aisle_min_in","Aisle Width_min_in","Aisle Width (in)","Min Aisle (in)",
    "Right Angle Aisle (in)","Right-Angle Aisle (in)","RA Aisle (in)"
]
POWER_KEYS = ["Power","power","Fuel","fuel","Drive","Drive Type","Power Type","PowerType"]
TYPE_KEYS  = ["Type","Category","Segment","Class","Class/Type","Truck Type"]

# ─────────────────────────────────────────────────────────────────────────────
# Small utils
# ─────────────────────────────────────────────────────────────────────────────
def _slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")

def _make_code(name: str) -> str:
    return _slug(name)[:40] or "item"

def _norm_spaces(s: str) -> str:
    return " ".join((s or "").split())

def _lower(s: Any) -> str:
    return str(s or "").lower()

def _num(s: Any) -> Optional[float]:
    if s is None: return None
    m = re.search(r"-?\d+(?:\.\d+)?", str(s))
    return float(m.group(0)) if m else None

def _to_lbs(val: float, unit: str) -> float:
    u = unit.lower()
    if "kg" in u: return float(val) * 2.20462
    if "metric" in u and "ton" in u: return float(val) * 2204.62
    if "ton" in u: return float(val) * 2000.0
    return float(val)

def _to_inches(val: float, unit: str) -> float:
    u = unit.lower()
    if "ft" in u or "'" in u: return float(val) * 12.0
    return float(val)

def _text_from_keys(row: Dict[str,Any], keys: List[str]) -> str:
    for k in keys:
        v = row.get(k)
        if v:
            return str(v)
    return ""

def _num_from_keys(row: Dict[str,Any], keys: List[str]) -> Optional[float]:
    for k in keys:
        if k in row and str(row[k]).strip() != "":
            v = _num(row[k])
            if v is not None:
                return v
    return None

def _normalize_capacity_lbs(row: Dict[str,Any]) -> Optional[float]:
    for k in CAPACITY_KEYS:
        if k in row:
            s = str(row[k]).strip()
            if not s: 
                continue
            if re.search(r"\bkg\b", s, re.I):
                v = _num(s); return _to_lbs(v, "kg") if v is not None else None
            if re.search(r"\btonne\b|\bmetric\s*ton\b|\b(?<!f)\bt\b", s, re.I):
                v = _num(s); return _to_lbs(v, "metric ton") if v is not None else None
            if re.search(r"\btons?\b", s, re.I):
                v = _num(s); return _to_lbs(v, "ton") if v is not None else None
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

def _is_reach_or_vna(row: Dict[str,Any]) -> bool:
    t = (_text_from_keys(row, TYPE_KEYS) + " " + str(row.get("Model","")) + " " + str(row.get("Model Name",""))).lower()
    if any(word in t for word in ["reach","vna","order picker","turret"]):
        return True
    return bool(re.search(r"\b(cqd|rq|vna)\b", t))

def _is_three_wheel(row: Dict[str,Any]) -> bool:
    drive = (_lower(row.get("Drive Type")) + " " + _lower(row.get("Drive")))
    model = (_lower(row.get("Model")) + " " + _lower(row.get("Model Name")))
    return ("three wheel" in drive) or ("3-wheel" in drive) or (" 3 wheel" in drive) or ("sq" in model)

def _power_of(row: Dict[str,Any]) -> str:
    return _text_from_keys(row, POWER_KEYS).lower()

def _capacity_of(row: Dict[str,Any]) -> Optional[float]:
    return _normalize_capacity_lbs(row)

def _height_of(row: Dict[str,Any]) -> Optional[float]:
    return _normalize_height_in(row)

def _aisle_of(row: Dict[str,Any]) -> Optional[float]:
    return _normalize_aisle_in(row)

def _tire_of(row: Dict[str,Any]) -> str:
    t = _lower(row.get("Tire Type") or row.get("Tires") or row.get("Tire") or "")
    if "non-mark" in t: return "non-marking cushion"
    if "cushion" in t or "press" in t: return "cushion"
    if "pneumatic" in t or "super elastic" in t or "solid" in t: return "pneumatic"
    return t

def _safe_model_name(m: Dict[str, Any]) -> str:
    for k in ("Model","Model Name","model","code","name","Code"):
        if m.get(k):
            return str(m[k]).strip()
    return "N/A"

# ─────────────────────────────────────────────────────────────────────────────
# Catalog IO
# ─────────────────────────────────────────────────────────────────────────────
def _read_catalog_excel(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if _pd is None:
        log.warning("pandas not available; catalog reading will be minimal")
        return rows
    if not os.path.exists(path):
        log.warning("catalog excel not found: %s", path)
        return rows
    df = _pd.read_excel(path)
    for _, r in df.iterrows():
        rows.append({
            "Option": str(r.get("Option") or r.get("Name") or "").strip(),
            "Benefit": str(r.get("Benefit") or "").strip(),
            "Type": str(r.get("Type") or "").strip(),
            "Subcategory": str(r.get("Subcategory") or "").strip(),
        })
    return rows

@lru_cache(maxsize=1)
def load_catalog_rows() -> List[Dict[str, str]]:
    log.info("[ai_logic] Using catalog: %s (exists=%s)", _OPTIONS_XLSX, os.path.exists(_OPTIONS_XLSX))
    return _read_catalog_excel(_OPTIONS_XLSX)

def _bucketize(rows: List[Dict[str,str]]) -> Tuple[Dict[str,str], Dict[str,str], Dict[str,str]]:
    options: Dict[str,str] = {}
    attachments: Dict[str,str] = {}
    tires: Dict[str,str] = {}
    for r in rows:
        name = (r.get("Option") or "").strip()
        ben  = (r.get("Benefit") or "").strip()
        typ  = _lower(r.get("Type"))
        sub  = _lower(r.get("Subcategory"))
        if not name:
            continue
        # Flexible mapping:
        if "tire" in typ or "tyre" in typ or "tire" in sub or "tyre" in sub:
            tires[name] = ben
        elif "attach" in typ or "attachment" in typ or "clamp" in name.lower():
            attachments[name] = ben
        else:
            options[name] = ben
    return options, attachments, tires

@lru_cache(maxsize=1)
def load_catalogs() -> Tuple[Dict[str,str], Dict[str,str], Dict[str,str]]:
    rows = load_catalog_rows()
    options, attachments, tires = _bucketize(rows)
    log.info("[ai_logic] Loaded buckets: tires=%d attachments=%d options=%d", len(tires), len(attachments), len(options))
    return options, attachments, tires

def load_options() -> Dict[str,str]:
    return load_catalogs()[0]

def load_attachments() -> Dict[str,str]:
    return load_catalogs()[1]

def load_tires() -> Dict[str,str]:
    return load_catalogs()[2]

def load_tires_as_options() -> Dict[str,str]:
    # Sometimes older code expects tires inside 'options' space
    tires = load_tires()
    return {f"Tire: {k}": v for k, v in tires.items()}

def options_lookup_by_name(name: str) -> Optional[str]:
    n = (name or "").strip().lower()
    for bucket in load_catalogs():
        for k, v in bucket.items():
            if k.lower() == n:
                return v
    return None

def option_benefit(name: str) -> str:
    return options_lookup_by_name(name) or ""

def list_all_from_excel(section: str) -> List[Dict[str,str]]:
    s = (section or "").lower()
    o, a, t = load_catalogs()
    if s in ("option","options"): return [{"name": k, "benefit": v} for k, v in o.items()]
    if s in ("attachment","attachments"): return [{"name": k, "benefit": v} for k, v in a.items()]
    if s in ("tire","tires","tyre","tyres"): return [{"name": k, "benefit": v} for k, v in t.items()]
    return []

# ─────────────────────────────────────────────────────────────────────────────
# Intent / env flags
# ─────────────────────────────────────────────────────────────────────────────
_TIRES_PAT    = re.compile(r"\b(tires?|tyres?|tire\s*types?)\b", re.I)
_ATTACH_PAT   = re.compile(r"\b(attach(ment)?s?)\b", re.I)
_OPTIONS_PAT  = re.compile(r"\b(option|options)\b", re.I)
_TELEM_PAT    = re.compile(r"\b(fics|fleet\s*management|telemetry|portal)\b", re.I)

def _wants_sections(q: str) -> Dict[str, bool]:
    t = (q or "")
    return {
        "tires": bool(_TIRES_PAT.search(t)),
        "attachments": bool(_ATTACH_PAT.search(t)),
        "options": bool(_OPTIONS_PAT.search(t)),
        "telemetry": bool(_TELEM_PAT.search(t)),
        "any": any([
            _TIRES_PAT.search(t), _ATTACH_PAT.search(t), _OPTIONS_PAT.search(t), _TELEM_PAT.search(t)
        ])
    }

def _env_flags(q: str) -> Dict[str, bool]:
    ql = (q or "").lower()
    return {
        "cold": any(k in ql for k in ("cold","freezer","subzero","winter")),
        "indoor": any(k in ql for k in ("indoor","warehouse","inside","epoxy","polished","concrete")),
        "outdoor": any(k in ql for k in ("outdoor","yard","rain","snow","dust","gravel","dirt")),
        "dark": any(k in ql for k in ("dark","dim","night","poor lighting","low light")),
        "mentions_clamp": bool(re.search(r"\bclamp|paper\s*roll|bale|drum|carton|block\b", ql)),
        "mentions_align": bool(re.search(r"\balign|tight\s*aisle|narrow|staging\b", ql)),
        "mentions_widths": bool(re.search(r"\bvar(y|ied)\s*width|mixed\s*pallet|different\s*width\b", ql)),
        "asks_non_mark": bool(re.search(r"non[-\s]?mark", ql)),
    }

def parse_catalog_intent(user_q: str) -> dict:
    t = (user_q or "").strip().lower()
    which = None
    if ("attachments" in t and "options" in t) or "both" in t:
        which = "both"
    elif "attachments" in t or "attachment" in t:
        which = "attachments"
    elif "options" in t or "option" in t:
        which = "options"
    elif re.search(r"\b(tires?|tyres?)\b", t):
        which = "tires"
    elif _TELEM_PAT.search(t):
        which = "telemetry"
    list_all = bool(re.search(r'\b(list|show|give|display)\b.*\b(all|full|everything)\b', t))
    return {"which": which, "list_all": list_all}

# ─────────────────────────────────────────────────────────────────────────────
# Ranking helpers (accuracy upgrades)
# ─────────────────────────────────────────────────────────────────────────────
def _prioritize_lighting(items: List[Dict[str,Any]], q_lower: str) -> List[Dict[str,Any]]:
    """If the query is about dark/low light, float lighting to the top."""
    if not any(k in q_lower for k in ("dark","dim","night","poor lighting","low light")):
        return items
    def _is_light(x: Dict[str,Any]) -> bool:
        t = (x.get("name","") + " " + x.get("benefit","")).lower()
        return any(w in t for w in ("light","led","beacon","work light","blue light","rear working light"))
    lights   = [x for x in items if _is_light(x)]
    nonlight = [x for x in items if not _is_light(x)]
    return lights + nonlight

def _drop_ac_when_cold(items: List[Dict[str,Any]], q_lower: str) -> List[Dict[str,Any]]:
    """In cold-only contexts, remove A/C so heater/cab/wipers rise."""
    if not any(k in q_lower for k in ("cold","freezer","subzero","winter")):
        return items
    txt = [(x.get("name","") + " " + x.get("benefit","")).lower() for x in items]
    out: List[Dict[str,Any]] = []
    for i, x in enumerate(items):
        if "air conditioner" in txt[i] or "a/c" in txt[i]:
            continue
        out.append(x)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Main catalog selector
# ─────────────────────────────────────────────────────────────────────────────
def recommend_options_from_sheet(user_q: str, limit: int = 6) -> dict:
    """
    Scenario-aware selector. Shows ONLY the sections the user requested
    (unless nothing requested, then returns a sensible default mix).
    Cold: boost cab/heater/defrost/lights; hide A/C.
    Indoor: prefer Sideshifter/Fork Positioner; suppress clamps unless mentioned.
    """
    wants = _wants_sections(user_q)
    env   = _env_flags(user_q)

    options, attachments, tires = load_catalogs()

    # carve telemetry out of options by name/benefit text
    telemetry = {
        n: b for n, b in options.items()
        if _TELEM_PAT.search((n + " " + (b or "")).lower())
    }

    def rank(bucket: Dict[str, str]) -> List[Dict[str,Any]]:
        if not bucket:
            return []
        ql = (user_q or "").lower()
        scored: List[Tuple[float, Dict[str,Any]]] = []
        for name, benefit in bucket.items():
            text = (name + " " + (benefit or "")).lower()
            s = 0.01

            # generic alignment bumps
            for w in ("indoor","warehouse","outdoor","yard","dust","debris","visibility",
                      "lighting","safety","cold","freezer","rain","snow","cab","comfort",
                      "vibration","filtration","radiator","screen","pre air cleaner",
                      "dual air filter","heater","wiper","windshield","work light","led"):
                if w in ql and w in text:
                    s += 0.7

            # cold
            if env["cold"] and any(k in text for k in ("cab","heater","defrost","wiper","rain-proof","glass","windshield","work light","led")):
                s += 2.0
            if env["cold"] and ("air conditioner" in text or "a/c" in text):
                s -= 2.0

            # dark
            if env["dark"] and any(k in text for k in ("light","led","beacon","blue light","work light")):
                s += 1.5

            # indoor ergonomics/precision
            if env["indoor"]:
                if "sideshifter" in text or "side shifter" in text:
                    s += 1.6
                if "fork positioner" in text:
                    s += 1.4

            # debris/yard protection
            if any(k in ql for k in ("debris","yard","gravel","dirty","recycling","foundry","sawmill")) \
               and any(k in text for k in ("radiator","screen","pre air cleaner","dual air filter","filtration","belly pan","protection")):
                s += 1.3

            # telematics direct ask
            if _TELEM_PAT.search(ql) and _TELEM_PAT.search(text):
                s += 2.2

            scored.append((s, {"name": name, "benefit": benefit}))
        scored.sort(key=lambda t: t[0], reverse=True)
        ranked = [row for _, row in scored]

        # contextual post-adjustments
        if bucket is options:
            ranked = _drop_ac_when_cold(ranked, (user_q or "").lower())
            ranked = _prioritize_lighting(ranked, (user_q or "").lower())
        return ranked

    result: Dict[str, List[Dict[str,Any]]] = {}

    # If the user explicitly asked, show *only* those sections.
    if wants["any"]:
        if wants["tires"]:
            ranked_tires = rank(tires)
            if env["asks_non_mark"]:
                ranked_tires = [t for t in ranked_tires if "non-mark" in (t["name"] + " " + t.get("benefit","")).lower()] or ranked_tires
            result["tires"] = ranked_tires[:limit] if limit else ranked_tires

        if wants["attachments"]:
            ranked_atts = rank(attachments)
            # suppress clamps for indoor unless user actually mentions clamp/materials
            if env["indoor"] and not env["mentions_clamp"]:
                ranked_atts = [a for a in ranked_atts if not re.search(r"\bclamp\b", a["name"].lower())]
            # prefer alignment tools when indoor
            if env["indoor"]:
                ranked_atts.sort(key=lambda a: int(("sideshifter" in a["name"].lower()) or ("fork positioner" in a["name"].lower())), reverse=True)
            result["attachments"] = ranked_atts[:limit] if limit else ranked_atts

        if wants["options"]:
            ranked_opts = rank(options)
            result["options"] = ranked_opts[:limit] if limit else ranked_opts

        if wants["telemetry"]:
            r = rank(telemetry)
            result["telemetry"] = r[:limit] if limit else r

        return result

    # Broad question (no explicit section words): default to options + a few atts
    ranked_opts = rank(options)
    ranked_atts = rank(attachments)

    # Indoor default: prefer alignment, cut random clamps if not asked
    if env["indoor"] and not env["mentions_clamp"]:
        ranked_atts = [a for a in ranked_atts if not re.search(r"\bclamp\b", a["name"].lower())]
        ranked_atts.sort(key=lambda a: int(("sideshifter" in a["name"].lower()) or ("fork positioner" in a["name"].lower())), reverse=True)

    result["options"] = ranked_opts[:limit]
    result["attachments"] = ranked_atts[:4]

    # Only add tires/telemetry if they are hinted
    q = user_q or ""
    if _TIRES_PAT.search(q):
        result["tires"] = rank(tires)[:4]
    if _TELEM_PAT.search(q):
        result["telemetry"] = rank(telemetry)[:3]

    return result

# ─────────────────────────────────────────────────────────────────────────────
# Renderer
# ─────────────────────────────────────────────────────────────────────────────
def render_sections_markdown(result: dict) -> str:
    """
    Render only sections that have items. Skips empty/absent sections.
    Sections: 'tires', 'attachments', 'options', 'telemetry'
    """
    order = ["tires", "attachments", "options", "telemetry"]
    labels = {"tires": "Tires", "attachments": "Attachments", "options": "Options", "telemetry": "Telemetry"}

    lines: List[str] = []
    for key in order:
        arr = result.get(key) or []
        if not arr:
            continue

        seen = set()
        section_lines: List[str] = []
        for item in arr:
            name = (item.get("name") or "").strip()
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            ben = (item.get("benefit") or "").strip().replace("\n", " ")
            section_lines.append(f"- {name}" + (f" — {ben}" if ben else ""))

        if section_lines:
            lines.append(f"**{labels[key]}:**")
            lines.extend(section_lines)

    return "\n".join(lines) if lines else "(no matching items)"

# ─────────────────────────────────────────────────────────────────────────────
# Forklift model recommendation — lightweight + safe tuple returns
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_json(path: str) -> List[Dict[str,Any]]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            # allow {"rows":[...]} shapes
            data = data.get("rows") or data.get("data") or []
        return list(data) if isinstance(data, list) else []
    except Exception as e:
        log.warning("failed to load json %s: %s", path, e)
        return []

@lru_cache(maxsize=1)
def load_models() -> List[Dict[str,Any]]:
    rows = _load_json(_MODELS_JSON)
    log.info("[ai_logic] Loaded models=%d", len(rows))
    return rows

@lru_cache(maxsize=1)
def load_accounts() -> List[Dict[str,Any]]:
    rows = _load_json(_ACCOUNTS_JSON)
    log.info("[ai_logic] Loaded accounts=%d", len(rows))
    return rows

def _extract_capacity(ql: str) -> Optional[float]:
    # e.g., "5000 lb", "5,000 lbs", "2.5 ton", "3t"
    m = re.search(r"(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(lb|lbs|pounds?)\b", ql)
    if m:
        return float(m.group(1).replace(",", ""))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:t|tons?|tonnes?)\b", ql)
    if m:
        return _to_lbs(float(m.group(1)), "ton")
    return None

def _extract_height(ql: str) -> Optional[float]:
    # e.g., "lift 188 in", "to 240"", "16 ft"
    m = re.search(r"(\d+(?:\.\d+)?)\s*(in|inch|inches)\b", ql)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(ft|feet|')\b", ql)
    if m:
        return _to_inches(float(m.group(1)), "ft")
    return None

def filter_models(models: List[Dict[str,Any]], q: str) -> List[Dict[str,Any]]:
    """Very lightweight filtering by capacity/height/power cues from the question."""
    ql = (q or "").lower()
    want_cap = _extract_capacity(ql)
    want_ht  = _extract_height(ql)
    want_pwr = "diesel" if "diesel" in ql else ("lp" if "lp" in ql or "lpg" in ql or "propane" in ql else ("electric" if "electric" in ql or "lithium" in ql or "liq" in ql else None))

    out: List[Dict[str,Any]] = []
    for m in models:
        cap = _capacity_of(m) or 0
        ht  = _height_of(m) or 0
        pwr = _power_of(m)
        if want_cap and cap and cap < want_cap * 0.85:  # allow some tolerance
            continue
        if want_ht and ht and ht < want_ht * 0.90:
            continue
        if want_pwr and want_pwr not in pwr:
            continue
        out.append(m)
    return out

def select_models_for_question(q: str, k: int = 5) -> Tuple[List[Dict[str,Any]], List[str]]:
    """
    Returns (hits, allowed_names)
    - Always returns a tuple to avoid unpack errors.
    - Simple scoring by textual cues; falls back to first k if nothing matches.
    """
    models = load_models()
    if not models:
        return ([], [])

    base = filter_models(models, q)
    if not base:
        base = models[:]

    ql = (q or "").lower()
    scored: List[Tuple[float, Dict[str,Any]]] = []
    for m in base:
        txt = (str(m.get("Model","")) + " " + str(m.get("Model Name","")) + " " + str(m.get("Description",""))).lower()
        s = 0.0
        if any(w in ql for w in ("rough","outdoor","yard","pneumatic")) and "pneumatic" in txt:
            s += 1.5
        if any(w in ql for w in ("indoor","warehouse","smooth","cushion")) and "cushion" in txt:
            s += 1.5
        if "lithium" in ql and "lithium" in txt:
            s += 1.0
        if "reach" in ql and "reach" in txt:
            s += 1.0
        if "vna" in ql and "vna" in txt:
            s += 1.0
        cap = _capacity_of(m) or 0
        want_cap = _extract_capacity(ql)
        if want_cap:
            # prefer closer matches
            s -= abs(cap - want_cap) / max(want_cap, 1)
        scored.append((s, m))

    scored.sort(key=lambda t: t[0], reverse=True)
    hits = [m for _, m in scored[: max(k, 1)]]
    allowed = [ _safe_model_name(m) for m in hits ]
    return (hits, allowed)

def model_meta_for(m: Dict[str,Any]) -> Dict[str,Any]:
    return {
        "name": _safe_model_name(m),
        "power": _power_of(m),
        "capacity_lbs": _capacity_of(m),
        "lift_height_in": _height_of(m),
        "aisle_in": _aisle_of(m),
        "tire": _tire_of(m),
    }

def top_pick_meta(hits: List[Dict[str,Any]]) -> Dict[str,Any]:
    return model_meta_for(hits[0]) if hits else {}

def allowed_models_block(hits: List[Dict[str,Any]]) -> str:
    if not hits: return "(no models matched)"
    names = [ _safe_model_name(m) for m in hits ]
    return "\n".join(f"- {n}" for n in names)

def generate_forklift_context(q: str, hits: List[Dict[str,Any]]) -> str:
    if not hits:
        return "No close matches found in current model list."
    lines = ["Candidate models:"]
    for m in hits:
        meta = model_meta_for(m)
        lines.append(f"- {meta['name']} — {meta['power'] or 'power n/a'}, {int(meta['capacity_lbs'] or 0)} lb, lift {int(meta['lift_height_in'] or 0)} in, tire {meta['tire'] or 'n/a'}")
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# Debug helpers
# ─────────────────────────────────────────────────────────────────────────────
def debug_parse_and_rank(q: str) -> Dict[str,Any]:
    wants = _wants_sections(q)
    env = _env_flags(q)
    rec = recommend_options_from_sheet(q, limit=8)
    return {"wants": wants, "env": env, "result": rec}

# ─────────────────────────────────────────────────────────────────────────────
# Router compatibility shims (prevent import warnings)
# ─────────────────────────────────────────────────────────────────────────────
def refresh_catalog_caches() -> dict:
    """Return simple counts so the router can confirm caches are warm."""
    options, attachments, tires = load_catalogs()
    return {
        "options_count": len(options),
        "attachments_count": len(attachments),
        "tires_count": len(tires),
    }

def render_catalog_sections(user_q: str, limit: int = 6) -> dict:
    """
    Older router calls this. Delegate to the new selector so behavior stays consistent.
    """
    return recommend_options_from_sheet(user_q, limit)

# ─────────────────────────────────────────────────────────────────────────────
# “Defined but not accessed” silencer (legacy)
# ─────────────────────────────────────────────────────────────────────────────
def _is_attachment(name: str) -> bool:
    nl = _lower(name)
    return any(k in nl for k in (
        "clamp","sideshift","positioner","rotator","boom","pole","ram",
        "fork extension","extensions","push/ pull","push/pull",
        "slip-sheet","slipsheet","bale","carton","drum","bag push","load stabilizer"
    ))

# ─────────────────────────────────────────────────────────────────────────────
# Exports
# ─────────────────────────────────────────────────────────────────────────────
__all__ = [
    # Catalog IO / caches
    "load_catalogs", "load_catalog_rows", "refresh_catalog_caches",
    "load_options", "load_attachments", "load_tires_as_options", "load_tires",
    "options_lookup_by_name", "option_benefit",

    # Scenario picks & catalog renderers
    "recommend_options_from_sheet", "render_sections_markdown",
    "render_catalog_sections", "parse_catalog_intent",
    "generate_catalog_mode_response", "list_all_from_excel",

    # Model filtering & context
    "filter_models", "generate_forklift_context", "select_models_for_question",
    "allowed_models_block", "model_meta_for", "top_pick_meta",

    # Debug
    "debug_parse_and_rank",

    # Intentional small helpers sometimes imported
    "_num_from_keys",
]

# Back-compat: generate_catalog_mode_response delegates to recommend + markdown
def generate_catalog_mode_response(user_q: str, limit: int = 6) -> str:
    data = recommend_options_from_sheet(user_q, limit)
    return render_sections_markdown(data)

# Keep Pylance happy for legacy imports
_LEGACY_EXPORTS: Dict[str, Any] = {
    "list_all_from_excel": list_all_from_excel,
    "num_from_keys": _num_from_keys,
    "is_attachment": _is_attachment,
}
if hashlib.md5(str(sorted(_LEGACY_EXPORTS.keys())).encode()).hexdigest():
    pass
