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
    _pd = None  # pandas optional

# ─────────────────────────────────────────────────────────────────────────────
# Paths / constants (auto-detect your catalog .xlsx)
# ─────────────────────────────────────────────────────────────────────────────
def _find_catalog_xlsx() -> Optional[str]:
    env = os.environ.get("HELI_CATALOG_XLSX", "").strip()
    if env and os.path.isfile(env):
        return env
    here = os.path.dirname(__file__)
    candidates = [
        os.path.join(here, "data", "forklift_options_benefits.xlsx"),
        os.path.join(here, "forklift_options_benefits.xlsx"),
        os.path.join(os.getcwd(), "data", "forklift_options_benefits.xlsx"),
        os.path.join(os.getcwd(), "forklift_options_benefits.xlsx"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None

CATALOG_XLSX = _find_catalog_xlsx()
log.info("[ai_logic] Catalog path resolved: %s (exists=%s)", CATALOG_XLSX, bool(CATALOG_XLSX and os.path.isfile(CATALOG_XLSX)))

# Fallback seeds (never return empty if Excel missing)
_FALLBACK_TIRES: Dict[str, str] = {
    "Solid Tires": "Puncture-proof, low-maintenance tires for debris-prone floors.",
    "Dual Tires": "Wider footprint for better stability and traction on soft ground.",
    "Dual Solid Tires": "Combines puncture resistance with extra stability and load support.",
    "Non-Marking Tires": "Protect indoor floors by avoiding black scuff marks.",
    "Non-Marking Dual Tires": "Floor-safe traction with a wider, more stable footprint.",
}
_FALLBACK_OPTIONS: Dict[str, str] = {
    "Panel mounted Cab": "Enclosed cab panels for weather protection and operator comfort.",
    "Heater": "Keeps the cab warm for safer, more comfortable cold-weather operation.",
    "Air conditioner": "Cools the cab to reduce heat stress and maintain productivity.",
    "Glass Windshield with Wiper": "Clear forward visibility in rain and dust.",
    "Top Rain-proof Glass": "Overhead visibility while shielding the operator from rain.",
    "Rear Windshield Glass": "Improves rear visibility and shields from wind and debris.",
    "Dual Air Filter": "Enhanced engine air filtration for dusty environments.",
    "Pre air cleaner": "Cyclonic pre-cleaning extends main air filter life.",
    "Radiator protection bar": "Guards the radiator core from impacts.",
    "Air filter service indicator": "Tells you exactly when to change the filter, avoiding guesswork.",
    "LED Rear Working Light": "Bright, efficient rear lighting with long service life.",
    "Blue Light": "Pedestrian warning light to increase visibility in busy aisles.",
    "Backup Handle with Horn Button": "Safer reversing posture and quick horn access.",
    "Added cost for the cold storage package (for electric forklift)": "Components rated for freezer temps to reduce condensation and failures.",
    "HELI smart fleet management system FICS (Standard version（U.S. market supply suspended temporarily. Await notice.）": "Telematics for usage tracking, alerts, and basic analytics.",
    "HELI smart fleet management system FICS (Upgraded version（U.S. market supply suspended temporarily. Await notice.）": "Adds advanced reporting, diagnostics, and fleet insights.",
    "Portal access fee of FICS (each truck per year)（U.S. market supply suspended temporarily. Await notice.）": "Enables cloud portal access for data, reports, and alerts.",
    "3 Valve with Handle": "Adds a third hydraulic function to run basic attachments.",
    "4 Valve with Handle": "Enables two auxiliary functions for multi-function attachments.",
    "5 Valve with Handle": "Maximum hydraulic flexibility for specialized attachments.",
}
_FALLBACK_ATTACHMENTS: Dict[str, str] = {
    "Sideshifter": "Aligns loads without moving the truck; faster, cleaner placement.",
    "Fork Positioner": "Adjusts fork spread from the seat for mixed pallet sizes.",
    "Paper Roll Clamp": "Handles paper rolls gently without core damage.",
    "Push/ Pull (Slip-Sheet)": "Replaces pallets with slip-sheets to cut shipping costs and weight.",
    "Carpet Pole": "Handles coils, carpet, and tubing via a single pole/ram.",
    "Fork Extensions": "Temporarily lengthen forks for occasional oversized loads.",
    "Rotator": "Spins the carriage (often 180–360°) to dump bins or reorient loads.",
    "Bale Clamp": "Grips bales (paper, cotton, recycling) without pallets.",
    "Carton Clamp": "Pads squeeze large boxes/appliances so you can handle them pallet-free.",
    "Block Clamp": "Handles concrete/stone blocks with high holding force.",
    "Fork Clamp": "Forks that also clamp bulky or unpalletized items.",
    "Drum Clamp": "Lifts/tilts one or multiple 55-gal drums safely.",
    "Single Double Pallets Handler": "Switches between 1 wide pallet or 2 side-by-side.",
    "Load Stabilizer": "Top clamp plate holds light/unstable stacks steady while traveling/stacking.",
    "Bag Pusher": "Push plate drives big bags/sacks deep into a trailer or container.",
    "Bar Arm Clamp": "Side arms cradle and clamp bar/rod/tube bundles.",
}

# ─────────────────────────────────────────────────────────────────────────────
# Keys / normalizers (models)
# ─────────────────────────────────────────────────────────────────────────────
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

def _slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")

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

def _safe_model_name(m: Dict[str, Any]) -> str:
    for k in ("Model","Model Name","model","code","name","Code"):
        if m.get(k):
            return str(m[k]).strip()
    return "N/A"

# ─────────────────────────────────────────────────────────────────────────────
# Catalog loading
# ─────────────────────────────────────────────────────────────────────────────
def load_catalog_rows() -> List[Dict[str, str]]:
    """Read rows from the options/attachments/tires Excel. Returns [] if missing."""
    if not CATALOG_XLSX or not os.path.isfile(CATALOG_XLSX):
        log.warning("[ai_logic] Catalog .xlsx not found; continuing with fallbacks.")
        return []
    if _pd is None:
        log.warning("[ai_logic] pandas not available; cannot read Excel. Continuing with fallbacks.")
        return []
    try:
        df = _pd.read_excel(CATALOG_XLSX)
        cols = {c.lower().strip(): c for c in df.columns}
        records: List[Dict[str, str]] = []
        for _, row in df.iterrows():
            rec = {
                "Option":      str(row.get(cols.get("option", ""), "")).strip(),
                "Benefit":     str(row.get(cols.get("benefit", ""), "")).strip(),
                "Type":        str(row.get(cols.get("type", ""), "")).strip(),
                "Subcategory": str(row.get(cols.get("subcategory", ""), "")).strip(),
            }
            if not (rec["Option"] or rec["Benefit"] or rec["Type"] or rec["Subcategory"]):
                continue
            records.append(rec)
        log.info("[ai_logic] Loaded %d catalog rows from %s", len(records), os.path.basename(CATALOG_XLSX))
        return records
    except Exception as e:
        log.exception("[ai_logic] Failed reading Excel %s: %s", CATALOG_XLSX, e)
        return []

@lru_cache(maxsize=1)
def load_catalogs() -> Tuple[Dict[str,str], Dict[str,str], Dict[str,str]]:
    """
    Returns (options, attachments, tires) dicts mapping name -> benefit.
    """
    rows = load_catalog_rows()
    options: Dict[str,str] = {}
    attachments: Dict[str,str] = {}
    tires: Dict[str,str] = {}
    if not rows:
        # Fallbacks
        options.update(_FALLBACK_OPTIONS)
        attachments.update(_FALLBACK_ATTACHMENTS)
        tires.update(_FALLBACK_TIRES)
        log.info("[ai_logic] Using fallback buckets: options=%d attachments=%d tires=%d",
                 len(options), len(attachments), len(tires))
        return options, attachments, tires

    for r in rows:
        name = r.get("Option","").strip()
        ben  = r.get("Benefit","").strip()
        typ  = (r.get("Type","") or "").strip().lower()
        if not name:
            continue
        if typ == "tires" or typ == "tire":
            tires[name] = ben
        elif typ == "attachments" or typ == "attachment":
            attachments[name] = ben
        else:
            # treat everything else as 'options' (includes Telemetry)
            options[name] = ben

    # ensure we never return empty tires
    if not tires:
        tires.update(_FALLBACK_TIRES)

    log.info("[ai_logic] Buckets loaded: options=%d attachments=%d tires=%d", len(options), len(attachments), len(tires))
    return options, attachments, tires

def refresh_catalog_caches() -> None:
    """Allow external blueprints to refresh Excel caches at runtime."""
    load_catalog_rows.cache_clear()  # type: ignore[attr-defined]
    load_catalogs.cache_clear()      # type: ignore[attr-defined]
    log.info("[ai_logic] Catalog caches refreshed.")

# ─────────────────────────────────────────────────────────────────────────────
# Intent detection & environment flags
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
        "mentions_align": bool(re.search(r"\bsideshift(er)?\b|\bfork\s*positioner\b|\balign\b", ql)),
        "asks_non_mark": bool(re.search(r"non[-\s]?mark", ql)),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Ranking helpers
# ─────────────────────────────────────────────────────────────────────────────
def _prioritize_lighting(items: List[Dict[str, str]], q_lower: str) -> List[Dict[str, str]]:
    """If the query is about dark/low light, float lighting to the top."""
    if not any(k in q_lower for k in ("dark","dim","night","poor lighting","low light")):
        return items
    def is_light(x: Dict[str,str]) -> bool:
        t = (x.get("name","") + " " + x.get("benefit","")).lower()
        return any(w in t for w in ("light","led","beacon","work light","blue light","rear working light"))
    lights   = [x for x in items if is_light(x)]
    nonlight = [x for x in items if not is_light(x)]
    return lights + nonlight

def _drop_ac_when_cold(items: List[Dict[str, str]], q_lower: str) -> List[Dict[str, str]]:
    """In cold-only contexts, remove A/C so heater/cab/wipers rise."""
    if not any(k in q_lower for k in ("cold","freezer","subzero","winter")):
        return items
    new_list: List[Dict[str,str]] = []
    for x in items:
        t = (x.get("name","") + " " + x.get("benefit","")).lower()
        if "air conditioner" in t or "a/c" in t:
            continue
        new_list.append(x)
    return new_list

def _kw_score(q: str, name: str, benefit: str) -> float:
    """Lightweight scorer with context boosts/penalties."""
    ql = _lower(q)
    text = _lower(name + " " + (benefit or ""))

    score = 0.01  # small baseline
    for w in ("indoor","warehouse","outdoor","yard","dust","debris","visibility",
              "lighting","safety","cold","freezer","rain","snow","cab","comfort",
              "vibration","filtration","cooling","radiator","screen","pre air cleaner",
              "dual air filter","non-mark","pneumatic","cushion","heater","wiper",
              "windshield","work light","led"):
        if w in ql and w in text:
            score += 0.7

    if any(k in ql for k in ("cold","freezer","subzero","winter")) and any(
        k in text for k in ("cab","heater","defrost","wiper","rain-proof","glass","windshield","work light","led")
    ):
        score += 2.0
    if any(k in ql for k in ("cold","freezer","subzero","winter")) and ("air conditioner" in text or "a/c" in text):
        score -= 1.8

    if any(k in ql for k in ("dark","dim","poor lighting","night")) and any(
        k in text for k in ("light","led","beacon","blue light","work light")
    ):
        score += 1.5

    if any(k in ql for k in ("dust","debris","recycling","sawmill","dirty","foundry","yard")) and any(
        k in text for k in ("radiator","screen","pre air cleaner","dual air filter","filtration","belly pan","protection")
    ):
        score += 1.3

    if _TELEM_PAT.search(ql) and _TELEM_PAT.search(text):
        score += 2.2

    return score

def _rank_bucket(q: str, bucket: Dict[str, str], limit: int = 6) -> List[Dict[str, str]]:
    if not bucket:
        return []
    scored: List[Tuple[float, Dict[str,str]]] = []
    for name, benefit in bucket.items():
        s = _kw_score(q, name, benefit)
        scored.append((s, {"name": name, "benefit": benefit}))
    scored.sort(key=lambda t: t[0], reverse=True)
    out = [row for _, row in scored if _lower(row["name"]).strip()]
    return out[:limit] if isinstance(limit, int) and limit > 0 else out

# ─────────────────────────────────────────────────────────────────────────────
# Public: catalog query
# ─────────────────────────────────────────────────────────────────────────────
def recommend_options_from_sheet(user_q: str, limit: int = 6) -> Dict[str, List[Dict[str, str]]]:
    """
    Scenario-aware selector. Shows ONLY the sections the user requested
    (unless nothing requested, then returns a sensible default mix).
    """
    wants = _wants_sections(user_q)
    env   = _env_flags(user_q)
    ql    = (user_q or "").lower()

    options, attachments, tires = load_catalogs()
    telemetry = {n: b for n, b in options.items() if _TELEM_PAT.search(_lower(n + " " + (b or "")))}

    result: Dict[str, List[Dict[str,str]]] = {}

    def _safe_limit(items: List[Dict[str,str]]) -> List[Dict[str,str]]:
        return items[:limit] if (isinstance(limit, int) and limit > 0) else items

    # If the user explicitly asked, return ONLY those sections
    if wants["any"]:
        if wants["tires"]:
            ranked = _rank_bucket(user_q, tires, limit=0)
            if env["asks_non_mark"]:
                filtered = [t for t in ranked if "non-mark" in (t["name"] + " " + t.get("benefit","")).lower()]
                ranked = filtered or ranked
            result["tires"] = _safe_limit(ranked)

        if wants["attachments"]:
            ranked = _rank_bucket(user_q, attachments, limit=0)
            if env["indoor"] and not env["mentions_clamp"]:
                ranked = [a for a in ranked if not re.search(r"\bclamp\b", a["name"].lower())]
                # prefer alignment tools when indoor
                ranked.sort(key=lambda a: int(("sideshifter" in a["name"].lower()) or ("fork positioner" in a["name"].lower())), reverse=True)
            result["attachments"] = _safe_limit(ranked)

        if wants["options"]:
            ranked = _rank_bucket(user_q, options, limit=0)
            ranked = _drop_ac_when_cold(ranked, ql)
            ranked = _prioritize_lighting(ranked, ql)
            result["options"] = _safe_limit(ranked)

        if wants["telemetry"]:
            result["telemetry"] = _safe_limit(_rank_bucket(user_q, telemetry, limit=0))

        return result

    # Broad question: give helpful defaults (no tires unless hinted)
    ranked_opts = _rank_bucket(user_q, options, limit=0)
    ranked_opts = _drop_ac_when_cold(ranked_opts, ql)
    ranked_opts = _prioritize_lighting(ranked_opts, ql)

    ranked_atts = _rank_bucket(user_q, attachments, limit=0)
    if env["indoor"] and not env["mentions_clamp"]:
        ranked_atts = [a for a in ranked_atts if not re.search(r"\bclamp\b", a["name"].lower())]
        ranked_atts.sort(key=lambda a: int(("sideshifter" in a["name"].lower()) or ("fork positioner" in a["name"].lower())), reverse=True)

    result["options"] = _safe_limit(ranked_opts)
    result["attachments"] = _safe_limit(ranked_atts[:4])

    if _TIRES_PAT.search(user_q or ""):
        result["tires"] = _safe_limit(_rank_bucket(user_q, tires, limit=4))
    if _TELEM_PAT.search(user_q or ""):
        result["telemetry"] = _safe_limit(_rank_bucket(user_q, telemetry, limit=3))

    return result

# ─────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ─────────────────────────────────────────────────────────────────────────────
def render_sections_markdown(result: Dict[str, Any]) -> str:
    """
    Render only sections that have items. Skips empty/absent sections.
    Sections: 'tires', 'attachments', 'options', 'telemetry'
    """
    order = ["tires", "attachments", "options", "telemetry"]
    labels = {"tires": "Tires", "attachments": "Attachments", "options": "Options", "telemetry": "Telemetry"}

    lines: List[str] = []
    for key in order:
        arr = result.get(key) or []
        if not isinstance(arr, list) or not arr:
            continue

        section_lines: List[str] = []
        seen = set()
        for item in arr:
            if not isinstance(item, dict):
                # if something upstream sent a str by mistake, show it plainly
                item = {"name": str(item)}
            name = (item.get("name") or "").strip()
            if not name:
                continue
            nl = name.lower()
            if nl in seen:
                continue
            seen.add(nl)
            ben = (item.get("benefit") or "").strip().replace("\n", " ")
            section_lines.append(f"- {name}" + (f" — {ben}" if ben else ""))

        if section_lines:
            lines.append(f"**{labels[key]}:**")
            lines.extend(section_lines)

    return "\n".join(lines) if lines else "(no matching items)"

# Backward-compatible name some code calls elsewhere
def render_catalog_sections(result: Dict[str, Any], max_per_section: Optional[int] = None) -> str:
    """Thin wrapper; max_per_section kept for backward compat (ignored)."""
    return render_sections_markdown(result)

# Small helpers some modules import
def options_lookup_by_name(name: str) -> Optional[str]:
    options, attachments, tires = load_catalogs()
    for bucket in (options, attachments, tires):
        if name in bucket:
            return bucket[name]
    return None

def option_benefit(name: str) -> Optional[str]:
    return options_lookup_by_name(name)

# ─────────────────────────────────────────────────────────────────────────────
# Model recommendation stubs (kept minimal but won’t crash callers)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ModelHit:
    name: str
    score: float
    meta: Dict[str, Any]

@lru_cache(maxsize=1)
def _load_models_json() -> List[Dict[str, Any]]:
    # Your app loads models elsewhere; we provide a tiny safe fallback.
    candidates = [
        os.path.join(os.path.dirname(__file__), "data", "models.json"),
        os.path.join(os.getcwd(), "data", "models.json"),
        os.path.join(os.getcwd(), "models.json"),
    ]
    for p in candidates:
        try:
            if os.path.isfile(p):
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f) or []
        except Exception:
            pass
    return []

def filter_models(models: Optional[List[Dict[str,Any]]] = None) -> List[Dict[str,Any]]:
    return models or _load_models_json()

def generate_forklift_context(*args, **kwargs) -> str:
    """
    Compatible with older calls like generate_forklift_context(user_q, acct).
    Returns a lightweight prompt context summarizing detected needs.
    """
    user_q = args[0] if args else kwargs.get("user_q", "")
    env = _env_flags(user_q)
    bits = []
    if env["indoor"]: bits.append("Indoor use")
    if env["outdoor"]: bits.append("Outdoor/yard use")
    if env["cold"]: bits.append("Cold / freezer")
    if env["dark"]: bits.append("Low-light / lighting important")
    return " | ".join(bits) or "General use"

def select_models_for_question(user_q: str, k: int = 5) -> Tuple[List[ModelHit], List[str]]:
    """
    Very light keyword filter so upstream unpacking never fails.
    Returns (hits, allowed_model_codes).
    """
    models = filter_models()
    hits: List[ModelHit] = []
    ql = (user_q or "").lower()
    for m in models:
        name = _safe_model_name(m)
        t = (name + " " + json.dumps(m)).lower()
        score = 0.0
        if any(w in ql for w in ("rough", "yard", "outdoor")) and any(w in t for w in ("pneumatic","4x4","diesel","lp")):
            score += 2.0
        if any(w in ql for w in ("indoor","warehouse")) and any(w in t for w in ("cushion","electric","three wheel","3-wheel")):
            score += 1.8
        cap = _capacity_of(m) or 0
        if re.search(r"\b(\d{4,5})\b", ql):
            want = _num(re.search(r"\b(\d{4,5})\b", ql).group(1)) or 0
            score -= abs((cap or 0) - want) / 5000.0
        if score != 0.0:
            hits.append(ModelHit(name=name, score=score, meta=m))
    hits.sort(key=lambda h: h.score, reverse=True)
    allowed = [h.name for h in hits[:k]]
    return hits[:k], allowed

def allowed_models_block(hits: List[ModelHit]) -> str:
    if not hits:
        return "No obvious top picks based on the brief."
    lines = ["**Top Picks:**"]
    for h in hits:
        lines.append(f"- {h.name} (score {h.score:.2f})")
    return "\n".join(lines)

def model_meta_for(name: str) -> Dict[str, Any]:
    for m in _load_models_json():
        if _safe_model_name(m) == name:
            return m
    return {}

def top_pick_meta(hits: List[ModelHit]) -> Dict[str, Any]:
    return hits[0].meta if hits else {}

def debug_parse_and_rank(q: str) -> Dict[str, Any]:
    wants = _wants_sections(q)
    env = _env_flags(q)
    opts, atts, tires = load_catalogs()
    out = {
        "wants": wants,
        "env": env,
        "sample_options": _rank_bucket(q, opts, limit=5),
        "sample_attachments": _rank_bucket(q, atts, limit=5),
        "sample_tires": _rank_bucket(q, tires, limit=5),
    }
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Exports
# ─────────────────────────────────────────────────────────────────────────────
__all__ = [
    # Catalog IO / caches
    "load_catalogs", "load_catalog_rows", "refresh_catalog_caches",
    "options_lookup_by_name", "option_benefit",

    # Scenario picks & catalog renderers
    "recommend_options_from_sheet", "render_sections_markdown",
    "render_catalog_sections", "debug_parse_and_rank",

    # Model filtering & context
    "filter_models", "generate_forklift_context", "select_models_for_question",
    "allowed_models_block", "model_meta_for", "top_pick_meta",
]
