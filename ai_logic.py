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

# === Typed catalog loader (options / attachments / tires) =====================
# Uses the same Excel file path you already have in _OPTIONS_XLSX.
# Accepts either "name" or "option" as the name column, plus required "type" and optional "benefit".

def _read_typed_catalog_df():
    """
    Reads your Excel and normalizes columns:
      - name:   from 'name' or 'option'
      - type:   required; values: option | attachment | tire
      - benefit: optional; defaults to ''
    Returns a pandas DataFrame or None if missing/invalid.
    """
    if _pd is None or not os.path.exists(_OPTIONS_XLSX):
        return None
    # Use openpyxl for xlsx; falls back if not available
    try:
        df = _pd.read_excel(_OPTIONS_XLSX, engine="openpyxl")
    except Exception:
        df = _pd.read_excel(_OPTIONS_XLSX)

    cols = {str(c).lower().strip(): c for c in df.columns}
    name_col = cols.get("name") or cols.get("option")  # support your older sheet ("Option")
    type_col = cols.get("type")
    ben_col  = cols.get("benefit")

    if not name_col or not type_col:
        # We need at least name+type to build typed catalogs
        return None

    use_cols = [name_col, type_col] + ([ben_col] if ben_col else [])
    df = df[use_cols].copy()
    rename_map = {name_col: "name", type_col: "type"}
    if ben_col:
        rename_map[ben_col] = "benefit"
    df.rename(columns=rename_map, inplace=True)

    # Normalize fields
    df["name"]    = df["name"].astype(str).str.strip()
    df["type"]    = df["type"].astype(str).str.strip().str.lower()
    if "benefit" not in df.columns:
        df["benefit"] = ""
    else:
        df["benefit"] = df["benefit"].astype(str).str.strip()

    # Keep only non-empty names
    df = df[df["name"] != ""]
    return df

def load_catalogs() -> tuple[dict, dict, dict]:
    """
    Returns three dictionaries keyed by exact 'name':
      - options:     { name: benefit }
      - attachments: { name: benefit }
      - tires:       { name: benefit }

    These are built from your Excel 'type' column (option | attachment | tire).
    """
    df = _read_typed_catalog_df()
    if df is None or df.empty:
        return {}, {}, {}

    opts = df[df["type"] == "option"][["name", "benefit"]]
    atts = df[df["type"] == "attachment"][["name", "benefit"]]
    tirs = df[df["type"] == "tire"][["name", "benefit"]]

    options = {r["name"]: (r.get("benefit") or "") for _, r in opts.iterrows()}
    attachments = {r["name"]: (r.get("benefit") or "") for _, r in atts.iterrows()}
    tires = {r["name"]: (r.get("benefit") or "") for _, r in tirs.iterrows()}
    return options, attachments, tires

# === Sales Catalog helpers ===============================================

def _env_tags_for_name(nl: str) -> list[str]:
    """
    Best-use environment tags inferred from the option/attachment name.
    Deterministic so reps get predictable tags. Used later by the catalog mode.
    """
    nl = (nl or "").lower()
    tags = set()

    # Tires
    if "tire" in nl:
        if "non-mark" in nl:
            tags.add("Indoor • clean floors")
        if "dual" in nl:
            tags.add("Mixed • stability/ramps")
        if "solid" in nl:
            tags.add("Outdoor • rough/debris")
        if not tags:
            tags.add("General use")

    # Lighting / pedestrian awareness
    if "light" in nl or "beacon" in nl or "blue spot" in nl or "red side" in nl:
        tags.update(["Pedestrian-heavy", "Low-light / 2nd shift"])
    if "radar" in nl or "ops" in nl:
        tags.add("Safety/Policy")

    # Cab / climate
    if any(k in nl for k in ["heater","cab","windshield","wiper","air conditioner"]):
        tags.add("Outdoor weather")
    if "cold storage" in nl:
        tags.add("Cold storage / freezer")

    # Filtration / protection
    if any(k in nl for k in ["radiator","belly pan","screen","air filter","pre air"]):
        tags.add("Dust/debris sites")

    # Hydraulics / controls / seats
    if "valve" in nl or "finger control" in nl:
        tags.update(["Multi-function attachments", "Ergonomics"])
    if "seat" in nl:
        tags.add("Long shifts / ergonomics")

    # Fuel / LPG
    if "lpg" in nl or "fuel" in nl:
        tags.add("IC/LPG operations")

    # Attachments
    if "sideshift" in nl or "side shift" in nl:
        tags.update(["Tight aisles", "Frequent pallet alignment"])
    if "positioner" in nl:
        tags.update(["Mixed pallet widths", "Fast changeovers"])
    if "paper roll clamp" in nl:
        tags.add("Paper/converter plants")
    if "push" in nl and "slip" in nl:
        tags.add("Slip-sheeted goods (no pallets)")
    if "carpet" in nl or "ram" in nl:
        tags.add("Rolled goods")
    if "extension" in nl:
        tags.add("Occasional long loads")

    if not tags:
        tags.add("General use")
    return sorted(tags)

# Basic attachment detector used for grouping
_ATTACHMENT_KEYS = [
    "sideshift","side shift","fork positioner","positioner","clamp","rotator",
    "push/pull","push pull","slip-sheet","slipsheet","bale","carton","appliance",
    "drum","jib","boom","fork extension","extensions","spreader","multi-pallet",
    "double pallet","triple pallet","roll clamp","paper roll","coil ram",
    "carpet pole","layer picker","pole"
]

def _is_attachment(name_lower: str) -> bool:
    return any(k in name_lower for k in _ATTACHMENT_KEYS) and "tire" not in name_lower

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

# --- Fallback flag extractor (only if you don't already have one) ---
if '_need_flags_from_text' not in globals():
# --- Unified flag extractor (strong patterns; safe to always define/override) ---
    def _need_flags_from_text(user_q: str) -> dict:
        t = (user_q or "").lower()
        f = {}

        # Environment / duty
        f["indoor"]        = re.search(r'\bindoor|warehouse|inside|factory|production|line\b', t) is not None
        f["outdoor"]       = re.search(r'\boutdoor|yard|dock|lot|asphalt|gravel|dirt|parking\b', t) is not None
        f["mixed"]         = ("indoor" in t and "outdoor" in t) or ("both" in t) or ("mixed" in t)
        f["yard"]          = "yard" in t
        f["soft_ground"]   = re.search(r'soft\s*ground|mud|sand', t) is not None
        f["gravel"]        = "gravel" in t or "dirt" in t
        f["rough"]         = re.search(r'rough|uneven|broken|pothole|curb|rail|speed\s*bumps?', t) is not None
        f["debris"]        = re.search(r'debris|nails|screws|scrap|glass|shavings|chips?', t) is not None
        f["puncture"]      = re.search(r'puncture|flats?|tire\s*damage', t) is not None
        f["heavy_loads"]   = re.search(r'\b(7k|7000|8k|8000)\b|heavy\s*loads?|coil|paper\s*rolls?', t) is not None
        f["long_runs"]     = re.search(r'long\s*shifts?|multi[-\s]?shift|continuous', t) is not None

        # Tires (one-line versions you asked for)
        f["non_marking"]   = bool(re.search(r'non[-\s]?mark|no\s*marks?|black\s*marks?|avoid\s*marks?|scuff', t))

        # Work content → attachments
        f["alignment_frequent"] = re.search(r'align|line\s*up|tight\s*aisles|staging', t) is not None
        f["varied_width"]       = bool(re.search(r'vary|mixed\s*pallet|different\s*width|multiple\s*widths|mix\s*of\s*\d+\s*["in]?\s*(and|&)\s*\d+\s*["in]?\s*pallets?', t))
        f["paper_rolls"]        = re.search(r'paper\s*roll|newsprint|tissue', t) is not None
        f["slip_sheets"]        = re.search(r'slip[-\s]?sheet', t) is not None
        f["carpet"]             = "carpet" in t or "textile" in t
        f["long_loads"]         = bool(re.search(r'long|oversize|over[-\s]?length|overhang|\b\d+\s*[- ]?ft\b|\b\d+\s*foot\b|\b\d+\s*feet\b|crate[s]?', t))
        f["weighing"]           = re.search(r'weigh|scale|check\s*weight', t) is not None

        # Visibility / safety
        f["pedestrian_heavy"]   = re.search(r'pedestrian|foot\s*traffic|busy|congested|blind\s*corner|walkway', t) is not None
        f["poor_visibility"]    = re.search(r'low\s*light|dim|night|second\s*shift|poor\s*lighting', t) is not None

        # Hydraulics / functions
        f["extra_hydraulics"]   = "4th function" in t or "fourth function" in t
        f["multi_function"]     = "multiple clamp" in t or "multiple attachments" in t
        f["ergonomics"]         = re.search(r'ergonomic|fatigue|wrist|reach|comfort', t) is not None

        # Climate / environment
        f["cold"]               = re.search(r'cold|freezer|refrigerated|winter', t) is not None
        f["hot"]                = re.search(r'hot|heat|summer|foundry|high\s*ambient', t) is not None

        # Policy / compliance
        f["speed_control"]      = re.search(r'limit\s*speed|speeding|zoned\s*speed', t) is not None
        f["ops_required"]       = re.search(r'ops|operator\s*presence|osha|insurance|audit|policy', t) is not None

        # Misc site/config
        f["tall_operator"]      = "tall operator" in t or "headroom" in t
        f["high_loads"]         = re.search(r'high\s*mast|tall\s*stacks|top\s*heavy|elevated', t) is not None
        f["special_color"]      = "special color" in t or "paint" in t
        f["rigging"]            = "rigging" in t or "lift with crane" in t
        f["telematics"]         = "fics" in t or "fleet management" in t or "telematics" in t

        # Power hints
        f["power_lpg"]          = re.search(r'\b(lpg|propane|lp[-\s]?gas)\b', t) is not None
        f["electric"]           = re.search(r'\b(lithium|li[-\s]?ion|electric|battery)\b', t) is not None
        return f

# === REPLACEMENT: Excel-driven tires / attachments / options recommender ===
from typing import Dict, Any, List, Tuple

def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    # Unify punctuation/spacing variants
    s = re.sub(r"[\/\-–—_]+", " ", s)           # slashes & dashes -> space
    s = re.sub(r"[()]+", " ", s)                # drop parentheses
    s = re.sub(r"\bslip\s*[- ]?\s*sheet\b", "slipsheet", s)  # "slip sheet" -> "slipsheet"
    s = re.sub(r"\s+", " ", s)                  # squeeze spaces
    return s

def _lut_by_name(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Build a lookup by normalized name and seed a single alias so
    all push/pull slip-sheet variants map to the Excel row
    'Push/ Pull (Slip-Sheet)'.
    """
    lut: Dict[str, Dict[str, Any]] = {}

    # First, index by normalized sheet names
    for it in items or []:
        nm_sheet = (it.get("name") or it.get("Name") or it.get("option") or "").strip()
        if not nm_sheet:
            continue
        lut[_norm(nm_sheet)] = it

    # Alias: all "push/pull slip sheet" variants → the canonical sheet row
    canonical = next((it for it in items if (it.get("name") or it.get("option")) == "Push/ Pull (Slip-Sheet)"), None)
    if canonical:
        # normalized key that all variants reduce to via _norm
        lut["push pull slipsheet"] = canonical

    return lut

def _get_with_default(lut: Dict[str, Dict[str, Any]], name: str, default_benefit: str) -> Dict[str, Any]:
    """Pull row from Excel; if not present, return synthetic row with default benefit.
    Always include a 'best_for' hint for tires so catalog renderers never KeyError."""
    row = lut.get(_norm(name))

    # Best-for hints (kept lightweight & consistent with your UI copy)
    nlow = (name or "").strip().lower()
    tire_best_for = ""
    if "non-marking dual" in nlow:
        tire_best_for = "Indoor clean floors + added stability/ramps"
    elif "non-marking" in nlow:
        tire_best_for = "Indoor, polished/epoxy floors"
    elif "dual solid" in nlow:
        tire_best_for = "Rough/debris sites with heavier loads"
    elif nlow == "dual tires" or (" dual" in nlow and "tire" in nlow):
        tire_best_for = "Mixed indoor/outdoor travel"
    elif "solid tire" in nlow or (("solid" in nlow) and ("tire" in nlow)):
        tire_best_for = "Outdoor yards, debris/rough pavement"

    if row:
        nm  = row.get("name") or row.get("Name") or name
        ben = (row.get("benefit") or row.get("Benefit") or "").strip() or default_benefit
        out = {"name": nm, "benefit": ben}
    else:
        out = {"name": name, "benefit": default_benefit}

    # Only attach 'best_for' when we have a meaningful hint (mostly tires)
    if tire_best_for:
        out["best_for"] = tire_best_for
    return out

def _pick_tire_from_flags(flags: Dict[str, Any], excel_lut: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Decision order (most specific first) + hard fallback so a tire is ALWAYS suggested:
    - Explicit non-marking → Non-Marking (dual if mixed/outdoor/heavy/stability)
    - Rough/debris/puncture → Solid (dual solid if soft ground/heavy)
    - Yard/soft ground/mixed/outdoor → Dual Tires
    - FINAL FALLBACKS (general knowledge):
        * Indoor-only → Non-Marking Tires
        * Outdoor-only → Solid Tires
        * Mixed/unspecified → Dual Tires
    """
    f = {k: bool(flags.get(k)) for k in [
        "non_marking", "rough", "debris", "puncture",
        "outdoor", "indoor", "soft_ground", "yard", "gravel", "mixed",
        "heavy_loads", "long_runs", "high_loads"
    ]}

    heavy_or_stability = f["heavy_loads"] or f["high_loads"]

    # Non-marking first
    if f["non_marking"]:
        if f["outdoor"] or f["mixed"] or f["yard"] or heavy_or_stability:
            return _get_with_default(
                excel_lut, "Non-Marking Dual Tires",
                "Dual non-marking tread — keeps floors clean with extra footprint for stability."
            )
        return _get_with_default(
            excel_lut, "Non-Marking Tires",
            "Non-marking compound — prevents black marks on painted/epoxy floors."
        )

    # Rough / debris / puncture
    if f["puncture"] or f["debris"] or f["rough"] or f["gravel"]:
        if f["soft_ground"] or f["heavy_loads"]:
            return _get_with_default(
                excel_lut, "Dual Solid Tires",
                "Puncture-proof dual solids — added footprint and stability on rough/soft ground."
            )
        return _get_with_default(
            excel_lut, "Solid Tires",
            "Puncture-proof solid tires — best for debris-prone or rough surfaces."
        )

    # Yard / soft ground / frequent outside or mixed
    if f["yard"] or f["soft_ground"] or f["outdoor"] or f["mixed"]:
        return _get_with_default(
            excel_lut, "Dual Tires",
            "Wider footprint for traction and stability on soft or uneven ground."
        )

    # === FALLBACKS so we NEVER return None ===
    if f["indoor"] and not f["outdoor"]:
        return _get_with_default(
            excel_lut, "Non-Marking Tires",
            "Indoor warehouse floors — non-marking avoids scuffs on concrete/epoxy."
        )
    if f["outdoor"] and not f["indoor"]:
        return _get_with_default(
            excel_lut, "Solid Tires",
            "Outdoor pavement/yard — reduced flats and lower maintenance."
        )
    # Mixed/unspecified
    return _get_with_default(
        excel_lut, "Dual Tires",
        "Mixed or unspecified environment — dual improves footprint and stability."
    )

def _pick_attachments_from_excel(flags: Dict[str, Any], excel_lut: Dict[str, Dict[str, Any]], max_items: int = 6) -> List[Dict[str, Any]]:
    """
    Picks only relevant attachments. If nothing explicit is triggered but it's clearly a pallet/indoor scenario,
    we add Sideshifter as a sensible default (and Fork Positioner only when 'varied_width' is true).
    """
    out: List[Dict[str, Any]] = []
    t = (flags.get("_raw_text") or "").lower()
    pallets_mentioned = bool(re.search(r'\bpallet(s)?\b', t))

    def maybe_add(names: List[Tuple[str, str]]):
        for nm, default_ben in names:
            row = excel_lut.get(_norm(nm))
            if row:
                out.append(_get_with_default(excel_lut, nm, default_ben))

    # Explicit cues → attachments
    if flags.get("alignment_frequent"):
        maybe_add([("Sideshifter", "Aligns loads without repositioning the truck—faster, cleaner placement.")])
    if flags.get("varied_width"):
        maybe_add([("Fork Positioner", "Adjust fork spread from the seat for mixed pallet sizes.")])
    if flags.get("paper_rolls"):
        maybe_add([("Paper Roll Clamp", "Secure, damage-reducing handling for paper rolls.")])
    if flags.get("slip_sheets"):
        maybe_add([("Push/ Pull (Slip-Sheet)", "Handles slip-sheeted cartons—eliminates pallets and cuts freight weight.")])
    if flags.get("carpet"):
        maybe_add([("Carpet Pole", "Safe handling of carpet or coil-like rolled goods.")])
    if flags.get("long_loads"):
        maybe_add([("Fork Extensions", "Supports longer or over-length loads safely.")])
    if flags.get("weighing"):
        maybe_add([("Hydraulic weight system (+/-10% difference)", "On-truck weighing for faster ship/receive checks.")])

    # If nothing explicit fired, add a sensible default for indoor pallet work
    if not out and (flags.get("indoor") or pallets_mentioned):
        maybe_add([("Sideshifter", "Aligns loads without repositioning the truck—faster, cleaner placement.")])
        # only add Fork Positioner when we genuinely detect width variation
        if flags.get("varied_width"):
            maybe_add([("Fork Positioner", "Adjust fork spread from the seat for mixed pallet sizes.")])

    # Dedup & trim
    seen = set()
    uniq = []
    for it in out:
        k = _norm(it["name"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(it)
        if len(uniq) >= max_items:
            break
    return uniq

def _pick_options_from_excel(flags: Dict[str, Any], excel_lut: Dict[str, Dict[str, Any]], max_items: int = 6) -> List[Dict[str, Any]]:
    """
    Non-attachment options (lighting, cab/comfort, protection/cooling, controls,
    brakes, telematics, speed control, LPG bits, cold storage, OPS).
    Only returns items present in your Excel by name.
    """
    picks: List[Dict[str, Any]] = []

    def add_if_present(name: str, default_benefit: str):
        row = excel_lut.get(_norm(name))
        if not row:
            return
        # Skip items whose Benefit mentions supply suspended (as in your sheet note)
        ben_txt = (row.get("benefit") or row.get("Benefit") or "").lower()
        if "suspend" in ben_txt:
            return
        item = _get_with_default(excel_lut, name, default_benefit)
        if all(_norm(item["name"]) != _norm(x["name"]) for x in picks):
            picks.append(item)

    # Pedestrian / visibility
    if flags.get("pedestrian_heavy") or flags.get("poor_visibility"):
        add_if_present("LED Rotating Light", "High-visibility 360° beacon to alert pedestrians.")
        add_if_present("Blue spot Light", "Projects a visible spot ahead/behind to warn pedestrians at intersections.")
        add_if_present("Red side line Light", "Creates a visual exclusion zone along the sides for safer aisles.")
        add_if_present("Visible backward radar, if applicable", "Audible/visual alerts for obstacles while reversing.")
        add_if_present("Rear Working Light", "Improves rear visibility for night or dim aisles.")
        add_if_present("LED Rear Working Light", "Bright, low-draw rear work lighting for staging areas.")
        add_if_present("Blue Light", "Extra directional warning for blind corners.")

    # Cab / comfort / climate
    if flags.get("outdoor") or flags.get("cold") or flags.get("hot"):
        add_if_present("Panel mounted Cab", "Weather protection for outdoor duty; reduces operator fatigue.")
        add_if_present("Heater", "Keeps operator warm in cold environments; improves productivity.")
        add_if_present("Air conditioner", "Comfort in hot conditions; maintains focus during long shifts.")
        add_if_present("Glass Windshield with Wiper", "Visibility in rain/snow and wind; safer outdoor travel.")
        add_if_present("Top Rain-proof Glass", "Protects from precipitation while keeping overhead visibility.")
        add_if_present("Rear Windshield Glass", "Reduces wind and rain ingress from the rear.")

    # Protection / cooling / filtration
    if flags.get("rough") or flags.get("debris") or flags.get("yard") or flags.get("gravel"):
        add_if_present("Radiator protection bar", "Shields radiator core from impacts and debris.")
        add_if_present("Steel Belly Pan", "Protects underside from debris, ruts, and snagging.")
        add_if_present("Removable radiator screen", "Keeps fins clear of debris; easier maintenance on dusty sites.")
        add_if_present("Dual Air Filter", "Improved filtration for dusty yards.")
        add_if_present("Pre air cleaner", "Cyclonic pre-separation for heavy dust; longer filter life.")
        add_if_present("Air filter service indicator", "Visual indicator for timely filter service on dusty routes.")
        add_if_present("Tilt or Steering cylinder boot", "Protects cylinder rods from grit and pitting.")

    # Hydraulics / controls (beyond attachments)
    if flags.get("extra_hydraulics") or flags.get("multi_function"):
        add_if_present("3 Valve with Handle", "Adds third function for attachments; simple handle control.")
        add_if_present("4 Valve with Handle", "Adds fourth hydraulic circuit for complex tools.")
        add_if_present("5 Valve with Handle", "Maximum hydraulic flexibility for specialized attachments.")
        add_if_present("Finger control system(2valve),if applicable.should work together with MSG65 seat",
                       "Compact fingertip controls; less reach and wrist strain.")
        add_if_present("Finger control system(3valve),if applicable.should work together with MSG65 seat",
                       "Fingertip precision for three hydraulic functions; reduced fatigue.")
        add_if_present("Finger control system(4valve), if applicable, should work together with MSG65 seat",
                       "Compact fingertip controls for complex four-function attachments.")

    # Brakes / axles (heavy IC duty)
    if flags.get("heavy_loads") or flags.get("long_runs") or flags.get("outdoor"):
        add_if_present("Wet Disc Brake axle for CPCD50-100", "Lower maintenance braking under heavy duty cycles.")

    # Speed / safety systems
    if flags.get("speed_control"):
        add_if_present("Speed Control system (not for diesel engine)", "Zone or global speed limiting for safer aisles.")
    if flags.get("ops_required"):
        add_if_present("Full OPS", "Operator presence system for OSHA-aligned safety interlocks.")

    # Telematics
    if flags.get("telematics"):
        add_if_present("HELI smart fleet management system FICS (Standard version（U.S. market supply suspended temporarily. Await notice.）",
                       "Track utilization, impacts, and maintenance; improve fleet uptime.")
        add_if_present("HELI smart fleet management system FICS (Upgraded version（U.S. market supply suspended temporarily. Await notice.）",
                       "Expanded analytics and controls for large fleets.")
        add_if_present("Portal access fee of FICS (each truck per year)（U.S. market supply suspended temporarily. Await notice.）",
                       "Annual portal access for data and dashboards.")

    # Seats (comfort/ergonomics)
    if flags.get("ergonomics") or flags.get("long_runs"):
        add_if_present("Heli Suspension Seat with armrest", "Improves comfort and posture over long shifts.")
        add_if_present("Grammar Full Suspension Seat MSG65", "Premium suspension; pairs with fingertip controls.")

    # LPG specifics
    if flags.get("power_lpg"):
        add_if_present("Swing Out Drop LPG Bracket", "Faster, safer tank changes; reduces strain.")
        add_if_present("LPG Tank", "Extra tank for quick swaps on multi-shift operations.")
        add_if_present("Low Fuel Indicator Light", "Prevents surprise stalls; prompts timely tank change.")

    # Cold storage (electric only)
    if flags.get("electric") and flags.get("cold"):
        add_if_present("Added cost for the cold storage package (for electric forklift)",
                       "Seals and heaters for cold rooms; consistent performance in freezer aisles.")

    # Overhead guard height (clearance needs)
    if flags.get("tall_operator") or flags.get("high_loads"):
        add_if_present("more 100mm higher overhead guard OHG as 2370mm (93”)",
                       "Extra headroom and load clearance where needed.")

    # Special paint (brand/site requirement)
    if flags.get("special_color"):
        add_if_present("Special Color Painting", "Matches facility requirement or corporate branding.")

    # Lifting eyes (site rigging/transport)
    if flags.get("rigging"):
        add_if_present("Lifting eyes", "Simplifies hoisting the truck safely during site work or transport.")

    # Trim to budget
    return picks[:max_items]

def recommend_options_from_sheet(user_q: str, max_total: int = 6) -> dict:
    """
    Returns ONLY items that exist in /data/forklift_options_benefits.xlsx:
      {
        "tire": {"name","benefit"} | None,
        "attachments": [{"name","benefit"}, ...],
        "options": [{"name","benefit"}, ...]
      }
    """
    try:
        items = load_options()
    except Exception:
        items = []

    lut = _lut_by_name(items)

    # Start with your main flagger if present
    flags = _need_flags_from_text(user_q) if '_need_flags_from_text' in globals() else {}

    # Keep the raw text for attachment defaults that key off wording like “pallets”
    flags["_raw_text"] = user_q or ""

    # Lightweight power/env/text fallbacks so we don’t miss obvious cues
    t = (user_q or "").lower()
    flags.setdefault("power_lpg", bool(re.search(r'\b(lpg|propane|lp[-\s]?gas)\b', t)))
    flags.setdefault("electric", bool(re.search(r'\b(lithium|li[-\s]?ion|electric|battery)\b', t)))

    # Surface non-marking / environment cues (helps tire pick ALWAYS return something sensible)
    flags.setdefault("non_marking", bool(re.search(r'non[-\s]?mark|no\s*marks?|black\s*marks?|avoid\s*marks?|scuff', t)))
    flags.setdefault("rough", bool(re.search(r'rough|uneven|broken|pothole|curb|rail|speed\s*bumps?', t)))
    flags.setdefault("debris", bool(re.search(r'debris|nails|screws|scrap|glass|shavings|chips?', t)))
    flags.setdefault("gravel", "gravel" in t)
    flags.setdefault("yard", "yard" in t)
    flags.setdefault("outdoor", bool(re.search(r'\boutdoor|dock|lot|asphalt|gravel|dirt|parking\b', t)))
    flags.setdefault("indoor", bool(re.search(r'\bindoor|warehouse|inside|factory|floor\b', t)))
    flags.setdefault("mixed", ("indoor" in t and "outdoor" in t) or ("mixed" in t) or ("both" in t))

    # Attachment/option triggers that your main flagger might miss
    flags.setdefault("alignment_frequent", bool(re.search(r'align|line\s*up|tight\s*aisles|staging', t)))
    flags.setdefault("varied_width", bool(re.search(r'vary|mixed\s*pallet|different\s*width|multiple\s*widths|mix\s*of\s*\d+\s*["in]?\s*(?:and|&)\s*\d+\s*["in]?\s*pallets?', t)))
    flags.setdefault("paper_rolls", bool(re.search(r'paper\s*roll|newsprint|tissue', t)))
    flags.setdefault("slip_sheets", bool(re.search(r'slip[-\s]?sheet', t)))
    flags.setdefault("carpet", "carpet" in t or "textile" in t)
    flags.setdefault("long_loads", bool(re.search(r'long|oversize|over[-\s]?length|overhang|\b\d+\s*[- ]?ft\b|\b\d+\s*foot\b|\b\d+\s*feet\b|crate[s]?', t)))
    flags.setdefault("weighing", bool(re.search(r'weigh|scale|check\s*weight', t)))
    flags.setdefault("pedestrian_heavy", bool(re.search(r'pedestrian|foot\s*traffic|busy|congested|blind\s*corner|walkway', t)))
    flags.setdefault("poor_visibility", bool(re.search(r'low\s*light|dim|night|second\s*shift|poor\s*lighting', t)))
    flags.setdefault("extra_hydraulics", ("4th function" in t) or ("fourth function" in t))
    flags.setdefault("multi_function", ("multiple clamp" in t) or ("multiple attachments" in t))
    flags.setdefault("ergonomics", bool(re.search(r'ergonomic|fatigue|wrist|reach|comfort', t)))
    flags.setdefault("long_runs", bool(re.search(r'long\s*shifts?|multi[-\s]?shift|continuous', t)))
    flags.setdefault("cold", bool(re.search(r'cold|freezer|refrigerated|winter', t)))
    flags.setdefault("hot", bool(re.search(r'hot|heat|summer|foundry|high\s*ambient', t)))
    flags.setdefault("speed_control", bool(re.search(r'limit\s*speed|speeding|zoned\s*speed', t)))
    flags.setdefault("ops_required", bool(re.search(r'ops|operator\s*presence|osha|insurance|audit|policy', t)))
    flags.setdefault("tall_operator", ("tall operator" in t) or ("headroom" in t))
    flags.setdefault("high_loads", bool(re.search(r'high\s*mast|tall\s*stacks|top\s*heavy|elevated', t)))
    flags.setdefault("special_color", ("special color" in t) or ("paint" in t))
    flags.setdefault("rigging", ("rigging" in t) or ("lift with crane" in t))

    # Tire (ALWAYS returns something if you used the updated _pick_tire_from_flags)
    tire = _pick_tire_from_flags(flags, lut)

    # Attachments (relevant + default Sideshifter where sensible)
    attachments = _pick_attachments_from_excel(flags, lut, max_items=min(6, max_total))

    # Options (strict cue-gated)
    options = _pick_options_from_excel(flags, lut, max_items=min(6, max_total))

    return {"tire": tire, "attachments": attachments, "options": options}

def generate_catalog_mode_response(user_q: str, max_per_section: int = 6) -> str:
    """
    Clean, plain-text "catalog" answer (no markdown/italics/bold quotes).
    - If the user asks to "list all" options/attachments/tires, show the FULL catalog
      from the Excel (grouped) with one-line benefits.
    - Otherwise: concise, scenario-aware picks (Tires, Attachments, Options).
    """

    # ---------- text cleaners (ASCII, no fancy punctuation / markdown) ----------
    def _plain(s: str) -> str:
        if not isinstance(s, str):
            return s
        s = (s
             .replace("“", '"').replace("”", '"')
             .replace("‘", "'").replace("’", "'")
             .replace("—", "-").replace("–", "-")
             .replace("\u00a0", " "))
        s = s.replace("**", "").replace("__", "")
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r" ?\n ?", "\n", s)
        return s.strip()

    # ---------- detect "list all" intent ----------
    t = (user_q or "").lower()
    list_all = bool(re.search(r"\b(list|show|give|display)\b.*\ball\b", t)) or \
               bool(re.search(r"\b(all|every)\b.*\b(option|attachment|tire)s?\b", t)) or \
               bool(re.search(r"\b(full|complete)\b.*\b(list|catalog)\b", t))

    # ---------- load rows once ----------
    try:
        rows = load_options()
    except Exception:
        rows = []

    # quick lookup by normalized name
    lut = { _norm(r.get("name") or r.get("option") or ""): r for r in rows }

    def _row(nm: str) -> dict | None:
        return lut.get(_norm(nm))

    def _benefit(nm: str, default: str) -> str:
        r = _row(nm)
        ben = (r.get("benefit") if r else "") or default
        return ben.strip()

    def _line(name: str, benefit: str | None) -> str:
        name = _plain(name or "")
        ben  = _plain((benefit or "").strip())
        return f"- {name}" + (f" - {ben}" if ben else "")

    # ---------- FULL CATALOG OUTPUT ----------
    if list_all:
        tires, atts, opts = [], [], []
        for o in _options_iter():  # yields: {code,name,benefit,category,lname}
            nm = o["name"]
            ben = o.get("benefit", "")
            nl = o["lname"]
            if "tire" in nl:
                tires.append((nm, ben))
            elif _is_attachment(nl):
                atts.append((nm, ben))
            else:
                opts.append((nm, ben))

        tires.sort(key=lambda x: x[0].lower())
        atts.sort(key=lambda x: x[0].lower())
        opts.sort(key=lambda x: x[0].lower())

        out: list[str] = []
        out.append("Catalog (All Available)")
        out.append("")
        out.append("Tires:")
        if tires:
            out.extend([_line(n, b) for n, b in tires])
        else:
            out.append("- None found in catalog")
        out.append("")
        out.append("Attachments:")
        if atts:
            out.extend([_line(n, b) for n, b in atts])
        else:
            out.append("- None found in catalog")
        out.append("")
        out.append("Options:")
        if opts:
            out.extend([_line(n, b) for n, b in opts])
        else:
            out.append("- None found in catalog")

        return _plain("\n".join(out))

    # ---------- SCENARIO-AWARE COMPACT OUTPUT (no markdown) ----------
    flags = _need_flags_from_text(user_q) if '_need_flags_from_text' in globals() else {}

    rec = recommend_options_from_sheet(user_q, max_total=max_per_section)
    tire_pick = rec.get("tire")
    attachments = rec.get("attachments", [])[:max_per_section]
    options     = rec.get("options", [])[:max_per_section]

    lines: list[str] = []

    # Tires
    lines.append("Tires (recommended):")
    if tire_pick:
        best_for = ", ".join(_env_tags_for_name(tire_pick.get("name", ""))) or "General use"
        lines.append(_line(tire_pick.get("name",""), tire_pick.get("benefit","")))
        lines.append(f"  Best used for: {best_for}")
    else:
        lines.append("- Not specified")

    # Attachments
    lines.append("")
    lines.append("Attachments (relevant):")
    if attachments:
        for a in attachments:
            lines.append(_line(a.get("name",""), a.get("benefit","")))
    else:
        lines.append("- Not specified")

    # Options
    lines.append("")
    lines.append("Options (relevant):")
    if options:
        for o in options:
            lines.append(_line(o.get("name",""), o.get("benefit","")))
    else:
        lines.append("- Not specified")

    return _plain("\n".join(lines))

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

    # ---- Select tire + options from the Excel sheet (explicit keys)
    rec = recommend_options_from_sheet(user_q, max_total=6)
    chosen_tire = rec.get("tire")
    attachments = rec.get("attachments", [])
    non_attachments = rec.get("options", [])

    # Indoor sanity override (single copy only):
    # If environment is Indoor and the chosen tire is any "Dual ..." variant without
    # an explicit non-marking or stability cue, switch to Non-Marking Tires.
    _user_l = (user_q or "").lower()
    if env == "Indoor" and chosen_tire and "dual" in (chosen_tire.get("name", "").lower()):
        has_nonmark = bool(re.search(r'non[-\s]?mark', _user_l))
        has_stability = bool(re.search(r'(ramp|slope|incline|grade|wide load|long load|top heavy|high mast)', _user_l))
        if not has_nonmark and not has_stability:
            nm_row = options_lookup_by_name().get("non-marking tires")
            if nm_row:
                chosen_tire = {
                    "name": nm_row["name"],
                    "benefit": (nm_row.get("benefit") or "Non-marking compound prevents black marks on painted/epoxy floors.")
                }
            else:
                chosen_tire = {
                    "name": "Non-Marking Tires",
                    "benefit": "Non-marking compound prevents black marks on painted/epoxy floors."
                }

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
    if want.get("cap_lbs"):
        lines.append(f"- {int(round(want['cap_lbs'])):,} lb")
    else:
        lines.append("- Not specified")


    # Tire Type from Excel recommender
    lines.append("\nTire Type:")
    if chosen_tire:
        lines.append(f"- {chosen_tire['name']} — {chosen_tire.get('benefit','').strip() or ''}".rstrip(" —"))
    else:
        lines.append("- Not specified")

    # Attachments from Excel recommender (only if present in your sheet)
# Attachments from Excel recommender (only if present in your sheet)
    lines.append("\nAttachments:")
    if attachments:
        for a in attachments:
            benefit = (a.get("benefit","") or "").strip()
            lines.append(f"- {a['name']}" + (f" — {benefit}" if benefit else ""))
    else:
        lines.append("- Not specified")

    # Options (non-attachments) — cue-gated and pulled from your Excel only
    # Options (non-attachments) — pulled directly from Excel matches
    lines.append("\nOptions:")
    if non_attachments:
        for o in non_attachments:
            ben = (o.get("benefit","") or "").strip()
            lines.append(f"- {o['name']}" + (f" — {ben}" if ben else ""))
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
    lines.append("- I need better all-terrain capability. — Ask: What specific terrains do you operate on? | Reframe: This model excels in diverse conditions. | Proof: Proven performance in various environments. | Next: Shall we schedule a demo?")
    lines.append("- Are lithium batteries reliable? — Ask: What concerns do you have about battery performance? | Reframe: Lithium offers longer life and less maintenance. | Proof: Industry-leading warranty on batteries. | Next: Would you like to see the specs?")
    lines.append("- How does this compare to diesel? — Ask: What are your priorities, emissions or power? | Reframe: Lithium is cleaner and quieter. | Proof: Lower operational costs over time. | Next: Can I provide a cost analysis?")
    lines.append("- What about service and support? — Ask: What level of support do you expect? | Reframe: We offer comprehensive service plans. | Proof: Dedicated support team available. | Next: Shall we discuss service options?")
    lines.append("- Is it suitable for heavy-duty tasks? — Ask: What tasks will you be performing? | Reframe: Designed for robust applications. | Proof: Tested under heavy loads. | Next: Would you like to see a demonstration?")
    lines.append("- I'm concerned about the upfront cost. — Ask: What budget constraints are you working with? | Reframe: Consider total cost of ownership. | Proof: Lower energy and maintenance costs. | Next: Can I help with financing options?")

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
