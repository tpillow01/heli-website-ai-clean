"""
ai_logic.py — catalog loader + intent-driven answers for tires/attachments/options/telemetry
- Reads forklift_options_benefits.xlsx (env HELI_CATALOG_XLSX or ./data/forklift_options_benefits.xlsx)
- Graceful fallback to built-in defaults if Excel missing/empty
- Intent detection: shows ONLY the sections the user asked about
- Simple scenario heuristics (indoor epoxy/non-marking, cold weather, debris, soft ground)
- Back-compat shims for older routes: render_catalog_sections(), recommend_options_from_sheet()
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Imports & logging
# ─────────────────────────────────────────────────────────────────────────────
import os, re, json, logging
from functools import lru_cache
from typing import Any, Dict, List, Tuple, Optional

log = logging.getLogger("ai_logic")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

try:
    import pandas as pd  # optional
except Exception:  # pandas may not be present in some builds
    pd = None  # type: ignore

# ─────────────────────────────────────────────────────────────────────────────
# Paths & tiny helpers
# ─────────────────────────────────────────────────────────────────────────────
CATALOG_XLSX = os.environ.get(
    "HELI_CATALOG_XLSX",
    os.path.join(os.path.dirname(__file__), "data", "forklift_options_benefits.xlsx"),
)

def _norm(s: Any) -> str:
    s = ("" if s is None else str(s)).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _lo(s: Any) -> str:
    return _norm(s).lower()

# ─────────────────────────────────────────────────────────────────────────────
# Fallback data — used if Excel missing/empty
# ─────────────────────────────────────────────────────────────────────────────
FALLBACK_ROWS: List[Dict[str, str]] = [
    # Tires
    {"name":"Solid Tires","benefit":"Puncture-proof, low-maintenance tires for debris-prone floors.","type":"tire","subcategory":"Tire"},
    {"name":"Dual Tires","benefit":"Wider footprint for better stability and traction on soft ground.","type":"tire","subcategory":"Tire"},
    {"name":"Dual Solid Tires","benefit":"Combines puncture resistance with extra stability and load support.","type":"tire","subcategory":"Tire"},
    {"name":"Non-Marking Tires","benefit":"Protect indoor floors by avoiding black scuff marks.","type":"tire","subcategory":"Tire"},
    {"name":"Non-Marking Dual Tires","benefit":"Floor-safe traction with a wider, more stable footprint.","type":"tire","subcategory":"Tire"},

    # Options (comfort / weather / safety / telemetry)
    {"name":"Panel mounted Cab","benefit":"Enclosed cab panels for weather protection and operator comfort.","type":"option","subcategory":"Weather/Cab"},
    {"name":"Heater","benefit":"Keeps the cab warm for safer, more comfortable cold-weather operation.","type":"option","subcategory":"Weather/Cab"},
    {"name":"Air conditioner","benefit":"Cools the cab to reduce heat stress and maintain productivity.","type":"option","subcategory":"Weather/Cab"},
    {"name":"Glass Windshield with Wiper","benefit":"Clear forward visibility in rain and dust.","type":"option","subcategory":"Weather/Cab"},
    {"name":"Top Rain-proof Glass","benefit":"Overhead visibility while shielding the operator from rain.","type":"option","subcategory":"Weather/Cab"},
    {"name":"Rear Windshield Glass","benefit":"Improves rear visibility and shields from wind and debris.","type":"option","subcategory":"Weather/Cab"},
    {"name":"LED Rear Working Light","benefit":"Bright, efficient rear lighting with long service life.","type":"option","subcategory":"Safety Lighting"},
    {"name":"LED Rotating Light","benefit":"High-visibility 360° beacon to alert pedestrians.","type":"option","subcategory":"Safety Lighting"},
    {"name":"Blue Light","benefit":"Pedestrian warning light to increase visibility in busy aisles.","type":"option","subcategory":"Safety Lighting"},
    {"name":"Heli Suspension Seat with armrest","benefit":"Reduces vibration for better comfort and control.","type":"option","subcategory":"Ergonomics"},
    {"name":"Grammar Full Suspension Seat MSG65","benefit":"Premium suspension and adjustability for long-shift comfort.","type":"option","subcategory":"Ergonomics"},

    # Filtration/protection
    {"name":"Radiator protection bar","benefit":"Guards the radiator core from impacts.","type":"option","subcategory":"Protection"},
    {"name":"Steel Belly Pan","benefit":"Shields undercarriage components from debris and impacts.","type":"option","subcategory":"Protection"},
    {"name":"Removable radiator screen","benefit":"Easy cleaning to keep cooling performance high.","type":"option","subcategory":"Filtration/Cooling"},
    {"name":"Dual Air Filter","benefit":"Enhanced engine air filtration for dusty environments.","type":"option","subcategory":"Filtration/Cooling"},
    {"name":"Pre air cleaner","benefit":"Cyclonic pre-cleaning extends main air filter life.","type":"option","subcategory":"Filtration/Cooling"},
    {"name":"Air filter service indicator","benefit":"Tells you exactly when to change the filter, avoiding guesswork.","type":"option","subcategory":"Filtration/Cooling"},

    # Hydraulics / controls
    {"name":"3 Valve with Handle","benefit":"Adds a third hydraulic function to run basic attachments.","type":"option","subcategory":"Hydraulic Control"},
    {"name":"4 Valve with Handle","benefit":"Enables two auxiliary functions for multi-function attachments.","type":"option","subcategory":"Hydraulic Control"},
    {"name":"5 Valve with Handle","benefit":"Maximum hydraulic flexibility for specialized attachments.","type":"option","subcategory":"Hydraulic Control"},

    # Telemetry (FICS)
    {"name":"HELI smart fleet management system FICS (Standard)","benefit":"Telematics for usage tracking, alerts, and basic analytics.","type":"option","subcategory":"Telemetry"},
    {"name":"HELI smart fleet management system FICS (Upgraded)","benefit":"Adds advanced reporting, diagnostics, and fleet insights.","type":"option","subcategory":"Telemetry"},
    {"name":"Portal access fee of FICS (per truck per year)","benefit":"Enables cloud portal access for data, reports, and alerts.","type":"option","subcategory":"Telemetry"},

    # A few attachments (common)
    {"name":"Sideshifter","benefit":"Aligns loads without repositioning — faster, cleaner placement.","type":"attachment","subcategory":"Fork Handling"},
    {"name":"Fork Positioner","benefit":"Adjust fork spread from the seat for mixed pallet sizes.","type":"attachment","subcategory":"Fork Handling"},
    {"name":"Paper Roll Clamp","benefit":"Secure, damage-reducing handling for paper rolls.","type":"attachment","subcategory":"Clamps"},
    {"name":"Push/ Pull (Slip-Sheet)","benefit":"Handles slip-sheeted cartons — eliminates pallets.","type":"attachment","subcategory":"Special Handling"},
    {"name":"Carpet Pole","benefit":"Handles coils, carpet, and tubing via a single pole/ram.","type":"attachment","subcategory":"Poles/Booms"},
    {"name":"Fork Extensions","benefit":"Supports longer or over-length loads safely.","type":"attachment","subcategory":"Fork Handling"},
]

# ─────────────────────────────────────────────────────────────────────────────
# Read Excel safely
# ─────────────────────────────────────────────────────────────────────────────
def _read_catalog_df() -> Optional[List[Dict[str,str]]]:
    exists = os.path.exists(CATALOG_XLSX)
    log.info("[ai_logic] Using catalog: %s (exists=%s)", CATALOG_XLSX, exists)
    if not pd or not exists:
        return None
    try:
        df = pd.read_excel(CATALOG_XLSX, engine="openpyxl")
    except Exception:
        # fallback engine if openpyxl not present
        df = pd.read_excel(CATALOG_XLSX)
    if df is None or df.empty:
        return None

    # tolerant column mapping
    cols = { _lo(c): c for c in df.columns }
    name_col    = cols.get("name") or cols.get("option")
    benefit_col = cols.get("benefit") or cols.get("description") or cols.get("desc")
    type_col    = cols.get("type") or cols.get("category")
    subcat_col  = cols.get("subcategory")

    if not name_col:
        return None

    out: List[Dict[str,str]] = []
    for _, row in df.iterrows():
        nm  = _norm(row.get(name_col))
        ben = _norm(row.get(benefit_col)) if benefit_col else ""
        typ = _lo(row.get(type_col)) if type_col else ""
        sub = _norm(row.get(subcat_col)) if subcat_col else ""

        if not nm:
            continue

        # normalize type
        if typ in ("attachments","attachment","att"):
            typ = "attachment"
        elif typ in ("tires","tire"):
            typ = "tire"
        elif typ in ("options","option","opt",""):
            # If subcategory says Tire, treat as tire
            if _lo(sub) == "tire":
                typ = "tire"
            else:
                typ = "option"

        out.append({"name": nm, "benefit": ben, "type": typ, "subcategory": sub or ""})
    return out or None

@lru_cache(maxsize=1)
def _catalog_rows() -> List[Dict[str,str]]:
    rows = _read_catalog_df()
    if rows:
        return rows
    # fallback ensures the app answers even without Excel
    return FALLBACK_ROWS

# ─────────────────────────────────────────────────────────────────────────────
# Buckets and simple lookups
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def load_catalogs() -> Tuple[Dict[str,str], Dict[str,str], Dict[str,str]]:
    rows = _catalog_rows()
    tires: Dict[str,str] = {}
    options: Dict[str,str] = {}
    attachments: Dict[str,str] = {}

    for r in rows:
        t = _lo(r.get("type"))
        nm = _norm(r.get("name"))
        ben = _norm(r.get("benefit"))
        if not nm:
            continue
        if t == "tire":
            tires[nm] = ben
        elif t == "attachment":
            attachments[nm] = ben
        else:
            options[nm] = ben

    log.info("[ai_logic] Loaded buckets: tires=%d attachments=%d options=%d",
             len(tires), len(attachments), len(options))
    return options, attachments, tires

def refresh_catalog_caches():
    _catalog_rows.cache_clear()
    load_catalogs.cache_clear()

# ─────────────────────────────────────────────────────────────────────────────
# Intent detection
# ─────────────────────────────────────────────────────────────────────────────
def _intent(user_q: str) -> Dict[str, bool]:
    q = _lo(user_q)
    wants_tires       = bool(re.search(r"\btires?\b|\btyres?\b|tire types?|non[-\s]?mark", q))
    wants_attachments = bool(re.search(r"\battachments?\b", q))
    wants_options     = bool(re.search(r"\boptions?\b", q))
    wants_telemetry   = bool(re.search(r"\btelemetry\b|\bfics\b|\bfleet\s+management\b", q))
    list_all          = bool(re.search(r"\b(all|list|types|full)\b", q))
    # scenarios
    indoor_nm         = bool(re.search(r"\bindoor\b|epoxy|polished|non[-\s]?mark|no\s+scuff|no\s+marks", q))
    cold              = bool(re.search(r"\bcold\b|freezer|cold\s+storage|subzero", q))
    debris            = bool(re.search(r"\bdebris|nails|scrap|puncture|flats?\b", q))
    soft_ground       = bool(re.search(r"\bsoft\s+ground|gravel|dirt|yard|uneven\b", q))
    dark_aisles       = bool(re.search(r"\bdark\b|low\s+light|poor\s+visibility|night", q))

    return dict(
        tires=wants_tires,
        attachments=wants_attachments,
        options=wants_options or wants_telemetry,  # telemetry is a subset of options bucket
        telemetry=wants_telemetry,
        list_all=list_all,
        indoor_nm=indoor_nm,
        cold=cold,
        debris=debris,
        soft=soft_ground,
        dark=dark_aisles,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Selection logic (simple & deterministic)
# ─────────────────────────────────────────────────────────────────────────────
def _pick_tires(i: Dict[str,bool], tires: Dict[str,str]) -> List[Tuple[str,str]]:
    # if “types/all/list” → show all tire types available
    if i["list_all"]:
        return [(n, tires[n]) for n in tires.keys()]

    # scenario-based
    names = list(tires.keys())
    names_l = [n.lower() for n in names]

    def have(name_part: str) -> Optional[str]:
        for n in names:
            if name_part.lower() in n.lower():
                return n
        return None

    picks: List[str] = []
    if i["indoor_nm"]:
        # prefer Non-Marking (Cushion/Pneumatic wording varies by sheets; match loosely)
        for key in ("non-marking dual", "non-marking"):
            h = have(key)
            if h and h not in picks: picks.append(h)
    if i["debris"]:
        for key in ("dual solid", "solid"):
            h = have(key)
            if h and h not in picks: picks.append(h)
    if i["soft"]:
        for key in ("dual tires", "dual"):
            h = have("dual")
            if h and h not in picks: picks.append(h)
    # default if nothing matched
    if not picks:
        # prefer a single “representative” tire rather than always “Dual”
        for prefer in ("Solid Tires", "Non-Marking Tires", "Dual Tires"):
            h = have(prefer)
            if h:
                picks.append(h); break

    return [(n, tires.get(n, "")) for n in picks]

def _pick_attachments(i: Dict[str,bool], attachments: Dict[str,str]) -> List[Tuple[str,str]]:
    if i["list_all"]:
        return [(n, attachments[n]) for n in attachments.keys()]

    order = []
    if i["cold"]:
        order += ["Sideshifter", "Fork Positioner"]  # common in cold aisles (alignment from seat)
    if i["debris"] or i["soft"]:
        order += ["Fork Extensions"]  # generic helper

    # general standbys if nothing triggered
    if not order:
        order = ["Sideshifter", "Fork Positioner", "Fork Extensions", "Carpet Pole", "Paper Roll Clamp", "Push/ Pull (Slip-Sheet)"]

    out: List[Tuple[str,str]] = []
    seen = set()
    for want in order:
        # fuzzy contain
        for n, b in attachments.items():
            if _lo(want) in _lo(n) and n not in seen:
                out.append((n, b)); seen.add(n)
    # ensure something
    if not out:
        for n, b in list(attachments.items())[:5]:
            out.append((n, b))
    return out

def _pick_options(i: Dict[str,bool], options: Dict[str,str]) -> List[Tuple[str,str]]:
    if i["telemetry"]:
        # ONLY telemetry subset
        tele = [(n, b) for n, b in options.items() if "fics" in _lo(n) or "telemetry" in _lo(n) or "fleet" in _lo(n)]
        if tele:
            return tele
        # fallback to a named set if Excel lacks explicit telemetry rows
        fall = [
            ("HELI smart fleet management system FICS (Standard)", options.get("heli smart fleet management system fics (standard)", "Telematics for usage tracking, alerts, and basic analytics.")),
            ("HELI smart fleet management system FICS (Upgraded)", options.get("heli smart fleet management system fics (upgraded)", "Adds advanced reporting, diagnostics, and fleet insights.")),
            ("Portal access fee of FICS (per truck per year)", options.get("portal access fee of fics (per truck per year)", "Enables cloud portal access for data, reports, and alerts.")),
        ]
        # dedupe to present only those that exist or fallback text
        seen = set()
        out: List[Tuple[str,str]] = []
        for n, b in fall:
            if n not in seen:
                out.append((n, b)); seen.add(n)
        return out

    if i["list_all"]:
        return [(n, options[n]) for n in options.keys()]

    order: List[str] = []
    if i["cold"]:
        order += [
            "Panel mounted Cab", "Heater", "Glass Windshield with Wiper",
            "Top Rain-proof Glass", "Rear Windshield Glass",
            "LED Rear Working Light", "LED Rotating Light", "Blue Light"
        ]
    if i["dark"]:
        order += ["LED Rear Working Light", "LED Rotating Light", "Blue Light"]
    if i["debris"]:
        order += ["Radiator protection bar", "Steel Belly Pan", "Removable radiator screen", "Pre air cleaner", "Dual Air Filter", "Air filter service indicator"]

    # comfort catch-alls
    order += ["Heli Suspension Seat with armrest", "Grammar Full Suspension Seat MSG65"]

    out: List[Tuple[str,str]] = []
    seen = set()
    for want in order:
        for n, b in options.items():
            if _lo(want) in _lo(n) and n not in seen:
                out.append((n, b)); seen.add(n)

    if not out:
        for n, b in list(options.items())[:8]:
            out.append((n, b))
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Public main: recommend_from_catalog + renderer that hides unused sections
# ─────────────────────────────────────────────────────────────────────────────
def recommend_from_catalog(user_q: str, max_items: int = 6) -> Dict[str, List[Tuple[str,str]]]:
    opts, atts, tires = load_catalogs()
    i = _intent(user_q)

    result: Dict[str, List[Tuple[str,str]]] = {}

    # If user asked for a specific thing, only compute that bucket
    asked_specific = i["tires"] or i["attachments"] or i["options"] or i["telemetry"]
    if i["tires"] or (not asked_specific):
        t = _pick_tires(i, tires)[:max_items]
        if t:
            result["tires"] = t

    if i["attachments"] or (not asked_specific):
        a = _pick_attachments(i, atts)[:max_items]
        if a:
            result["attachments"] = a

    if i["options"] or i["telemetry"] or (not asked_specific):
        o = _pick_options(i, opts)[:max_items]
        if o:
            # if telemetry was explicitly asked, restrict to telemetry-only above; already handled
            result["options"] = o

    return result

def generate_catalog_mode_response(user_q: str, max_per_section: int = 6) -> str:
    picks = recommend_from_catalog(user_q, max_items=max_per_section)
    i = _intent(user_q)

    sections_order: List[Tuple[str,str]] = []
    # Only include sections the user asked for; if no specific ask, include the non-empty ones in standard order
    if i["tires"] or i["telemetry"] or i["attachments"] or i["options"]:
        if i["tires"] and "tires" in picks:         sections_order.append(("tires","Tires"))
        if i["attachments"] and "attachments" in picks: sections_order.append(("attachments","Attachments"))
        if (i["telemetry"] or i["options"]) and "options" in picks:
            title = "Telemetry" if i["telemetry"] else "Options"
            sections_order.append(("options", title))
    else:
        for key, title in (("tires","Tires"), ("attachments","Attachments"), ("options","Options")):
            if key in picks and picks[key]:
                sections_order.append((key, title))

    if not sections_order:
        return "No relevant items found."

    lines: List[str] = []
    for key, title in sections_order:
        lines.append(f"**{title}:**")
        for n, b in picks.get(key, []):
            lines.append(f"- {n}" + (f" — {b}" if b else ""))
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# JSON loaders for other parts of the app (light / optional)
# ─────────────────────────────────────────────────────────────────────────────
def _load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

accounts_raw = _load_json("accounts.json") or []
models_raw   = _load_json("models.json") or []
log.info("[ai_logic] Loaded accounts=%d | models=%d", len(accounts_raw if isinstance(accounts_raw,list) else []), len(models_raw if isinstance(models_raw,list) else []))

# Minimal stubs to satisfy imports elsewhere; safe no-ops.
def model_meta_for(row: Dict[str, Any]) -> Tuple[str, str, str]:
    code = _norm(row.get("Model") or row.get("model") or row.get("Code") or row.get("name") or "N/A")
    cls = _norm(row.get("Class") or row.get("class") or "")
    pwr = _norm(row.get("Power") or row.get("power") or "")
    return code, cls, pwr

def top_pick_meta(user_q: str) -> Optional[Tuple[str,str,str]]:
    # not ranking models here; return None so callers can handle gracefully
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Back-compat shims for older blueprints/routes
# ─────────────────────────────────────────────────────────────────────────────
def render_catalog_sections(user_q: str, max_per_section: int = 6) -> str:
    """Legacy import expected by options_attachments_router."""
    return generate_catalog_mode_response(user_q, max_per_section)

def recommend_options_from_sheet(user_q: str, max_per_section: int = 6):
    """Legacy import expected by some routes. Returns dict with tires/attachments/options."""
    return recommend_from_catalog(user_q, max_items=max_per_section)

# ─────────────────────────────────────────────────────────────────────────────
# Exports
# ─────────────────────────────────────────────────────────────────────────────
__all__ = [
    # Catalog IO
    "load_catalogs", "refresh_catalog_caches",

    # Recommend & render
    "recommend_from_catalog", "generate_catalog_mode_response",

    # Back-compat (legacy names)
    "render_catalog_sections", "recommend_options_from_sheet",

    # Minimal helpers used elsewhere
    "model_meta_for", "top_pick_meta",
]
