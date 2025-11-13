"""
ai_logic.py
Catalog helpers (tires / attachments / options / telemetry), intent -> picks,
and model recommendation utilities used across the Heli AI app.

This file is defensive:
- Works with or without pandas.
- Handles missing Excel by falling back to a small, built-in seed catalog.
- Exposes every helper your app/blueprints have tried importing.
- Never echoes the user's raw question into the results.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Imports / logging
# ─────────────────────────────────────────────────────────────────────────────
import os, json, re, logging, hashlib
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("ai_logic")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

try:
    import pandas as _pd  # optional
except Exception:
    _pd = None

# ─────────────────────────────────────────────────────────────────────────────
# Paths / constants
# ─────────────────────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(__file__)
_OPTIONS_XLSX = os.environ.get(
    "HELI_CATALOG_XLSX",
    os.path.join(_BASE_DIR, "data", "forklift_options_benefits.xlsx")
)

# Optional: if you keep models/accounts JSON beside the app, these names help.
_MODELS_JSON = os.environ.get("HELI_MODELS_JSON", os.path.join(_BASE_DIR, "models.json"))
_ACCOUNTS_JSON = os.environ.get("HELI_ACCOUNTS_JSON", os.path.join(_BASE_DIR, "accounts.json"))

# Intent patterns
_TIRES_PAT    = re.compile(r"\b(tires?|tyres?|tire\s*types?)\b", re.I)
_ATTACH_PAT   = re.compile(r"\b(attach(ment)?s?)\b", re.I)
_OPTIONS_PAT  = re.compile(r"\b(option|options)\b", re.I)
_TELEM_PAT    = re.compile(r"\b(fics|fleet\s*management|telemetry|portal)\b", re.I)

# Attach "hint" pattern (used for cold-only suppression)
_ATTACH_HINT  = re.compile(
    r"\b(attach(ment)?|clamp|sideshift|positioner|fork|boom|pole|ram|"
    r"push\s*/?\s*pull|slip[-\s]?sheet|paper\s*roll)\b", re.I
)

# ─────────────────────────────────────────────────────────────────────────────
# Tiny utils
# ─────────────────────────────────────────────────────────────────────────────
def _lower(s: Any) -> str:
    return str(s or "").lower()

def _norm(s: Any) -> str:
    return " ".join(str(s or "").split())

def _as_list(x) -> list:
    if isinstance(x, list):
        return x
    return [x] if x else []

# ─────────────────────────────────────────────────────────────────────────────
# Built-in fallback catalog (used if Excel missing or pandas unavailable)
# ─────────────────────────────────────────────────────────────────────────────
_FALLBACK_ROWS = [
    # Tires
    {"Option": "Solid Tires",               "Benefit": "Puncture-proof, low-maintenance tires for debris-prone floors.",                            "Type": "Tires",   "Subcategory": "Tire"},
    {"Option": "Dual Tires",                "Benefit": "Wider footprint for better stability and traction on soft ground.",                          "Type": "Tires",   "Subcategory": "Tire"},
    {"Option": "Dual Solid Tires",          "Benefit": "Combines puncture resistance with extra stability and load support.",                        "Type": "Tires",   "Subcategory": "Tire"},
    {"Option": "Non-Marking Tires",         "Benefit": "Protect indoor floors by avoiding black scuff marks.",                                        "Type": "Tires",   "Subcategory": "Tire"},
    {"Option": "Non-Marking Dual Tires",    "Benefit": "Floor-safe traction with a wider, more stable footprint.",                                   "Type": "Tires",   "Subcategory": "Tire"},
    # Options (subset)
    {"Option": "Panel mounted Cab",         "Benefit": "Enclosed cab panels for weather protection and operator comfort.",                           "Type": "Options", "Subcategory": "Operator Comfort"},
    {"Option": "Heater",                    "Benefit": "Keeps the cab warm for safer, more comfortable cold-weather operation.",                      "Type": "Options", "Subcategory": "Climate"},
    {"Option": "Air conditioner",           "Benefit": "Cools the cab to reduce heat stress and maintain productivity.",                              "Type": "Options", "Subcategory": "Climate"},
    {"Option": "Glass Windshield with Wiper","Benefit":"Clear forward visibility in rain and dust.",                                                 "Type": "Options", "Subcategory": "Operator Comfort"},
    {"Option": "Top Rain-proof Glass",      "Benefit": "Overhead visibility while shielding the operator from rain.",                                 "Type": "Options", "Subcategory": "General"},
    {"Option": "Rear Working Light",        "Benefit": "Illuminates the work area behind the truck for safer reversing.",                             "Type": "Options", "Subcategory": "Visibility"},
    {"Option": "LED Rear Working Light",    "Benefit": "Bright, efficient rear lighting with long service life.",                                     "Type": "Options", "Subcategory": "Visibility"},
    {"Option": "Blue Light",                "Benefit": "Pedestrian warning light to increase visibility in busy aisles.",                             "Type": "Options", "Subcategory": "Visibility"},
    {"Option": "Red side line Light",       "Benefit": "Creates visible ‘no-go’ safety zones along the truck’s sides.",                               "Type": "Options", "Subcategory": "Visibility"},
    {"Option": "Dual Air Filter",           "Benefit": "Enhanced engine air filtration for dusty environments.",                                      "Type": "Options", "Subcategory": "Filtration/ Cooling"},
    {"Option": "Pre air cleaner",           "Benefit": "Cyclonic pre-cleaning extends main air filter life.",                                         "Type": "Options", "Subcategory": "General"},
    {"Option": "Radiator protection bar",   "Benefit": "Guards the radiator core from impacts.",                                                      "Type": "Options", "Subcategory": "Filtration/ Cooling"},
    {"Option": "Air filter service indicator","Benefit":"Tells you exactly when to change the filter, avoiding guesswork.",                           "Type": "Options", "Subcategory": "Filtration/ Cooling"},
    {"Option": "Added cost for the cold storage package (for electric forklift)", "Benefit": "Freezer-rated components to reduce condensation/failures.", "Type": "Options", "Subcategory": "Climate"},
    {"Option": "HELI smart fleet management system FICS (Standard version（U.S. market supply suspended temporarily. Await notice.）",
     "Benefit": "Telematics for usage tracking, alerts, and basic analytics.", "Type": "Options", "Subcategory": "Telemetry"},
    {"Option": "HELI smart fleet management system FICS (Upgraded version（U.S. market supply suspended temporarily. Await notice.）",
     "Benefit": "Adds advanced reporting, diagnostics, and fleet insights.", "Type": "Options", "Subcategory": "Telemetry"},
    {"Option": "Portal access fee of FICS (each truck per year)（U.S. market supply suspended temporarily. Await notice.）",
     "Benefit": "Enables cloud portal access for data, reports, and alerts.", "Type": "Options", "Subcategory": "Telemetry"},
    # Attachments (subset)
    {"Option": "Sideshifter",               "Benefit": "Aligns loads without moving the truck; faster, cleaner placement.",                           "Type": "Attachments", "Subcategory": "Hydraulic Assist"},
    {"Option": "Fork Positioner",           "Benefit": "Adjusts fork spread from the seat for mixed pallet sizes.",                                   "Type": "Attachments", "Subcategory": "Fork Handling"},
    {"Option": "Bale Clamp",                "Benefit": "Grips bales (paper, cotton, recycling) without pallets.",                                     "Type": "Attachments", "Subcategory": "Clamp"},
    {"Option": "Carton Clamp",              "Benefit": "Pads squeeze large boxes/appliances to handle pallet-free.",                                  "Type": "Attachments", "Subcategory": "Clamp"},
    {"Option": "Drum Clamp",                "Benefit": "Lifts/tilts one or multiple 55-gal drums safely.",                                            "Type": "Attachments", "Subcategory": "Clamp"},
    {"Option": "Fork Extensions",           "Benefit": "Temporarily lengthen forks for occasional oversized loads.",                                   "Type": "Attachments", "Subcategory": "Fork Handling"},
    {"Option": "Rotator",                   "Benefit": "Spins the carriage to dump bins or reorient loads.",                                          "Type": "Attachments", "Subcategory": "Rotation"},
    {"Option": "Bag Pusher",                "Benefit": "Push plate drives big bags/sacks deep into a trailer or container.",                           "Type": "Attachments", "Subcategory": "Push/ Pull"},
]

# ─────────────────────────────────────────────────────────────────────────────
# Catalog loading
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def load_catalog_rows() -> List[Dict[str, str]]:
    """Return catalog rows from Excel, or fallback seed if Excel not available."""
    if _pd is not None and os.path.exists(_OPTIONS_XLSX):
        try:
            df = _pd.read_excel(_OPTIONS_XLSX)
            df = df.fillna("")
            rows = []
            for _, r in df.iterrows():
                rows.append({
                    "Option": str(r.get("Option", "")).strip(),
                    "Benefit": str(r.get("Benefit", "")).strip(),
                    "Type": str(r.get("Type", "")).strip(),
                    "Subcategory": str(r.get("Subcategory", "")).strip()
                })
            log.info("[ai_logic] Using catalog: %s (exists=True)", _OPTIONS_XLSX)
            return rows
        except Exception as e:
            log.warning("[ai_logic] Failed reading %s: %s — falling back.", _OPTIONS_XLSX, e)
    else:
        log.info("[ai_logic] Using fallback catalog (pandas/Excel missing)")
    return _FALLBACK_ROWS[:]

@lru_cache(maxsize=1)
def load_catalogs() -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Returns three dicts: (options, attachments, tires) mapping name -> benefit.
    """
    rows = load_catalog_rows()
    options: Dict[str, str] = {}
    attachments: Dict[str, str] = {}
    tires: Dict[str, str] = {}
    for r in rows:
        name = _norm(r.get("Option"))
        ben  = _norm(r.get("Benefit"))
        typ  = _norm(r.get("Type"))
        if not name:
            continue
        if typ.lower() == "options":
            options[name] = ben
        elif typ.lower() == "attachments":
            attachments[name] = ben
        elif typ.lower() == "tires":
            tires[name] = ben
    log.info("[ai_logic] Loaded buckets: tires=%d attachments=%d options=%d", len(tires), len(attachments), len(options))
    return options, attachments, tires

def load_options() -> Dict[str, str]:
    return load_catalogs()[0]

def load_attachments() -> Dict[str, str]:
    return load_catalogs()[1]

def load_tires_as_options() -> Dict[str, str]:
    return load_catalogs()[2]

def load_tires() -> Dict[str, str]:
    return load_catalogs()[2]

def refresh_catalog_caches() -> None:
    """Clear cached catalog readers (used by blueprints calling 'refresh_catalog_caches')."""
    load_catalog_rows.cache_clear()
    load_catalogs.cache_clear()
    log.info("[ai_logic] Catalog caches cleared")

# Lookups
def options_lookup_by_name(name: str) -> Optional[Dict[str, str]]:
    """Find an item by name across all buckets."""
    n = _norm(name)
    opts, atts, tires = load_catalogs()
    for bucket_name, bucket in (("options", opts), ("attachments", atts), ("tires", tires)):
        if n in bucket:
            return {"bucket": bucket_name, "name": n, "benefit": bucket[n]}
    return None

def option_benefit(name: str) -> str:
    found = options_lookup_by_name(name)
    return found["benefit"] if found else ""

# ─────────────────────────────────────────────────────────────────────────────
# Intent helpers
# ─────────────────────────────────────────────────────────────────────────────
def _wants_sections(q: str) -> Dict[str, bool]:
    t = q or ""
    return {
        "tires": bool(_TIRES_PAT.search(t)),
        "attachments": bool(_ATTACH_PAT.search(t)),
        "options": bool(_OPTIONS_PAT.search(t)),
        "telemetry": bool(_TELEM_PAT.search(t)),
        "any": bool(_TIRES_PAT.search(t) or _ATTACH_PAT.search(t) or _OPTIONS_PAT.search(t) or _TELEM_PAT.search(t)),
    }

def _env_flags(q: str) -> Dict[str, bool]:
    ql = _lower(q)
    return {
        "cold": any(k in ql for k in ("cold", "freezer", "subzero", "winter")),
        "indoor": any(k in ql for k in ("indoor", "warehouse", "inside", "epoxy", "polished", "concrete")),
        "outdoor": any(k in ql for k in ("outdoor", "yard", "rain", "snow", "dust", "gravel", "dirt")),
        "dark": any(k in ql for k in ("dark", "dim", "night", "poor lighting", "low light")),
        "mentions_clamp": bool(re.search(r"\bclamp|paper\s*roll|bale|drum|carton|block\b", ql)),
        "mentions_align": bool(re.search(r"\b(side\s*shift|sideshifter|align|positioner|narrow|tight\s*aisle)\b", ql)),
        "asks_non_mark": bool(re.search(r"non[-\s]?mark", ql)),
        "mentions_attach": bool(_ATTACH_HINT.search(ql)),
    }

def parse_catalog_intent(user_q: str) -> Dict[str, Any]:
    t = _lower(user_q)
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
# Ranking helpers
# ─────────────────────────────────────────────────────────────────────────────
def _kw_score(q: str, name: str, benefit: str) -> float:
    """Lightweight scorer with scenario bumps (cold/dark/indoor/yard/telemetry)."""
    ql = _lower(q)
    text = _lower(name + " " + (benefit or ""))
    score = 0.01

    for w in ("indoor","warehouse","outdoor","yard","dust","debris","visibility",
              "lighting","safety","cold","freezer","rain","snow","cab","comfort",
              "vibration","filtration","radiator","screen","pre air cleaner",
              "dual air filter","heater","wiper","windshield","work light","led",
              "non-mark","pneumatic","cushion"):
        if w in ql and w in text:
            score += 0.7

    # cold: cab/heater/wiper/glass/lights up; AC down
    if any(k in ql for k in ("cold","freezer","subzero","winter")):
        if any(k in text for k in ("cab","heater","defrost","wiper","rain-proof","glass","windshield","work light","led")):
            score += 2.0
        if "air conditioner" in text or "a/c" in text:
            score -= 2.0

    # dark: lights up
    if any(k in ql for k in ("dark","dim","night","poor lighting","low light")):
        if any(k in text for k in ("light","led","beacon","blue light","work light")):
            score += 1.6

    # dusty/yard protection
    if any(k in ql for k in ("debris","yard","gravel","dirty","recycling","foundry","sawmill")):
        if any(k in text for k in ("radiator","screen","pre air cleaner","dual air filter","filtration","belly pan","protection")):
            score += 1.3

    # Telemetry alignment
    if _TELEM_PAT.search(ql) and _TELEM_PAT.search(text):
        score += 2.2

    return score

def _rank_bucket(q: str, bucket: Dict[str, str], limit: int = 6) -> List[Dict[str, str]]:
    if not bucket:
        return []
    scored = []
    for name, benefit in bucket.items():
        s = _kw_score(q, name, benefit)
        scored.append((s, {"name": name, "benefit": benefit}))
    scored.sort(key=lambda t: t[0], reverse=True)
    out = [row for _, row in scored if _norm(row["name"])]
    return out[:limit] if isinstance(limit, int) and limit > 0 else out

# ─────────────────────────────────────────────────────────────────────────────
# Safety net: never echo user's question; drop malformed rows
# ─────────────────────────────────────────────────────────────────────────────
def sanitize_catalog_result(user_q: str, result: dict | None) -> dict:
    user_norm = _norm(user_q).lower()

    if not isinstance(result, dict):
        result = {}

    def _clean_list(lst):
        out = []
        for it in _as_list(lst):
            if not isinstance(it, dict):
                continue
            name = _norm(it.get("name"))
            if not name:
                continue
            # Never echo question text
            if name.lower() == user_norm:
                continue
            ben = _norm(it.get("benefit", ""))
            out.append({"name": name, "benefit": ben})
        return out

    cleaned = {
        "tires": _clean_list(result.get("tires")),
        "attachments": _clean_list(result.get("attachments")),
        "options": _clean_list(result.get("options")),
        "telemetry": _clean_list(result.get("telemetry")),
    }
    # Drop empties entirely
    return {k: v for k, v in cleaned.items() if v}

# ─────────────────────────────────────────────────────────────────────────────
# Primary selector (Catalog mode)
# ─────────────────────────────────────────────────────────────────────────────
def recommend_options_from_sheet(user_q: str, limit: int = 6) -> dict:
    """
    Scenario-aware selector. Shows ONLY the sections the user requested
    (unless nothing requested, then returns a sensible default mix).
    Now includes hard fallbacks so 'tires' never returns empty when asked.
    """
    wants = _wants_sections(user_q)
    env   = _env_flags(user_q)

    # Load primary buckets
    options, attachments, tires = load_catalogs()

    # Defensive: if a requested bucket is empty, build it from the in-file fallback rows
    if wants["tires"] and not tires:
        tires = {r["Option"]: r["Benefit"] for r in _FALLBACK_ROWS if _lower(r.get("Type")) == "tires"}
        log.warning("[ai_logic] Tires bucket was empty; using fallback seed (%d items).", len(tires))

    if wants["attachments"] and not attachments:
        attachments = {r["Option"]: r["Benefit"] for r in _FALLBACK_ROWS if _lower(r.get("Type")) == "attachments"}
        log.warning("[ai_logic] Attachments bucket was empty; using fallback seed (%d items).", len(attachments))

    if wants["options"] and not options:
        options = {r["Option"]: r["Benefit"] for r in _FALLBACK_ROWS if _lower(r.get("Type")) == "options"}
        log.warning("[ai_logic] Options bucket was empty; using fallback seed (%d items).", len(options))

    # carve telemetry out of options by name/benefit text
    telemetry = {
        n: b for n, b in options.items()
        if _TELEM_PAT.search(_lower(n + " " + (b or "")))
    }

    def postfilter_opts(lst: List[Dict[str, str]]) -> List[Dict[str, str]]:
        out = lst[:]
        if env["cold"]:
            out = [o for o in out if "air conditioner" not in _lower(o.get("name"))]
        if env["dark"]:
            # float lighting to top
            def _is_light(x: Dict[str, str]) -> bool:
                t = _lower((x.get("name","") + " " + x.get("benefit","")))
                return any(w in t for w in ("light","led","beacon","work light","blue light","rear working light"))
            lights = [x for x in out if _is_light(x)]
            non    = [x for x in out if x not in lights]
            out = lights + non
        return out

    def rank_all(bucket: Dict[str, str], k: int = 0) -> List[Dict[str, str]]:
        """Rank a bucket; if ranking yields nothing, return the full bucket as list."""
        ranked = _rank_bucket(user_q, bucket, limit=k)
        if not ranked:
            ranked = [{"name": n, "benefit": b} for n, b in bucket.items()]
        return ranked

    result: dict[str, list] = {}

    # Explicit requests → return only those sections
    if wants["any"]:
        if wants["tires"]:
            ranked_tires = rank_all(tires, k=0)
            if env["asks_non_mark"]:
                rt2 = [t for t in ranked_tires if "non-mark" in _lower(t["name"] + " " + t.get("benefit",""))]
                if rt2:
                    ranked_tires = rt2
            result["tires"] = ranked_tires[:limit] if limit else ranked_tires

        if wants["attachments"]:
            ranked_atts = rank_all(attachments, k=0)
            if env["indoor"] and not env["mentions_clamp"]:
                ranked_atts = [a for a in ranked_atts if not re.search(r"\bclamp\b", _lower(a["name"]))]
            if env["indoor"]:
                ranked_atts.sort(
                    key=lambda a: int(("sideshifter" in _lower(a["name"])) or ("fork positioner" in _lower(a["name"]))),
                    reverse=True
                )
            result["attachments"] = ranked_atts[:limit] if limit else ranked_atts

        if wants["options"]:
            ranked_opts = postfilter_opts(rank_all(options, k=0))
            result["options"] = ranked_opts[:limit] if limit else ranked_opts

        if wants["telemetry"]:
            tel = rank_all(telemetry, k=0)
            result["telemetry"] = tel[:limit] if limit else tel

        # Basic observability
        log.info("[ai_logic] catalog explicit → sizes: tires=%s atts=%s opts=%s telem=%s",
                 len(result.get("tires", [])), len(result.get("attachments", [])),
                 len(result.get("options", [])), len(result.get("telemetry", [])))
        return result

    # Broad question (no explicit keywords): default to options + some attachments
    ranked_opts = postfilter_opts(rank_all(options, k=0))
    ranked_atts = rank_all(attachments, k=0)

    if env["indoor"] and not env["mentions_clamp"]:
        ranked_atts = [a for a in ranked_atts if not re.search(r"\bclamp\b", _lower(a["name"]))]
        ranked_atts.sort(
            key=lambda a: int(("sideshifter" in _lower(a["name"])) or ("fork positioner" in _lower(a["name"]))),
            reverse=True
        )

    result["options"] = ranked_opts[: (limit or 6)]
    result["attachments"] = ranked_atts[:4]

    if _TIRES_PAT.search(user_q or ""):
        rt = rank_all(tires, k=0)
        result["tires"] = rt[:4]
    if _TELEM_PAT.search(user_q or ""):
        tel = rank_all(telemetry, k=0)
        result["telemetry"] = tel[:3]

    log.info("[ai_logic] catalog broad → sizes: tires=%s atts=%s opts=%s telem=%s",
             len(result.get("tires", [])), len(result.get("attachments", [])),
             len(result.get("options", [])), len(result.get("telemetry", [])))
    return result

# ─────────────────────────────────────────────────────────────────────────────
# Markdown renderers (kept extremely defensive)
# ─────────────────────────────────────────────────────────────────────────────
def render_sections_markdown(result: dict) -> str:
    """
    Render only sections that have items. Skips empty/absent sections.
    Sections: 'tires', 'attachments', 'options', 'telemetry'
    """
    order = ["tires", "attachments", "options", "telemetry"]
    labels = {"tires": "Tires", "attachments": "Attachments", "options": "Options", "telemetry": "Telemetry"}

    if not isinstance(result, dict):
        return "(no matching items)"

    lines: List[str] = []
    for key in order:
        arr = result.get(key) or []
        if not arr:
            continue
        section_lines = []
        seen = set()
        for item in arr:
            if not isinstance(item, dict):
                continue
            name = _norm(item.get("name"))
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            ben = _norm(item.get("benefit", "")).replace("\n", " ")
            section_lines.append(f"- {name}" + (f" — {ben}" if ben else ""))
        if section_lines:
            lines.append(f"**{labels[key]}:**")
            lines.extend(section_lines)
    return "\n".join(lines) if lines else "(no matching items)"

def render_catalog_sections(result: dict, max_per_section: int | None = None, **_: Any) -> str:
    """
    Backwards-compatible wrapper some older routes call with a 'max_per_section' kwarg.
    We ignore that arg and simply render what we have (already size-limited upstream).
    """
    return render_sections_markdown(result)

# A simple utility some routes expect to exist (lists everything for a section)
def list_all_from_excel(section: str) -> List[Dict[str, str]]:
    opts, atts, tires = load_catalogs()
    section = (section or "").strip().lower()
    if section == "options":
        return [{"name": k, "benefit": v} for k, v in opts.items()]
    if section == "attachments":
        return [{"name": k, "benefit": v} for k, v in atts.items()]
    if section == "tires":
        return [{"name": k, "benefit": v} for k, v in tires.items()]
    return []

# ─────────────────────────────────────────────────────────────────────────────
# Forklift recommendation stubs (safe, minimal, never crash)
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_models() -> List[Dict[str, Any]]:
    if os.path.exists(_MODELS_JSON):
        try:
            with open(_MODELS_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "models" in data:
                return _as_list(data["models"])
        except Exception as e:
            log.warning("[ai_logic] Failed to read models.json: %s", e)
    # very small placeholder so recommendation flow won't crash
    return [{"Model": "CPD25-GA2CLi", "Power": "Electric", "Capacity_lbs": 5000},
            {"Model": "CPCD50",       "Power": "Diesel",   "Capacity_lbs": 11000}]

@lru_cache(maxsize=1)
def _load_accounts() -> List[Dict[str, Any]]:
    if os.path.exists(_ACCOUNTS_JSON):
        try:
            with open(_ACCOUNTS_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "accounts" in data:
                return _as_list(data["accounts"])
        except Exception as e:
            log.warning("[ai_logic] Failed to read accounts.json: %s", e)
    return []

def generate_forklift_context(user_q: str, account: Dict[str, Any] | None = None) -> str:
    """
    Accepts (user_q, account) to match your route's call signature.
    Returns a small text context the chat prompt can use.
    """
    bits = []
    if account:
        name = account.get("name") or account.get("Sold to Name") or account.get("account") or ""
        industry = account.get("industry") or account.get("SIC") or account.get("R12 Segment (Ship to ID)") or ""
        if name:
            bits.append(f"Customer: {name}")
        if industry:
            bits.append(f"Industry/Segment: {industry}")
    bits.append(f"Question: {user_q}")
    return "\n".join(bits)

def model_meta_for(m: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(m, dict):
        return {"name": "N/A", "power": "", "capacity_lbs": None}
    name = _norm(m.get("Model") or m.get("Model Name") or m.get("code") or m.get("name") or "N/A")
    power = _norm(m.get("Power") or m.get("Drive") or m.get("Power Type") or "")
    cap   = None
    for k in ("Capacity_lbs","Capacity (lbs)","Rated Capacity (lbs)","capacity_lbs","Load Capacity (lbs)","Rated Capacity"):
        if k in m and str(m[k]).strip():
            try:
                cap = float(re.findall(r"-?\d+(?:\.\d+)?", str(m[k]))[0])
                break
            except Exception:
                pass
    return {"name": name, "power": power, "capacity_lbs": cap}

def filter_models(user_q: str, k: int = 8) -> List[Dict[str, Any]]:
    """Naive keyword filter to keep flow working; replace with your real retrieval when ready."""
    models = _load_models()
    ql = _lower(user_q)
    scored = []
    for m in models:
        meta = model_meta_for(m)
        text = _lower(meta["name"] + " " + (meta["power"] or ""))
        s = 0.01
        # tiny signals
        if "electric" in ql and "electric" in text: s += 1.0
        if "diesel" in ql and "diesel" in text:     s += 1.0
        if "5000" in ql and (meta["capacity_lbs"] or 0) >= 4500 and (meta["capacity_lbs"] or 0) <= 6000: s += 1.2
        if "indoor" in ql and "electric" in text:   s += 0.4
        if "rough" in ql and "diesel" in text:      s += 0.4
        scored.append((s, m))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [m for _, m in scored[:k]]

def select_models_for_question(user_q: str, k: int = 5) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns (hits, allowed) to match your route's tuple-unpack.
    Never returns None; never raises 'not enough values to unpack'.
    """
    hits = filter_models(user_q, k=k)
    allowed = hits[:]  # in a real system you could apply compliance filters here
    return hits, allowed

def allowed_models_block(models: List[Dict[str, Any]]) -> str:
    if not models:
        return "(no matching models)"
    lines = ["**Recommended Models:**"]
    for m in models:
        meta = model_meta_for(m)
        bits = [meta["name"]]
        if meta["power"]: bits.append(meta["power"])
        if meta["capacity_lbs"]: bits.append(f'{int(meta["capacity_lbs"]):,} lbs')
        lines.append("- " + " — ".join(bits))
    return "\n".join(lines)

def top_pick_meta(model: Dict[str, Any]) -> Dict[str, Any]:
    """Thin helper so existing imports don't fail."""
    return model_meta_for(model)

def debug_parse_and_rank(q: str) -> Dict[str, Any]:
    """Debug aid if you want to log how a query was parsed and ranked."""
    opts, atts, tires = load_catalogs()
    return {
        "wants": _wants_sections(q),
        "env": _env_flags(q),
        "preview": {
            "options_top3": _rank_bucket(q, opts, limit=3),
            "attachments_top3": _rank_bucket(q, atts, limit=3),
            "tires_top3": _rank_bucket(q, tires, limit=3),
        }
    }

# ─────────────────────────────────────────────────────────────────────────────
# __all__ for explicit consumers
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
    "_wants_sections", "_env_flags",
]

# ─────────────────────────────────────────────────────────────────────────────
# Small compatibility shim (if something imports it)
# ─────────────────────────────────────────────────────────────────────────────
def generate_catalog_mode_response(user_q: str, limit: int = 6) -> str:
    """
    Convenience: select + sanitize + render as markdown.
    """
    raw = recommend_options_from_sheet(user_q, limit=limit)
    safe = sanitize_catalog_result(user_q, raw)
    return render_sections_markdown(safe)
