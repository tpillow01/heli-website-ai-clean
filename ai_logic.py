"""
ai_logic.py — full replacement (2025‑11‑12)

Fixes & behavior:
- ✅ No Pylance undefined-name errors (adds `_make_code`, `_is_attachment`, removes bad export refs).
- ✅ Excel ingestion tolerant to column variants (Name/Option, Benefit/Description, Type/Category, Subcategory).
- ✅ Tire logic never misfires on "tire/tyre clamp".
- ✅ No more defaulting to **Dual Tires** when intent is ambiguous — now returns “(no specific tire triggered)”.
- ✅ "List‑all" works: e.g., “what are all tire types”, “what attachments can I put on my truck”, etc.
- ✅ Telematics/telemetry detection (telemetry/telematics/FICS/fleet mgmt/GPS) with fuzzy Excel matching.
- ✅ Operator comfort broadened (seats, cab, AC, heat) with fuzzy Excel matching.
- ✅ Cold‑weather options improved (heater, cab glass set, lights, cold storage package for electric).
- ✅ Backward‑compatible public API (same functions/exports your app calls).

Notes:
- If pandas/openpyxl aren’t installed, catalog-driven functions gracefully return empty output.
- All list-all functions show Excel content as-is; maintain catalog to control wording.
"""
from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import os, json, re, hashlib, logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as _pd  # type: ignore
except Exception:  # pragma: no cover
    _pd = None

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
_OPTIONS_XLSX = os.environ.get(
    "HELI_CATALOG_XLSX",
    os.path.join(os.path.dirname(__file__), "data", "forklift_options_benefits.xlsx"),
)

# ─────────────────────────────────────────────────────────────────────────────
# Utilities & normalization
# ─────────────────────────────────────────────────────────────────────────────
_DEF_CODE_RX = re.compile(r"[^a-z0-9]+")

def _make_code(name: str) -> str:
    nm = (name or "").strip().lower()
    nm = _DEF_CODE_RX.sub("-", nm).strip("-")
    return nm or "item"


def _norm_ws(s: str | None) -> str:
    s0 = (s or "").strip()
    return " ".join(s0.split()).lower()

_CANON_SUBCAT: Dict[str, str] = {
    "hydraulic assist": "Hydraulic Assist",
    "filtration/ cooling": "Filtration/Cooling",
    "filtration/cooling": "Filtration/Cooling",
    "fork handling": "Fork Handling",
    "hydraulic control": "Hydraulic Control",
    "tire": "Tire",
    "tires": "Tire",
}

def _canon_subcat(s: str | None) -> str:
    if not s:
        return ""
    key = _norm_ws(s)
    return _CANON_SUBCAT.get(key, s.strip())

# Public plain-text cleanup

def _plain(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = (s.replace("“", '"').replace("”", '"')
           .replace("‘", "'").replace("’", "'")
           .replace("—", "-").replace("–", "-"))
    s = s.replace("**", "").replace("__", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" ?\n ?", "\n", s)
    return s.strip()

# ─────────────────────────────────────────────────────────────────────────────
# Tire/option detection regexes
# ─────────────────────────────────────────────────────────────────────────────
_TIRE_WORD = r"(?:tire|tyre)s?"
_RE_CLAMP = re.compile(r"(?i)\bclamp\b")
_RE_NON_MARKING = re.compile(r"(?i)\bnon[-\s]?marking\b|\bnm\s+(?:cushion|pneumatic)\b")
_RE_TIRE_CORE = re.compile(rf"(?i)\b{_TIRE_WORD}\b(?!\s*clamp)")
_RE_TIRE_VARIANTS = re.compile(r"(?i)\b(?:dual(?:\s+solid)?\s+tires?|solid\s+tires?|non[-\s]?marking\s+tires?)\b")

# ─────────────────────────────────────────────────────────────────────────────
# Excel ingestion & normalization
# ─────────────────────────────────────────────────────────────────────────────

def _read_catalog_df():
    """Return normalized DataFrame with columns: name, benefit, type, subcategory."""
    if _pd is None or not os.path.exists(_OPTIONS_XLSX):
        logging.warning("[ai_logic] Excel not found or pandas missing: %s", _OPTIONS_XLSX)
        return None
    try:
        df = _pd.read_excel(_OPTIONS_XLSX, engine="openpyxl")
    except Exception:
        df = _pd.read_excel(_OPTIONS_XLSX)
    if df is None or df.empty:
        logging.warning("[ai_logic] Excel read but empty.")
        return None

    cols = {str(c).lower().strip(): c for c in df.columns}
    name_col    = cols.get("name") or cols.get("option")
    benefit_col = cols.get("benefit") or cols.get("desc") or cols.get("description")
    type_col    = cols.get("type") or cols.get("category")
    subcat_col  = cols.get("subcategory")
    if not name_col:
        logging.error("[ai_logic] Excel must have a 'Name' or 'Option' column.")
        return None

    df = df.copy()
    df["__name__"] = df[name_col].astype(str).str.strip()
    df["__benefit__"] = (df[benefit_col].astype(str).str.strip() if benefit_col else "")
    df["__type__"] = (df[type_col].astype(str).str.strip().str.lower() if type_col else "")
    df["__subcategory__"] = (df[subcat_col].astype(str).str.strip() if subcat_col else "")
    df["__subcategory__"] = df["__subcategory__"].map(_canon_subcat)

    df["__type__"] = df["__type__"].replace({
        "options": "option", "opt": "option", "option": "option",
        "attachments": "attachment", "att": "attachment", "attachment": "attachment",
        "tires": "tire", "tire": "tire",
    })

    def _infer_type(nm: str, tp: str) -> str:
        if tp:
            return tp
        ln = (nm or "").lower()
        if any(k in ln for k in (
            "clamp","sideshift","side shift","side-shift","positioner","rotator",
            "boom","pole","ram","fork extension","extensions","push/ pull","push/pull",
            "slip-sheet","slipsheet","bale","carton","drum","load stabilizer","inverta"
        )):
            return "attachment"
        if any(k in ln for k in ("tire","tyre","pneumatic","cushion","non-mark","dual")):
            return "tire"
        return "option"

    df["__type__"] = df.apply(lambda r: _infer_type(r["__name__"], r["__type__"]), axis=1)

    out = df.loc[df["__name__"] != "", ["__name__","__benefit__","__type__","__subcategory__"]].rename(
        columns={"__name__":"name","__benefit__":"benefit","__type__":"type","__subcategory__":"subcategory"}
    )
    return out

@lru_cache(maxsize=1)
def load_catalogs() -> tuple[dict, dict, dict]:
    """Return (options, attachments, tires) as dict[name] = benefit."""
    df = _read_catalog_df()
    if df is None or df.empty:
        logging.warning("[ai_logic] load_catalogs(): Excel missing/empty")
        return {}, {}, {}

    for col in ("name","type","benefit"):
        if col not in df.columns:
            raise KeyError(f"[ai_logic] Missing required column in catalog: {col}")

    df = df.copy()
    df["name"]    = df["name"].astype(str).str.strip()
    df["benefit"] = df["benefit"].fillna("").astype(str).str.strip()
    df["type"]    = df["type"].astype(str).str.strip().str.lower()
    df = df[df["type"].isin({"option","attachment","tire"})]

    def bucket_dict(t: str) -> dict[str,str]:
        sub = df[df["type"] == t]
        sub = sub.loc[~sub["name"].str.lower().duplicated(keep="last")]
        return {row["name"]: row["benefit"] for _, row in sub.iterrows()}

    options = bucket_dict("option")
    attachments = bucket_dict("attachment")
    tires = bucket_dict("tire")
    logging.info("[ai_logic] load_catalogs(): options=%d attachments=%d tires=%d",
                 len(options), len(attachments), len(tires))
    return options, attachments, tires

@lru_cache(maxsize=1)
def load_catalog_rows() -> list[dict]:
    df = _read_catalog_df()
    if df is None or df.empty:
        return []
    return df.to_dict(orient="records")

# ─────────────────────────────────────────────────────────────────────────────
# Legacy-shaped accessors
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def load_options() -> List[dict]:
    options, _, tires = load_catalogs()
    rows: List[dict] = []
    for name, ben in {**options, **tires}.items():
        rows.append({"code": _make_code(name), "name": name, "benefit": ben})
    return rows

@lru_cache(maxsize=1)
def load_attachments() -> List[dict]:
    _, attachments, _ = load_catalogs()
    return [{"name": n, "benefit": b} for n, b in attachments.items()]

@lru_cache(maxsize=1)
def load_tires_as_options() -> List[dict]:
    _, _, tires = load_catalogs()
    return [{"code": _make_code(n), "name": n, "benefit": b} for n, b in tires.items()]

# Alias kept for backwards compatibility
load_tires = load_tires_as_options  # type: ignore

@lru_cache(maxsize=1)
def options_lookup_by_name() -> dict:
    return {o["name"].lower(): o for o in load_options()}

def option_benefit(name: str) -> Optional[str]:
    row = options_lookup_by_name().get((name or "").lower())
    return row["benefit"] if row else None


def refresh_catalog_caches() -> None:
    for fn in (load_catalogs, load_options, load_attachments, load_tires_as_options, options_lookup_by_name, load_catalog_rows):
        try:
            fn.cache_clear()  # type: ignore[attr-defined]
        except Exception:
            pass

# ─────────────────────────────────────────────────────────────────────────────
# Catalog intent parsing & list-all rendering
# ─────────────────────────────────────────────────────────────────────────────
_BOTH_PAT  = re.compile(r"\b(both\s+lists?|attachments\s+and\s+options|options\s+and\s+attachments)\b", re.I)
_ATT_PAT   = re.compile(r"\b(attachments?\s+only|attachments?)\b", re.I)
_OPT_PAT   = re.compile(r"\b(options?\s+only|options?)\b", re.I)
_TIRES_PAT = re.compile(r"\b(tires?|tyres?|tire\s*types?)\b", re.I)


def parse_catalog_intent(user_q: str) -> dict:
    t = (user_q or "").strip().lower()
    which = None
    if _BOTH_PAT.search(t) or ("attachments" in t and "options" in t):
        which = "both"
    elif _ATT_PAT.search(t):
        which = "attachments"
    elif _OPT_PAT.search(t):
        which = "options"
    elif _TIRES_PAT.search(t):
        which = "tires"

    # Anything that sounds like “list all / types / show me everything”
    list_all = bool(re.search(r"\b(list|show|give|display|types?)\b.*\b(all|full|everything|types?)\b", t))
    # Also treat “tire types” as list-all tires
    if "tire types" in t or "types of tires" in t:
        list_all = True

    return {"which": which, "list_all": list_all}


def _line(n: str, b: str) -> str:
    b = (b or "").strip()
    return f"- {n}" + (f" — {b}" if b else "")


def _dump_section(pairs: List[Tuple[str, str]], header: str) -> List[str]:
    out = [f"**{header}:**"]
    for nm, ben in pairs:
        out.append(_line(nm, ben))
    return out


def list_all_from_excel(user_text: str, max_per_section: int = 9999) -> str:
    df = _read_catalog_df()
    if df is None or df.empty:
        return "Catalog is empty or not loaded."
    atts = [(r["name"], r["benefit"]) for _, r in df[df["type"] == "attachment"].iterrows()]
    opts = [(r["name"], r["benefit"]) for _, r in df[df["type"] == "option"].iterrows()]
    tires = [(r["name"], r["benefit"]) for _, r in df[df["type"] == "tire"].iterrows()]

    lines: List[str] = []
    if tires: lines += _dump_section(tires[:max_per_section], "Tires")
    if atts:  lines += _dump_section(atts[:max_per_section], "Attachments")
    if opts:  lines += _dump_section(opts[:max_per_section], "Options")
    return "\n".join(lines) if lines else "No items found in the catalog."


# ─────────────────────────────────────────────────────────────────────────────
# Scenario classifiers and ranking (lightweight)
# ─────────────────────────────────────────────────────────────────────────────
_KEYWORDS = {
    "indoor":        [r"\bindoor\b", r"\binside\b", r"\bpolished\b", r"\bepoxy\b"],
    "outdoor":       [r"\boutdoor\b", r"\byard\b", r"\bconstruction\b"],
    "pedestrians":   [r"\bpedestrian", r"\bfoot traffic", r"\bpeople\b", r"\bbusy aisles\b"],
    "tight":         [r"\btight\b", r"\bnarrow\b", r"\baisle", r"\bturn\b"],
    "cold":          [r"\bcold\b", r"\bfreezer\b", r"\b\-?20", r"\bsubzero\b", r"\bcooler\b"],
    "wet":           [r"\bwet\b", r"\bslick\b", r"\bspill", r"\brain", r"\bice\b"],
    "debris":        [r"\bdebris\b", r"\bnails\b", r"\bscrap\b", r"\bpuncture\b"],
    "soft_ground":   [r"\bsoft\b", r"\bgravel\b", r"\bdirt\b", r"\bgrass\b", r"\bsoil\b", r"\bsand\b"],
    "heavy_loads":   [r"\bheavy\b", r"\bmax load\b", r"\bcapacity\b"],
    "long_runs":     [r"\blong runs\b", r"\blong distance\b", r"\bcontinuous\b"],
}

_SUBCAT_HINTS = {
    "Safety Lighting": ["pedestrians", "tight"],
    "Hydraulic Control": ["attachments", "heavy_loads"],
    "Weather/Cab": ["cold", "wet", "outdoor"],
    "Cooling/Filtration": ["dust", "debris", "long_runs", "outdoor"],
    "Protection": ["debris", "outdoor"],
}

_ITEM_BOOSTS = {
    "Blue Light": ["pedestrians", "tight", "busy aisles"],
    "Blue spot Light": ["pedestrians", "tight", "busy aisles"],
    "LED Rotating Light": ["pedestrians"],
    "Visible backward radar": ["pedestrians"],
    "Full OPS": ["pedestrians"],
    "Backup Handle": ["pedestrians"],
    "Fork Positioner": ["tight"],
    "Sideshifter": ["tight"],
    "Cold storage": ["cold"],
    "Heater": ["cold"],
    "Cab": ["cold", "outdoor", "wet"],
    "Windshield": ["wet", "outdoor"],
    "Top Rain-proof Glass": ["wet", "outdoor"],
    "Pre air cleaner": ["debris", "dust", "outdoor"],
    "Dual Air Filter": ["debris", "dust", "outdoor"],
    "Radiator protection": ["debris", "outdoor"],
    "Steel Belly Pan": ["debris", "outdoor"],
}


def _classify(query: str) -> set:
    tags = set()
    q = (query or "").lower()
    for tag, pats in _KEYWORDS.items():
        if any(re.search(p, q) for p in pats):
            tags.add(tag)
    return tags


# ─────────────────────────────────────────────────────────────────────────────
# Text → flags used by Excel pickers
# ─────────────────────────────────────────────────────────────────────────────

def _need_flags_from_text(user_q: str) -> dict:
    t = (user_q or "").lower()
    f: Dict[str, Any] = {}

    # Environment & surface
    f["indoor"]        = re.search(r"\bindoor|warehouse|inside|factory|production|line\b", t) is not None
    f["outdoor"]       = re.search(r"\boutdoor|yard|dock|lot|asphalt|gravel|dirt|parking\b", t) is not None
    f["mixed"]         = ("indoor" in t and "outdoor" in t) or ("both" in t) or ("mixed" in t)
    f["yard"]          = "yard" in t
    f["soft_ground"]   = re.search(r"soft\s*ground|mud|sand", t) is not None
    f["gravel"]        = "gravel" in t or "dirt" in t
    f["rough"]         = re.search(r"rough|uneven|broken|pothole|curb|rail|speed\s*bumps?", t) is not None
    f["debris"]        = re.search(r"debris|nails|screws|scrap|glass|shavings|chips?", t) is not None
    f["puncture"]      = re.search(r"puncture|flats?|tire\s*damage", t) is not None
    f["heavy_loads"]   = re.search(r"\b(7k|7000|8k|8000)\b|heavy\s*loads?|coil|paper\s*rolls?", t) is not None
    f["long_runs"]     = re.search(r"long\s*shifts?|multi[-\s]?shift|continuous", t) is not None

    f["non_marking"]   = bool(re.search(r"non[-\s]?mark|no\s*marks?|black\s*marks?|avoid\s*marks?|scuff", t))

    # Attachments & material
    f["alignment_frequent"] = re.search(r"align|line\s*up|tight\s*aisles|staging", t) is not None
    f["varied_width"]       = bool(re.search(r"vary|mixed\s*pallet|different\s*width|multiple\s*widths|mix\s*of\s*\d+\s*[\"in]?\s*(and|&)\s*\d+\s*[\"in]?\s*pallets?", t))
    f["paper_rolls"]        = re.search(r"paper\s*roll|newsprint|tissue", t) is not None
    f["slip_sheets"]        = re.search(r"slip[-\s]?sheet", t) is not None
    f["carpet"]             = "carpet" in t or "textile" in t
    f["long_loads"]         = bool(re.search(r"long|oversize|over[-\s]?length|overhang|\b\d+\s*[- ]?ft\b|\b\d+\s*foot\b|\b\d+\s*feet\b|crate[s]?", t))
    f["weighing"]           = re.search(r"weigh|scale|check\s*weight", t) is not None

    # People & visibility
    f["pedestrian_heavy"]   = re.search(r"pedestrian|foot\s*traffic|busy|congested|blind\s*corner|walkway", t) is not None
    f["poor_visibility"]    = re.search(r"low\s*light|dim|night|second\s*shift|poor\s*lighting", t) is not None

    # Controls & ergonomics
    f["extra_hydraulics"]   = "4th function" in t or "fourth function" in t
    f["multi_function"]     = "multiple clamp" in t or "multiple attachments" in t
    f["ergonomics"]         = re.search(r"ergonomic|fatigue|wrist|reach|comfort|operator comfort", t) is not None

    # Temperature & weather
    f["cold"]               = re.search(r"cold|freezer|refrigerated|winter|sub-zero|subzero", t) is not None
    f["hot"]                = re.search(r"hot|heat|summer|foundry|high\s*ambient", t) is not None

    # Safety / policy
    f["speed_control"]      = re.search(r"limit\s*speed|speeding|zoned\s*speed", t) is not None
    f["ops_required"]       = re.search(r"ops|operator\s*presence|osha|insurance|audit|policy", t) is not None

    # Power
    f["power_lpg"]          = re.search(r"\b(lpg|propane|lp[-\s]?gas)\b", t) is not None
    f["electric"]           = re.search(r"\b(lithium|li[-\s]?ion|electric|battery)\b", t) is not None

    # Fleet / telemetry
    f["telematics"]         = re.search(r"telemetry|telematics|fics|fleet\s*(mgmt|management)|gps|geo-?fence|impact\s*sensor", t) is not None

    f["_raw_text"]          = user_q
    return f


# ─────────────────────────────────────────────────────────────────────────────
# Name LUT for Excel lookups (fuzzy-ish)
# ─────────────────────────────────────────────────────────────────────────────

def _norm_key(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[\/\-–—_]+", " ", s)
    s = re.sub(r"[()]+", " ", s)
    s = re.sub(r"\bslip\s*[- ]?\s*sheet\b", "slipsheet", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _lut_by_name(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lut: Dict[str, Dict[str, Any]] = {}
    for it in items or []:
        nm_sheet = (it.get("name") or it.get("Name") or it.get("option") or "").strip()
        if nm_sheet:
            lut[_norm_key(nm_sheet)] = it
    canonical = next((it for it in items if (it.get("name") or it.get("option")) == "Push/ Pull (Slip-Sheet)"), None)
    if canonical:
        lut["push pull slipsheet"] = canonical
    return lut


def _get_with_default(lut: Dict[str, Dict[str, Any]], name: str, default_benefit: str) -> Dict[str, Any]:
    row = lut.get(_norm_key(name))
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
    elif "solid" in nlow and "tire" in nlow:
        tire_best_for = "Outdoor yards, debris/rough pavement"

    if row:
        nm  = row.get("name") or row.get("Name") or name
        ben = (row.get("benefit") or row.get("Benefit") or "").strip() or default_benefit
        out = {"name": nm, "benefit": ben}
    else:
        out = {"name": name, "benefit": default_benefit}
    if tire_best_for:
        out["best_for"] = tire_best_for
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Attachment/Option type guards (for Pylance + clarity)
# ─────────────────────────────────────────────────────────────────────────────

def _is_attachment(row: Dict[str, Any]) -> bool:
    """Heuristic: treat as attachment if name suggests a physical front-end implement."""
    nm = _norm_ws(str(row.get("name") or row.get("Name") or ""))
    tp = _norm_ws(str(row.get("type") or row.get("Type") or ""))
    sub = _canon_subcat(row.get("subcategory") or row.get("Subcategory"))
    if tp == "attachment" or (sub and sub.lower().startswith("fork")):
        return True
    return any(kw in nm for kw in [
        "clamp","sideshift","side shift","positioner","rotator","boom","pole","ram",
        "extension","fork extension","push/pull","push pull","slipsheet","bale","carton","drum",
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Excel-driven pickers
# ─────────────────────────────────────────────────────────────────────────────

def _pick_tire_from_flags(flags: Dict[str, Any], excel_lut: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    f = {k: bool(flags.get(k)) for k in [
        "non_marking", "rough", "debris", "puncture",
        "outdoor", "indoor", "soft_ground", "yard", "gravel", "mixed",
        "heavy_loads", "long_runs", "high_loads"
    ]}

    heavy_or_stability = f.get("heavy_loads") or f.get("high_loads")

    if f.get("non_marking"):
        if f.get("outdoor") or f.get("mixed") or f.get("yard") or heavy_or_stability:
            return _get_with_default(excel_lut, "Non-Marking Dual Tires",
                                     "Dual non-marking tread — keeps floors clean with extra footprint for stability.")
        return _get_with_default(excel_lut, "Non-Marking Tires",
                                 "Non-marking compound — prevents black marks on painted/epoxy floors.")

    if f.get("puncture") or f.get("debris") or f.get("rough") or f.get("gravel"):
        if f.get("soft_ground") or heavy_or_stability:
            return _get_with_default(excel_lut, "Dual Solid Tires",
                                     "Puncture-proof dual solids — added footprint and stability on rough/soft ground.")
        return _get_with_default(excel_lut, "Solid Tires",
                                 "Puncture-proof solid tires — best for debris-prone or rough surfaces.")

    if f.get("yard") or f.get("soft_ground") or f.get("outdoor") or f.get("mixed"):
        return _get_with_default(excel_lut, "Dual Tires",
                                 "Wider footprint for traction and stability on soft or uneven ground.")

    if f.get("indoor") and not f.get("outdoor"):
        return _get_with_default(excel_lut, "Non-Marking Tires",
                                 "Indoor warehouse floors — non-marking avoids scuffs on concrete/epoxy.")

    if f.get("outdoor") and not f.get("indoor"):
        return _get_with_default(excel_lut, "Solid Tires",
                                 "Outdoor pavement/yard — reduced flats and lower maintenance.")

    # Ambiguous → do not force a pick
    return None


def _maybe_add_unique(out: List[Dict[str, Any]], item: Dict[str, Any]) -> None:
    key = _norm_key(item.get("name", ""))
    if key and all(_norm_key(x.get("name", "")) != key for x in out):
        out.append(item)


def _pick_attachments_from_excel(flags: Dict[str, Any], excel_lut: Dict[str, Dict[str, Any]], max_items: int = 6) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    t = (flags.get("_raw_text") or "").lower()
    pallets_mentioned = bool(re.search(r"\bpallet(s)?\b", t))

    def maybe_add(names: List[Tuple[str, str]]):
        for nm, default_ben in names:
            row = excel_lut.get(_norm_key(nm))
            if row:
                _maybe_add_unique(out, _get_with_default(excel_lut, nm, default_ben))

    # Core triggers
    if flags.get("alignment_frequent"):
        maybe_add([("Sideshifter", "Aligns loads without repositioning — faster, cleaner placement.")])
    if flags.get("varied_width"):
        maybe_add([("Fork Positioner", "Adjust fork spread from the seat for mixed pallet sizes.")])
    if flags.get("paper_rolls"):
        maybe_add([("Paper Roll Clamp", "Secure, damage-reducing handling for paper rolls.")])
    if flags.get("slip_sheets"):
        maybe_add([("Push/ Pull (Slip-Sheet)", "Handles slip-sheeted cartons — eliminates pallets.")])
    if flags.get("carpet"):
        maybe_add([("Carpet Pole", "Safe handling of carpet or coil-like rolled goods.")])
    if flags.get("long_loads"):
        maybe_add([("Fork Extensions", "Supports longer or over-length loads safely.")])

    # Cold path: limit to alignment/positioner only (attachments freeze up)
    if flags.get("cold") and not out:
        if flags.get("alignment_frequent") or "tight aisle" in t or pallets_mentioned:
            maybe_add([("Sideshifter", "Aligns loads without repositioning — faster placement.")])
        if flags.get("varied_width"):
            maybe_add([("Fork Positioner", "Seat-adjustable fork spread for mixed pallets.")])
        return out[:max_items]

    # Generic prompt (“what attachments can I put on my truck”): show first N from Excel
    if not out and re.search(r"\bwhat\b.*\battachments\b", t):
        # Provide a quick representative set (alphabetical by key)
        base = [k for k in excel_lut.keys() if _is_attachment(excel_lut[k])]
        base.sort()
        for k in base[:max_items]:
            r = excel_lut[k]
            _maybe_add_unique(out, {"name": r.get("name") or r.get("Name") or k.title(),
                                    "benefit": (r.get("benefit") or r.get("Benefit") or "").strip()})

    return out[:max_items]


def _pick_options_from_excel(flags: Dict[str, Any], excel_lut: Dict[str, Dict[str, Any]], max_items: int = 6) -> List[Dict[str, Any]]:
    picks: List[Dict[str, Any]] = []

    def add_if_match(name: str, default_benefit: str):
        row = excel_lut.get(_norm_key(name))
        if not row:
            return
        ben_txt = (row.get("benefit") or row.get("Benefit") or "")
        if "suspend" in ben_txt.lower():
            return
        _maybe_add_unique(picks, _get_with_default(excel_lut, name, default_benefit))

    # Cold weather / cab package
    if flags.get("cold"):
        for nm, ben in [
            ("Panel mounted Cab", "Weather protection for outdoor duty; reduces operator fatigue."),
            ("Heater", "Keeps operator warm in cold environments; improves productivity."),
            ("Glass Windshield with Wiper", "Visibility in rain/snow; safer outdoor travel."),
            ("Top Rain-proof Glass", "Overhead visibility while shielding precipitation."),
            ("Rear Windshield Glass", "Reduces wind/snow ingress from the rear."),
        ]:
            add_if_match(nm, ben)
        if flags.get("electric"):
            add_if_match("Added cost for the cold storage package (for electric forklift)",
                         "Seals/heaters for freezer rooms; consistent performance in cold aisles.")
        for nm, ben in [
            ("LED Rear Working Light", "Bright, low-draw rear work lighting for dim winter shifts."),
            ("LED Rotating Light", "High-visibility 360° beacon to alert pedestrians."),
            ("Blue Light", "Directional warning for blind corners and intersections."),
            ("Visible backward radar, if applicable", "Audible/visual alerts while reversing."),
        ]:
            if len(picks) >= max_items:
                break
            add_if_match(nm, ben)
        if flags.get("ops_required"):
            add_if_match("Full OPS", "Operator presence system for safety interlocks.")
        if flags.get("speed_control") or flags.get("pedestrian_heavy"):
            add_if_match("Speed Control system (not for diesel engine)", "Limits travel speed to enhance safety.")
        return picks[:max_items]

    # Visibility & pedestrian safety
    if flags.get("pedestrian_heavy") or flags.get("poor_visibility"):
        for nm, ben in [
            ("LED Rotating Light", "High-visibility 360° beacon to alert pedestrians."),
            ("Blue spot Light", "Projects a visible spot to warn pedestrians at intersections."),
            ("Red side line Light", "Creates a visual exclusion zone along the sides."),
            ("Visible backward radar, if applicable", "Audible/visual alerts while reversing."),
            ("LED Rear Working Light", "Bright, low-draw rear lighting."),
            ("Blue Light", "Directional warning for blind corners."),
        ]:
            add_if_match(nm, ben)

    # Weather / heat
    if flags.get("outdoor") or flags.get("hot"):
        for nm, ben in [
            ("Panel mounted Cab", "Weather protection for outdoor duty; reduces fatigue."),
            ("Air conditioner", "Comfort in hot conditions; keeps productivity steady."),
            ("Glass Windshield with Wiper", "Rain/dust visibility."),
            ("Top Rain-proof Glass", "Overhead visibility while shielding precipitation."),
            ("Rear Windshield Glass", "Shields rear from wind/dust."),
        ]:
            add_if_match(nm, ben)

    # Protection / debris
    if flags.get("rough") or flags.get("debris") or flags.get("yard") or flags.get("gravel"):
        for nm, ben in [
            ("Radiator protection bar", "Shields radiator from impacts/debris."),
            ("Steel Belly Pan", "Protects the underside from debris and snags."),
            ("Removable radiator screen", "Keeps fins clear; easy cleaning."),
            ("Dual Air Filter", "Improved filtration for dusty yards."),
            ("Pre air cleaner", "Cyclonic pre-cleaning extends filter life."),
            ("Air filter service indicator", "Prompts timely filter service."),
            ("Tilt or Steering cylinder boot", "Protects cylinder rods from grit."),
        ]:
            add_if_match(nm, ben)

    # Hydraulics / controls
    if flags.get("extra_hydraulics") or flags.get("multi_function"):
        for nm, ben in [
            ("3 Valve with Handle", "Adds third function for attachments; simple handle control."),
            ("4 Valve with Handle", "Adds fourth hydraulic circuit for complex tools."),
            ("5 Valve with Handle", "Maximum hydraulic flexibility for specialized attachments."),
            ("Finger control system(2valve),if applicable.should work together with MSG65 seat",
             "Compact fingertip controls; less reach/wrist strain."),
            ("Finger control system(3valve),if applicable.should work together with MSG65 seat",
             "Fingertip precision for three hydraulic functions; reduced fatigue."),
            ("Finger control system(4valve), if applicable, should work together with MSG65 seat",
             "Fingertip controls for complex four-function attachments."),
        ]:
            add_if_match(nm, ben)

    # Duty cycle
    if flags.get("heavy_loads") or flags.get("long_runs") or flags.get("outdoor"):
        add_if_match("Wet Disc Brake axle for CPCD50-100", "Lower maintenance braking under heavy duty cycles.")

    # Safety / policy
    if flags.get("speed_control"):
        add_if_match("Speed Control system (not for diesel engine)", "Safer aisles via speed limiting.")
    if flags.get("ops_required"):
        add_if_match("Full OPS", "Operator presence system for OSHA-aligned safety interlocks.")

    # Ergonomics / operator comfort
    if flags.get("ergonomics") or flags.get("long_runs"):
        add_if_match("Heli Suspension Seat with armrest", "Improves comfort and posture over long shifts.")
        add_if_match("Grammar Full Suspension Seat MSG65", "Premium suspension; pairs with fingertip controls.")
        # If catalog contains cab/AC/Heater, include as comfort
        add_if_match("Panel mounted Cab", "Enclosed cab panels reduce fatigue and exposure.")
        add_if_match("Air conditioner", "Cools the cab to reduce heat stress.")
        add_if_match("Heater", "Keeps operator warm for safer winter operation.")

    # Fuel-specific
    if flags.get("power_lpg"):
        add_if_match("Swing Out Drop LPG Bracket", "Faster, safer tank changes; reduces strain.")
        add_if_match("LPG Tank", "Extra tank for quick swaps on multi-shift ops.")
        add_if_match("Low Fuel Indicator Light", "Prevents surprise stalls; prompt tank change.")

    if flags.get("electric") and flags.get("cold"):
        add_if_match("Added cost for the cold storage package (for electric forklift)",
                     "Seals/heaters for freezer rooms; consistent performance in cold aisles.")

    # Telematics / telemetry / fleet management — fuzzy scan of Excel
    if flags.get("telematics"):
        tele_keywords = ["fics", "fleet", "telemat", "telemetry", "gps", "geo", "impact", "sensor", "monitor"]
        for key, row in excel_lut.items():
            nm = row.get("name") or row.get("Name") or ""
            if any(k in key for k in tele_keywords) or any(k in nm.lower() for k in tele_keywords):
                _maybe_add_unique(picks, {"name": nm, "benefit": (row.get("benefit") or row.get("Benefit") or "").strip()})

    # Generic “what options…” prompt: give a representative set if still empty
    if not picks and re.search(r"\bwhat\b.*\boptions\b", (flags.get("_raw_text") or "").lower()):
        base = sorted([k for k in excel_lut.keys() if not _is_attachment(excel_lut[k]) and _canon_subcat(excel_lut[k].get("Subcategory") or excel_lut[k].get("subcategory")) != "Tire"])[:max_items]
        for k in base:
            r = excel_lut[k]
            _maybe_add_unique(picks, {"name": r.get("name") or r.get("Name") or k.title(),
                                      "benefit": (r.get("benefit") or r.get("Benefit") or "").strip()})

    return picks[:max_items]


def recommend_options_from_sheet(user_text: str, max_total: int = 6) -> Dict[str, Any]:
    flags = _need_flags_from_text(user_text)
    rows = load_catalog_rows()
    excel_lut = _lut_by_name(rows)
    tire = _pick_tire_from_flags(flags, excel_lut)
    k = max_total if isinstance(max_total, int) and max_total > 0 else 6
    attachments = _pick_attachments_from_excel(flags, excel_lut, max_items=k)
    options = _pick_options_from_excel(flags, excel_lut, max_items=k)
    return {"tire": tire, "attachments": attachments, "options": options}


def render_catalog_sections(user_text: str, max_per_section: int = 6) -> str:
    # If the prompt is clearly a list-all request, dump from Excel
    intent = parse_catalog_intent(user_text)
    if intent.get("which") in {"both", "attachments", "options", "tires"} and intent.get("list_all"):
        if intent["which"] == "tires":
            # Tires only
            _, _, tires = load_catalogs()
            items = [(n, b) for n, b in tires.items()]
            if not items:
                return "Catalog is empty or not loaded."
            return "\n".join(_dump_section(items[:max_per_section], "Tires"))
        return list_all_from_excel(user_text, max_per_section=max_per_section)

    # Otherwise, run scenario picker
    rec = recommend_options_from_sheet(user_text, max_total=max_per_section)
    lines: List[str] = []

    tire = rec.get("tire")
    lines.append("Tires (recommended):")
    if tire:
        lines.append(f"- {tire.get('name','')} — {tire.get('benefit','')}")
    else:
        lines.append("- (no specific tire triggered)")

    lines.append("Attachments (relevant):")
    atts = (rec.get("attachments") or [])[:max_per_section]
    if atts:
        for a in atts:
            lines.append(f"- {a.get('name','')} — {(a.get('benefit','') or '').strip()}")
    else:
        lines.append("- (none triggered)")

    lines.append("Options (relevant):")
    opts = (rec.get("options") or [])[:max_per_section]
    if opts:
        for o in opts:
            lines.append(f"- {o.get('name','')} — {(o.get('benefit','') or '').strip()}")
    else:
        lines.append("- (none triggered)")

    logging.info("[ai_logic] render_catalog_sections: scenario path, max=%s", max_per_section)
    return "\n".join(lines)


def generate_catalog_mode_response(user_q: str, max_per_section: int = 6) -> str:
    return render_catalog_sections(user_q, max_per_section=max_per_section)


# ─────────────────────────────────────────────────────────────────────────────
# Models JSON loading (accounts/models)
# ─────────────────────────────────────────────────────────────────────────────

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

logging.info("[ai_logic] Loaded accounts: %d | models: %d", len(accounts_raw), len(models_raw))

# ─────────────────────────────────────────────────────────────────────────────
# Model helpers & normalization
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


def _text_from_keys(row: Dict[str,Any], keys: List[str]) -> str:
    for k in keys:
        v = row.get(k)
        if v:
            return str(v)
    return ""


def _num(s: Any) -> Optional[float]:
    if s is None:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", str(s))
    return float(m.group(0)) if m else None


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
    drive = (str(row.get("Drive Type", "")) + " " + str(row.get("Drive", ""))).lower()
    model = (str(row.get("Model","")) + " " + str(row.get("Model Name",""))).lower()
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
    t = str(row.get("Tire Type","") or row.get("Tires","") or row.get("Tire","") or "").lower()
    if "non-mark" in t:
        return "non-marking cushion"
    if "cushion" in t or "press" in t:
        return "cushion"
    if "pneumatic" in t or "super elastic" in t or "solid" in t:
        return "pneumatic"
    return t


def _safe_model_name(m: Dict[str, Any]) -> str:
    for k in ("Model","Model Name","model","code","name","Code"):
        if m.get(k):
            return str(m[k]).strip()
    return "N/A"

# ─────────────────────────────────────────────────────────────────────────────
# Requirement parsing (capacity/height/aisle/power/env)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_capacity_lbs_intent(text: str) -> tuple[Optional[int], Optional[int]]:
    if not text:
        return (None, None)
    t = text.lower().replace("–", "-").replace("—", "-")
    t = re.sub(r"[~≈≃∼]", "", t)
    t = re.sub(r"\bapproximately\b|\bapprox\.?\b|\baround\b|\babout\b", "", t)

    UNIT_LB     = r'(?:lb\.?|lbs\.?|pound(?:s)?)'
    UNIT_KG     = r'(?:kg|kgs?|kilogram(?:s)?)'
    UNIT_TONNE  = r'(?:tonne|tonnes|metric\s*ton(?:s)?|(?<!f)\bt\b)'
    UNIT_TON    = r'(?:ton|tons)'
    KNUM        = r'(\d+(?:\.\d+)?)\s*k\b'
    NUM         = r'(\d[\d,\.]*)'
    LOAD_WORDS  = r'(?:capacity|load|payload|rating|lift|handle|carry|weight|wt)'

    def _n(s: str) -> float:
        return float(s.replace(",", ""))

    m = re.search(rf'(?:{KNUM}|{NUM})\s*\+\s*(?:{UNIT_LB})?', t)
    if m:
        val = m.group(1) or m.group(2)
        v = float(val.replace(",", ""))
        return (int(round(v*1000)) if m.group(1) else int(round(v)), None)

    m = re.search(rf'{KNUM}\s*-\s*{KNUM}', t)
    if m:
        return (int(round(_n(m.group(1))*1000)), int(round(_n(m.group(2))*1000)))

    m = re.search(rf'{NUM}\s*-\s*{NUM}\s*(?:{UNIT_LB})?', t)
    if m:
        a, b = int(round(_n(m.group(1)))), int(round(_n(m.group(2))))
        return (min(a,b), max(a,b))

    m = re.search(rf'between\s+{NUM}\s+and\s+{NUM}', t)
    if m:
        a, b = int(round(_n(m.group(1)))), int(round(_n(m.group(2))))
        return (min(a,b), max(a,b))

    m = re.search(rf'(?:up to|max(?:imum)?)\s+{NUM}\s*(?:{UNIT_LB})?', t)
    if m:
        return (None, int(round(_n(m.group(1)))))

    m = re.search(rf'(?:at least|minimum|min)\s+{NUM}\s*(?:{UNIT_LB})?', t)
    if m:
        return (int(round(_n(m.group(1)))), None)

    m = re.search(rf'{LOAD_WORDS}[^0-9k\-]*{KNUM}', t)
    if m:
        return (int(round(_n(m.group(1))*1000)), None)

    m = re.search(rf'{LOAD_WORDS}[^0-9\-]*{NUM}\s*(?:{UNIT_LB})?', t)
    if m:
        return (int(round(_n(m.group(1)))), None)

    m = re.search(rf'{KNUM}\s*(?:{UNIT_LB})?\b', t)
    if m:
        return (int(round(_n(m.group(1))*1000)), None)

    m = re.search(rf'{NUM}\s*{UNIT_LB}\b', t)
    if m:
        return (int(round(_n(m.group(1)))), None)

    m = re.search(rf'{NUM}\s*{UNIT_KG}\b', t)
    if m:
        return (int(round(_n(m.group(1))*2.20462)), None)

    m = re.search(rf'{NUM}\s*{UNIT_TONNE}\b', t)
    if m:
        return (int(round(_n(m.group(1))*2204.62)), None)

    m = re.search(rf'{NUM}\s*{UNIT_TON}\b', t)
    if m:
        return (int(round(_n(m.group(1))*2000)), None)

    m = re.search(rf'\b(\d[\d,]{3,5})\s*(?:{UNIT_LB})\b', t)
    if m:
        return (int(m.group(1).replace(",", "")), None)

    near = re.search(rf'(?:{LOAD_WORDS})\D{{0,12}}(\d{{4,5}})\b', t)
    if near:
        return (int(near.group(1)), None)

    return (None, None)


def _parse_requirements(q: str) -> Dict[str,Any]:
    ql = q.lower()
    cap_min, cap_max = _parse_capacity_lbs_intent(ql)
    cap_lbs = cap_min

    height_in = None
    for m in re.finditer(r'(\d[\d,\.]*)\s*(ft|feet|\'|in|\"|inches)\b', ql):
        raw, unit = m.group(1), m.group(2)
        ctx = ql[max(0, m.start()-18): m.end()+18]
        if re.search(r'\b(aisle|ra\s*aisle|right[-\s]?angle)\b', ctx):
            continue
        try:
            height_in = _to_inches(float(raw.replace(",","")), "ft") if unit in ("ft","feet","'") else float(raw.replace(",",""))
            break
        except:  # noqa: E722
            pass

    if height_in is None:
        m = re.search(r'(?:lift|raise|reach|height|clearance|mast)\D{0,12}(\d[\d,\.]*)\s*(ft|feet|\'|in|\"|inches)', ql)
        if m:
            raw, unit = m.group(1), m.group(2)
            try:
                height_in = _to_inches(float(raw.replace(",","")), "ft") if unit in ("ft","feet","'") else float(raw.replace(",",""))
            except:  # noqa: E722
                height_in = None

    aisle_in = None
    m = re.search(r'(?:aisle|aisles|aisle width)\D{0,12}(\d[\d,\.]*)\s*(?:in|\"|inches|ft|\')', ql)
    if m:
        raw, unitblob = m.group(1), m.group(0)
        try:
            aisle_in = _to_inches(float(raw.replace(",","")), "ft") if ("ft" in unitblob or "'" in unitblob) else float(raw.replace(",",""))
        except:  # noqa: E722
            pass

    if aisle_in is None:
        m = re.search(r'(?:right[-\s]?angle(?:\s+aisle|\s+stack(?:ing)?)?|ra\s*aisle|ras)\D{0,12}(\d[\d,\.]*)\s*(in|\"|inches|ft|\')', ql)
        if m:
            raw, unit = m.group(1), m.group(2)
            try:
                aisle_in = _to_inches(float(raw.replace(",","")), "ft") if unit in ("ft","'") else float(raw.replace(",",""))
            except:  # noqa: E722
                pass

    power_pref = None
    if any(w in ql for w in ["zero emission","zero-emission","emissions free","emissions-free","eco friendly","eco-friendly","green","battery powered","battery-powered","battery","lithium","li-ion","li ion","lead acid","lead-acid","electric"]):
        power_pref = "electric"
    if "diesel" in ql:
        power_pref = "diesel"
    if any(w in ql for w in ["lpg","propane","lp gas","gas (lpg)","gas-powered","gas powered"]):
        power_pref = "lpg"

    indoor  = any(w in ql for w in ["indoor","warehouse","inside","factory floor","distribution center","dc"])
    outdoor = any(w in ql for w in ["outdoor","yard","dock yard","construction","lumber yard","gravel","dirt","uneven","rough","pavement","parking lot","rough terrain","rough-terrain"])
    narrow  = ("narrow aisle" in ql) or ("very narrow" in ql) or ("vna" in ql) or ("turret" in ql) or ("reach truck" in ql) or ("stand-up reach" in ql) or (aisle_in is not None and aisle_in <= 96)

    tire_pref = None
    if any(w in ql for w in ["non-marking","non marking","nonmarking"]):
        tire_pref = "non-marking cushion"
    if tire_pref is None and any(w in ql for w in ["cushion","press-on","press on"]):
        tire_pref = "cushion"
    if any(w in ql for w in ["pneumatic","air filled","air-filled","rough terrain tires","rt tires","knobby","off-road","outdoor tires","solid pneumatic","super elastic","foam filled","foam-filled"]):
        tire_pref = "pneumatic"
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


# ─────────────────────────────────────────────────────────────────────────────
# Model filtering & ranking
# ─────────────────────────────────────────────────────────────────────────────

def _power_matches(pref: Optional[str], powr_text: str) -> bool:
    if not pref:
        return True
    p = pref.lower()
    t = (powr_text or "").lower()
    if p == "electric":
        return any(x in t for x in ("electric","lithium","li-ion","lead","battery"))
    if p == "lpg":
        return any(x in t for x in ("lpg","propane","lp gas","gas"))
    if p == "diesel":
        return "diesel" in t
    return p in t


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

        if cap_need:
            if cap <= 0:
                continue
            if cap < cap_need:
                continue
        if aisle_need and ais and ais > aisle_need:
            continue

        s = 0.0
        if cap_need and cap:
            over = (cap - cap_need) / cap_need
            s += (2.0 - min(2.0, max(0.0, over))) if over >= 0 else -5.0
        if power_pref:
            s += 1.0 if _power_matches(power_pref, powr) else -0.8
        if tire_pref:
            s += 0.6 if (tire_pref in (tire or "")) else -0.2
        if want["indoor"] and not want["outdoor"]:
            if tire == "cushion" or tire == "non-marking cushion":
                s += 0.4
            if "pneumatic" in (tire or ""):
                s -= 0.4
        if want["outdoor"] and not want["indoor"]:
            if "pneumatic" in (tire or ""):
                s += 0.5
            if "cushion" in (tire or ""):
                s -= 0.7
        if aisle_need:
            if ais:
                s += 0.8 if ais <= aisle_need else -1.0
            else:
                s += 0.6 if (narrow and reach_like) else 0.0
        elif narrow:
            s += 0.8 if reach_like else -0.2
        if height_need and hgt:
            s += 0.5 if hgt >= height_need else -0.4
        if cap_need and cap_need >= 4500 and three_wheel:
            s -= 0.8
        s += 0.05
        scored.append((s, m))

    if not scored:
        return []
    ranked = sorted(scored, key=lambda t: t[0], reverse=True)
    return [m for _, m in ranked[:limit]]


# ─────────────────────────────────────────────────────────────────────────────
# Chat context builders
# ─────────────────────────────────────────────────────────────────────────────

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


def generate_forklift_context(user_q: str, acct: Optional[Dict[str, Any]]) -> str:
    lines: List[str] = []
    if acct:
        lines.append(customer_block(acct))

    want = _parse_requirements(user_q)
    env = "Indoor" if (want["indoor"] and not want["outdoor"]) else ("Outdoor" if (want["outdoor"] and not want["indoor"]) else "Mixed/Not specified")

    hits = filter_models(user_q)

    rec = recommend_options_from_sheet(user_q, max_total=6)
    chosen_tire = rec.get("tire")
    attachments = rec.get("attachments", [])
    non_attachments = rec.get("options", [])

    _user_l = (user_q or "").lower()
    if env == "Indoor" and chosen_tire and "dual" in (chosen_tire.get("name", "").lower()):
        has_nonmark = bool(re.search(r'non[-\s]?mark', _user_l))
        has_stability = bool(re.search(r'(ramp|slope|incline|grade|wide load|long load|top heavy|high mast)', _user_l))
        if not has_nonmark and not has_stability:
            nm_row = options_lookup_by_name().get("non-marking tires")
            if nm_row:
                chosen_tire = {"name": nm_row["name"],
                               "benefit": (nm_row.get("benefit") or "Non-marking compound prevents black marks on painted/epoxy floors.")}
            else:
                chosen_tire = {"name": "Non-Marking Tires",
                               "benefit": "Non-marking compound prevents black marks on painted/epoxy floors."}

    lines.append("Customer Profile:")
    lines.append(f"- Environment: {env}")
    lines.append(f"- Capacity Min: {int(round(want['cap_lbs'])):,} lb" if want.get("cap_lbs") else "- Capacity Min: Not specified")

    lines.append("\nModel:")
    if hits:
        top = hits[0]
        top_name = _safe_model_name(top)
        lines.append(f"- Top Pick: {top_name}")
        if len(hits) > 1:
            alts = [_safe_model_name(m) for m in hits[1:5]]
            lines.append(f"- Alternates: {', '.join(alts)}")
        else:
            lines.append("- Alternates: None")
    else:
        lines.append("- Top Pick: N/A")
        lines.append("- Alternates: N/A")

    lines.append("\nPower:")
    if want.get("power_pref"):
        lines.append(f"- {want['power_pref']}")
    else:
        lines.append(f"- {(_text_from_keys(hits[0], POWER_KEYS) if hits else 'Not specified') or 'Not specified'}")

    lines.append("\nCapacity:")
    if want.get("cap_lbs"):
        lines.append(f"- {int(round(want['cap_lbs'])):,} lb")
    else:
        lines.append("- Not specified")

    lines.append("\nTire Type:")
    if chosen_tire:
        lines.append(f"- {chosen_tire['name']} — {chosen_tire.get('benefit','').strip() or ''}".rstrip(" —"))
    else:
        lines.append("- (no specific tire triggered)")

    lines.append("\nAttachments:")
    if attachments:
        for a in attachments:
            benefit = (a.get("benefit","") or "").strip()
            lines.append(f"- {a['name']}" + (f" — {benefit}" if benefit else ""))
    else:
        lines.append("- Not specified")

    lines.append("\nOptions:")
    if non_attachments:
        for o in non_attachments:
            ben = (o.get("benefit","") or "").strip()
            lines.append(f"- {o['name']}" + (f" — {ben}" if ben else ""))
    else:
        lines.append("- Not specified")

    lines.append("\nComparison:")
    if hits:
        lines.append("- Top pick vs peers: HELI advantages typically include tight turning (≈102 in).")
        lines.append("- We can demo on your dock to validate turning, lift, and cycle times.")
    else:
        lines.append("- No model comparison available for the current filters.")

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

    lines.append(user_q)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Public convenience & debug helpers
# ─────────────────────────────────────────────────────────────────────────────

def select_models_for_question(user_q: str, k: int = 5):
    hits = filter_models(user_q, limit=k)
    allowed = []
    for m in hits:
        nm = _safe_model_name(m)
        if nm != "N/A":
            allowed.append(nm)
    return hits, allowed


def _class_of(row: Dict[str, Any]) -> str:
    t = _text_from_keys(row, TYPE_KEYS)
    m = re.search(r"\bclass\s*([ivx]+)\b", (t or ""), re.I)
    if m:
        roman = m.group(1).upper()
        return roman.replace("V","V").replace("X","X")
    t = (t or "").strip().upper()
    if t in {"I","II","III","IV","V"}:
        return t
    return ""


def model_meta_for(row: Dict[str, Any]) -> tuple[str, str, str]:
    code = _safe_model_name(row)
    cls = _class_of(row)
    pwr = _power_of(row) or ""
    return (code, cls, pwr)


def top_pick_meta(user_q: str) -> Optional[tuple[str, str, str]]:
    hits = filter_models(user_q, limit=1)
    if not hits:
        return None
    return model_meta_for(hits[0])


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

        s = 0.0
        if want["cap_lbs"] and cap:
            over = (cap - want["cap_lbs"]) / want["cap_lbs"]
            s += (2.0 - min(2.0, max(0.0, over))) if over >= 0 else -5.0
        if want["power_pref"]:
            s += 1.0 if _power_matches(want["power_pref"], powr) else -0.8
        if want["tire_pref"]:
            s += 0.6 if want["tire_pref"] in (tire or "") else -0.2
        if want["indoor"] and not want["outdoor"]:
            if tire == "cushion" or tire == "non-marking cushion":
                s += 0.4
            if "pneumatic" in (tire or ""):
                s -= 0.4
        if want["outdoor"] and not want["indoor"]:
            if "pneumatic" in (tire or ""):
                s += 0.5
            if "cushion" in (tire or ""):
                s -= 0.7
        if want["aisle_in"]:
            if ais:
                s += 0.8 if ais <= want["aisle_in"] else -1.0
            else:
                s += 0.6 if (want["narrow"] and reach_like) else 0.0
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


# ─────────────────────────────────────────────────────────────────────────────
# Exports
# ─────────────────────────────────────────────────────────────────────────────
__all__ = [
    # Catalog IO / caches
    "load_catalogs", "load_catalog_rows", "refresh_catalog_caches",
    "load_options", "load_attachments", "load_tires_as_options", "load_tires",
    "options_lookup_by_name", "option_benefit",

    # Scenario picks & catalog renderers
    "recommend_options_from_sheet", "render_catalog_sections", "parse_catalog_intent",
    "generate_catalog_mode_response", "list_all_from_excel",

    # Model filtering & context
    "filter_models", "generate_forklift_context", "select_models_for_question", "allowed_models_block",
    "model_meta_for", "top_pick_meta",

    # Debug
    "debug_parse_and_rank",

    # Intentional small helpers
    "_plain", "_line", "_num_from_keys",
]


def allowed_models_block(allowed: List[str]) -> str:
    if not allowed:
        return "ALLOWED MODELS:\n(none – say 'No exact match from our lineup.')"
    return "ALLOWED MODELS:\n" + "\n".join(f"- {x}" for x in allowed)


# Touch-map so Pylance treats helpers as “used” without side effects.
_LEGACY_EXPORTS: Dict[str, Any] = {
    "list_all_from_excel": list_all_from_excel,
    "plain": _plain,
    "line_fmt": _line,
    "num_from_keys": _num_from_keys,
    "is_attachment": _is_attachment,
}
if hashlib.md5(str(sorted(_LEGACY_EXPORTS.keys())).encode()).hexdigest():
    pass
