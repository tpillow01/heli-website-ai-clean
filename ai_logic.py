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
_OPTIONS_XLSX = os.environ.get(
    "HELI_CATALOG_XLSX",
    os.path.join(os.path.dirname(__file__), "data", "forklift_options_benefits.xlsx")
)

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
# Catalog loader (Excel)
# ─────────────────────────────────────────────────────────────────────────────
def _canon_subcat(s: str) -> str:
    if not s: return ""
    key = _norm_spaces(str(s)).lower()
    canon = {
        "hydraulic assist": "Hydraulic Assist",
        "filtration/ cooling": "Filtration/Cooling",
        "filtration/cooling": "Filtration/Cooling",
        "fork handling": "Fork Handling",
        "hydraulic control": "Hydraulic Control",
        "tire": "Tire",
        "tires": "Tire",
        "safety lighting": "Safety Lighting",
        "protection": "Protection",
        "cooling/filtration": "Filtration/Cooling",
        "weather/cab": "Weather/Cab",
        "telemetry": "Telemetry",
    }
    return canon.get(key, s if isinstance(s, str) else str(s))

def _read_catalog_df():
    if _pd is None:
        log.warning("pandas not installed; catalog disabled.")
        return None
    if not os.path.exists(_OPTIONS_XLSX):
        log.warning("[ai_logic] catalog Excel not found: %s", _OPTIONS_XLSX)
        return None
    try:
        df = _pd.read_excel(_OPTIONS_XLSX, engine="openpyxl")
    except Exception:
        df = _pd.read_excel(_OPTIONS_XLSX)
    if df is None or df.empty:
        log.warning("[ai_logic] Excel loaded but empty.")
        return None

    cols = {str(c).lower().strip(): c for c in df.columns}
    name_col    = cols.get("name") or cols.get("option")
    benefit_col = cols.get("benefit") or cols.get("description") or cols.get("desc")
    type_col    = cols.get("type") or cols.get("category")
    subcat_col  = cols.get("subcategory")

    if not name_col:
        log.error("[ai_logic] Excel must have a 'Name' or 'Option' column.")
        return None

    df = df.copy()
    df["__name__"] = df[name_col].astype(str).str.strip()
    df["__benefit__"] = (df[benefit_col].astype(str).str.strip() if benefit_col else "")
    df["__type__"] = (df[type_col].astype(str).str.strip().str.lower() if type_col else "")
    df["__subcategory__"] = (df[subcat_col].astype(str).str.strip() if subcat_col else "")
    df["__subcategory__"] = df["__subcategory__"].map(_canon_subcat)

    # normalize type buckets
    df["__type__"] = df["__type__"].replace({
        "options": "option","opt":"option","option":"option",
        "attachments":"attachment","att":"attachment","attachment":"attachment",
        "tires":"tire","tire":"tire"
    })

    # infer type if missing, based on name
    def _infer_type(nm: str, tp: str) -> str:
        if tp: return tp
        ln = (nm or "").lower()
        if any(k in ln for k in (
            "clamp","sideshift","side shift","side-shift","positioner","rotator",
            "boom","pole","ram","fork extension","extensions","push/ pull","push/pull",
            "slip-sheet","slipsheet","bale","carton","drum","load stabilizer","inverta","bag push"
        )):
            return "attachment"
        if any(k in ln for k in ("tire","tyre","pneumatic","cushion","non-mark","dual","solid tire")):
            return "tire"
        return "option"

    df["__type__"] = df.apply(lambda r: _infer_type(r["__name__"], r["__type__"]), axis=1)

    out = df.loc[df["__name__"] != "", ["__name__","__benefit__","__type__","__subcategory__"]].rename(
        columns={"__name__":"name","__benefit__":"benefit","__type__":"type","__subcategory__":"subcategory"}
    )
    log.info("[ai_logic] Using catalog: %s (exists=%s)", _OPTIONS_XLSX, os.path.exists(_OPTIONS_XLSX))
    return out

@lru_cache(maxsize=1)
def load_catalogs() -> tuple[dict, dict, dict]:
    """
    Returns dictionaries keyed by item name:
      options:     { name: benefit }
      attachments: { name: benefit }
      tires:       { name: benefit }
    """
    df = _read_catalog_df()
    if df is None or df.empty:
        return {}, {}, {}

    df = df.copy()
    df["name"]    = df["name"].astype(str).str.strip()
    df["benefit"] = df["benefit"].fillna("").astype(str).str.strip()
    df["type"]    = df["type"].astype(str).str.strip().str.lower()
    df = df[df["type"].isin({"option","attachment","tire"})]

    def bucket(t: str) -> dict[str,str]:
        sub = df[df["type"] == t]
        sub = sub.loc[~sub["name"].str.lower().duplicated(keep="last")]
        return {r["name"]: r["benefit"] for _, r in sub.iterrows()}

    options     = bucket("option")
    attachments = bucket("attachment")
    tires       = bucket("tire")
    log.info("[ai_logic] Loaded buckets: tires=%d attachments=%d options=%d",
             len(tires), len(attachments), len(options))
    return options, attachments, tires

@lru_cache(maxsize=1)
def load_catalog_rows() -> list[dict]:
    df = _read_catalog_df()
    if df is None or df.empty:
        return []
    return df.to_dict(orient="records")

def refresh_catalog_caches():
    for fn in (load_catalogs, load_catalog_rows, load_options, load_attachments,
               load_tires_as_options, options_lookup_by_name):
        try:
            fn.cache_clear()
        except Exception:
            pass

# Legacy-shaped quick accessors (some parts of app import these)
@lru_cache(maxsize=1)
def load_options() -> List[dict]:
    options, _, tires = load_catalogs()
    # include tires as options-like entries so legacy screens still see codes
    rows = []
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

# Alias used elsewhere
load_tires = load_tires_as_options

@lru_cache(maxsize=1)
def options_lookup_by_name() -> dict:
    return {o["name"].lower(): o for o in load_options()}

def option_benefit(name: str) -> Optional[str]:
    row = options_lookup_by_name().get((name or "").lower())
    return row["benefit"] if row else None

# ─────────────────────────────────────────────────────────────────────────────
# Intent parsing for “list all …” & section filtering
# ─────────────────────────────────────────────────────────────────────────────
_BOTH_PAT  = re.compile(r'\b(attachments\s+and\s+options|options\s+and\s+attachments|both\s+lists?)\b', re.I)
_ATT_PAT   = re.compile(r'\b(attachments?\b|fork\s*attachments?)\b', re.I)
_OPT_PAT   = re.compile(r'\b(options?\b)\b', re.I)
_TIRES_PAT = re.compile(r'\b(tires?|tyres?|tire\s*types?)\b', re.I)
_TELEM_PAT = re.compile(r'\b(telem(?:atics)?|fics|fleet\s*(?:mgmt|management)|smart\s*fleet)\b', re.I)

def parse_catalog_intent(user_q: str) -> dict:
    t = _lower(user_q)
    which = None
    if _BOTH_PAT.search(t) or ("attachments" in t and "options" in t):
        which = "both"
    elif _ATT_PAT.search(t) and not _OPT_PAT.search(t):
        which = "attachments"
    elif _OPT_PAT.search(t) and not _ATT_PAT.search(t):
        which = "options"
    elif _TIRES_PAT.search(t):
        which = "tires"
    elif _TELEM_PAT.search(t):
        which = "telemetry"
    list_all = (
        bool(re.search(r'\b(list|show|give|display)\b.*\b(all|full|everything)\b', t)) or
        "tire types" in t or "types of tires" in t or "all tires" in t or
        "full list" in t
    )
    return {"which": which, "list_all": list_all}

# ─────────────────────────────────────────────────────────────────────────────
# Catalog renderers
# ─────────────────────────────────────────────────────────────────────────────
def _lines_of(bucket: Dict[str, str]) -> List[str]:
    if not bucket:
        return ["- (none found)"]
    items = sorted(bucket.items(), key=lambda kv: kv[0].lower())
    return [f"- {n}" + (f" — {b}" if b else "") for n, b in items]

def render_catalog_sections(user_text: str, max_per_section: int = 9999) -> str:
    """
    Renders only the section(s) requested.
    If user specifically asks about tires/telemetry/attachments/options, we show only that section.
    For 'both' or 'list all', we include all relevant sections.
    """
    intent = parse_catalog_intent(user_text)
    which = intent["which"]
    options, attachments, tires = load_catalogs()

    # Simple telemetry filter from the options sheet
    telemetry = {n: b for n, b in options.items() if re.search(r"\b(fics|fleet|telem|portal)\b", _lower(n + " " + b))}

    out: List[str] = []

    def take(bucket: Dict[str,str], title: str):
        lines = _lines_of(bucket)
        out.append(f"**{title}:**")
        out.extend(lines[:max_per_section])

    if which == "tires":
        take(tires, "Tires")
    elif which == "attachments":
        take(attachments, "Attachments")
    elif which == "options":
        # if the question looks like telemetry, show telemetry subset only
        if _TELEM_PAT.search(_lower(user_text)):
            take(telemetry, "Telemetry")
        else:
            take(options, "Options")
    elif which == "telemetry":
        take(telemetry, "Telemetry")
    else:
        # both or unspecified => polite full dump
        if tires:       take(tires, "Tires")
        if attachments: take(attachments, "Attachments")
        if options:     take(options, "Options")

    return "\n".join(out).strip() or "(no catalog content)"

def list_all_from_excel(user_text: str, max_per_section: int = 9999) -> str:
    """Public alias used by some routes; same behavior as render_catalog_sections."""
    return render_catalog_sections(user_text, max_per_section=max_per_section)

def generate_catalog_mode_response(user_q: str, max_per_section: int = 6) -> str:
    """
    Older call-site helper. Respects 'which' logic to avoid showing
    unrelated headers.
    """
    return render_catalog_sections(user_q, max_per_section=max_per_section)

# ─────────────────────────────────────────────────────────────────────────────
# JSON loads (accounts + models)
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

log.info("[ai_logic] Loaded accounts=%s | models=%s", len(accounts_raw), len(models_raw))

# ─────────────────────────────────────────────────────────────────────────────
# Requirement parsing (capacity/height/aisle/power/env) — lightweight
# ─────────────────────────────────────────────────────────────────────────────
def _parse_capacity_lbs_intent(text: str) -> tuple[Optional[int], Optional[int]]:
    if not text: return (None, None)
    t = _lower(text).replace("–","-").replace("—","-")
    t = re.sub(r"[~≈≃∼]", "", t)

    UNIT_LB     = r'(?:lb\.?|lbs\.?|pound(?:s)?)'
    UNIT_KG     = r'(?:kg|kgs?|kilogram(?:s)?)'
    UNIT_TONNE  = r'(?:tonne|tonnes|metric\s*ton(?:s)?|(?<!f)\bt\b)'
    UNIT_TON    = r'(?:ton|tons)'
    KNUM        = r'(\d+(?:\.\d+)?)\s*k\b'
    NUM         = r'(\d[\d,\.]*)'

    def _n(s: str) -> float: return float(s.replace(",", ""))

    m = re.search(rf'(?:{KNUM}|{NUM})\s*\+\s*(?:{UNIT_LB})?', t)
    if m:
        val = m.group(1) or m.group(2)
        v = float(val.replace(",", ""))
        return (int(round(v*1000)) if m.group(1) else int(round(v)), None)
    m = re.search(rf'{KNUM}\s*-\s*{KNUM}', t)
    if m: return (int(round(_n(m.group(1))*1000)), int(round(_n(m.group(2))*1000)))
    m = re.search(rf'{NUM}\s*-\s*{NUM}\s*(?:{UNIT_LB})?', t)
    if m:
        a, b = int(round(_n(m.group(1)))), int(round(_n(m.group(2))))
        return (min(a,b), max(a,b))
    m = re.search(rf'(?:up to|max(?:imum)?)\s+{NUM}\s*(?:{UNIT_LB})?', t)
    if m: return (None, int(round(_n(m.group(1)))))
    m = re.search(rf'(?:at least|minimum|min)\s+{NUM}\s*(?:{UNIT_LB})?', t)
    if m: return (int(round(_n(m.group(1)))), None)
    m = re.search(rf'{NUM}\s*{UNIT_LB}\b', t)
    if m: return (int(round(_n(m.group(1)))), None)
    m = re.search(rf'{NUM}\s*{UNIT_KG}\b', t)
    if m: return (int(round(_n(m.group(1))*2.20462)), None)
    m = re.search(rf'{NUM}\s*{UNIT_TONNE}\b', t)
    if m: return (int(round(_n(m.group(1))*2204.62)), None)
    m = re.search(rf'{NUM}\s*{UNIT_TON}\b', t)
    if m: return (int(round(_n(m.group(1))*2000)), None)
    near = re.search(r'(?:capacity|load|payload|lift|handle|carry|weight)\D{0,12}(\d{4,5})\b', t)
    if near: return (int(near.group(1)), None)
    return (None, None)

def _parse_requirements(q: str) -> Dict[str,Any]:
    ql = _lower(q)
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
        except:
            pass

    if height_in is None:
        m = re.search(r'(?:lift|raise|reach|height|clearance|mast)\D{0,12}(\d[\d,\.]*)\s*(ft|feet|\'|in|\"|inches)', ql)
        if m:
            raw, unit = m.group(1), m.group(2)
            try:
                height_in = _to_inches(float(raw.replace(",","")), "ft") if unit in ("ft","feet","'") else float(raw.replace(",",""))
            except:
                height_in = None

    aisle_in = None
    m = re.search(r'(?:aisle|aisles|aisle width)\D{0,12}(\d[\d,\.]*)\s*(?:in|\"|inches|ft|\')', ql)
    if m:
        raw, unitblob = m.group(1), m.group(0)
        try:
            aisle_in = _to_inches(float(raw.replace(",","")), "ft") if ("ft" in unitblob or "'" in unitblob) else float(raw.replace(",",""))
        except:
            pass

    if aisle_in is None:
        m = re.search(r'(?:right[-\s]?angle(?:\s+aisle|\s+stack(?:ing)?)?|ra\s*aisle|ras)\D{0,12}(\d[\d,\.]*)\s*(in|\"|inches|ft|\')', ql)
        if m:
            raw, unit = m.group(1), m.group(2)
            try:
                aisle_in = _to_inches(float(raw.replace(",","")), "ft") if unit in ("ft","'") else float(raw.replace(",",""))
            except:
                pass

    power_pref = None
    if any(w in ql for w in ["zero emission","emissions-free","battery","lithium","li-ion","lead-acid","electric"]):
        power_pref = "electric"
    if "diesel" in ql: power_pref = "diesel"
    if any(w in ql for w in ["lpg","propane","lp gas","gas-powered"]): power_pref = "lpg"

    indoor  = any(w in ql for w in ["indoor","warehouse","inside","factory floor","distribution center","dc"])
    outdoor = any(w in ql for w in ["outdoor","yard","construction","gravel","dirt","uneven","rough","pavement","parking lot"])
    narrow  = ("narrow aisle" in ql) or ("very narrow" in ql) or ("vna" in ql) or ("turret" in ql) or ("reach truck" in ql) or ("stand-up reach" in ql) or (aisle_in is not None and aisle_in <= 96)

    tire_pref = None
    if re.search(r"non[-\s]?mark", ql): tire_pref = "non-marking cushion"
    if tire_pref is None and any(w in ql for w in ["cushion","press-on","press on"]): tire_pref = "cushion"
    if any(w in ql for w in ["pneumatic","air filled","air-filled","rough terrain tires","solid pneumatic","super elastic","foam filled","foam-filled"]):
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
    if not pref: return True
    p = pref.lower()
    t = (powr_text or "").lower()
    if p == "electric": return any(x in t for x in ("electric","lithium","li-ion","lead","battery"))
    if p == "lpg": return any(x in t for x in ("lpg","propane","lp gas","gas"))
    if p == "diesel": return "diesel" in t
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
        if cap_need and cap_need >= 4500 and three_wheel:
            s -= 0.8
        s += 0.05
        scored.append((s, m))

    if not scored:
        return []
    ranked = sorted(scored, key=lambda t: t[0], reverse=True)
    return [m for _, m in ranked[:limit]]

# ─────────────────────────────────────────────────────────────────────────────
# Context block builder and public helpers used by your routes
# ─────────────────────────────────────────────────────────────────────────────
def _class_of(row: Dict[str, Any]) -> str:
    t = _text_from_keys(row, TYPE_KEYS)
    m = re.search(r'\bclass\s*([ivx]+)\b', (t or ""), re.I)
    if m:
        return m.group(1).upper()
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

def select_models_for_question(user_q: str, k: int = 5):
    """
    Legacy signature expected by heli_backup_ai.py.
    Always returns (hits, allowed_names_list) without raising.
    """
    try:
        hits = filter_models(user_q, limit=k)
    except Exception:
        hits = []
    allowed = []
    for m in hits:
        nm = _safe_model_name(m)
        if nm and nm != "N/A":
            allowed.append(nm)
    return hits, allowed

def allowed_models_block(allowed: List[str]) -> str:
    if not allowed:
        return "ALLOWED MODELS:\n(none – say 'No exact match from our lineup.')"
    return "ALLOWED MODELS:\n" + "\n".join(f"- {x}" for x in allowed)

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
    want = _parse_requirements(user_q)
    env = "Indoor" if (want["indoor"] and not want["outdoor"]) else ("Outdoor" if (want["outdoor"] and not want["indoor"]) else "Mixed/Not specified")
    hits = filter_models(user_q)

    lines: List[str] = []
    if acct:
        lines.append(customer_block(acct))

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

    # Keep this context compact; your UI shows Options/Attachments separately
    return "\n".join(lines)

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

__all__ = [
    # Catalog IO / caches
    "load_catalogs", "load_catalog_rows", "refresh_catalog_caches",
    "load_options", "load_attachments", "load_tires_as_options", "load_tires",
    "options_lookup_by_name", "option_benefit",

    # Scenario picks & catalog renderers
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

# Keep Pylance happy for legacy imports
_LEGACY_EXPORTS: Dict[str, Any] = {
    "list_all_from_excel": list_all_from_excel,
    "num_from_keys": _num_from_keys,
    "is_attachment": _is_attachment,
}
if hashlib.md5(str(sorted(_LEGACY_EXPORTS.keys())).encode()).hexdigest():
    pass
