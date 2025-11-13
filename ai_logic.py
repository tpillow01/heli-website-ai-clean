"""
ai_logic.py
Catalog helpers (tires / attachments / options / telemetry), intent -> picks,
and model recommendation utilities used across the Heli AI app.

This module is deliberately defensive:
- Works even if pandas is unavailable
- Tolerates Excel rows with odd spacing / capitalization
- Never crashes when a section contains strings instead of dicts
- Exposes all symbols your other modules import from ai_logic
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
    import pandas as _pd  # optional; we fall back if not present
except Exception:
    _pd = None

# ─────────────────────────────────────────────────────────────────────────────
# Paths / constants
# ─────────────────────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(__file__)
_DATA_DIR = os.path.join(_BASE_DIR, "data")

_OPTIONS_XLSX = os.environ.get(
    "HELI_CATALOG_XLSX",
    os.path.join(_DATA_DIR, "forklift_options_benefits.xlsx")
)

_MODELS_JSON = os.environ.get(
    "HELI_MODELS_JSON",
    os.path.join(_DATA_DIR, "models.json")
)

_ACCOUNTS_JSON = os.environ.get(
    "HELI_ACCOUNTS_JSON",
    os.path.join(_DATA_DIR, "accounts.json")
)

# Common column-name possibilities for models.json
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
# Catalog item coercion (prevents 'str' has no attribute 'get' crashes)
# ─────────────────────────────────────────────────────────────────────────────
def _as_item(x: Any) -> Dict[str,str]:
    """Coerce an element into {'name','benefit'}."""
    if isinstance(x, dict):
        return {
            "name": str(x.get("name","")).strip(),
            "benefit": str(x.get("benefit","")).strip()
        }
    if isinstance(x, str):
        parts = re.split(r"\s+—\s+|\s+-\s+", x, maxsplit=1)
        name = parts[0].strip() if parts else ""
        benefit = parts[1].strip() if len(parts) > 1 else ""
        return {"name": name, "benefit": benefit}
    return {"name": "", "benefit": ""}

def _coerce_section(arr: Any) -> List[Dict[str,str]]:
    out: List[Dict[str,str]] = []
    seen = set()
    for x in (arr or []):
        item = _as_item(x)
        nm = item["name"]
        if not nm:
            continue
        key = nm.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out

def _normalize_catalog_result(result: Optional[Dict[str, Any]]) -> Dict[str, List[Dict[str,str]]]:
    result = result or {}
    keys = ("tires","attachments","options","telemetry")
    return {k: _coerce_section(result.get(k)) for k in keys}

# ─────────────────────────────────────────────────────────────────────────────
# Excel loading
# ─────────────────────────────────────────────────────────────────────────────
def _read_excel_rows(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        log.info("[ai_logic] catalog not found: %s (exists=False)", path)
        return []
    if _pd is None:
        # ultra-simple csv-ish fallback: try to read as TSV/CSV with naive split
        rows: List[Dict[str,Any]] = []
        if path.lower().endswith(".csv"):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                hdr = None
                for line in f:
                    parts = [p.strip() for p in line.rstrip("\n").split(",")]
                    if hdr is None:
                        hdr = parts
                        continue
                    rows.append({hdr[i]: parts[i] if i < len(parts) else "" for i in range(len(hdr))})
            return rows
        log.warning("[ai_logic] pandas not available; cannot read Excel %s", path)
        return []
    try:
        df = _pd.read_excel(path, engine="openpyxl")
        df = df.fillna("")
        return df.to_dict(orient="records")
    except Exception as e:
        log.exception("Failed reading Excel %s: %s", path, e)
        return []

def _bucketize_rows(rows: List[Dict[str,Any]]) -> Tuple[Dict[str,str], Dict[str,str], Dict[str,str]]:
    """
    Returns (options, attachments, tires) dictionaries mapping name -> benefit.
    Uses 'Type' or 'Category' to split; also respects optional 'Subcategory' (e.g., 'Tire').
    """
    options: Dict[str,str] = {}
    attachments: Dict[str,str] = {}
    tires: Dict[str,str] = {}

    for r in rows:
        name = _norm_spaces(str(r.get("Option") or r.get("Name") or r.get("Item") or r.get("Title") or ""))
        if not name:
            continue
        benefit = _norm_spaces(str(r.get("Benefit") or r.get("Description") or r.get("Desc") or ""))

        rtype = _lower(str(r.get("Type") or r.get("Category") or ""))
        subcat = _lower(str(r.get("Subcategory") or r.get("SubCategory") or ""))

        # If Subcategory == 'Tire' force tires bucket
        if "tire" in subcat:
            tires[name] = benefit
            continue

        if "attach" in rtype or any(k in _lower(name) for k in ("clamp","rotator","sideshift","positioner","boom","pole","ram","stabilizer","carton","bale","drum","block")):
            attachments[name] = benefit
        elif "tire" in rtype:
            tires[name] = benefit
        else:
            options[name] = benefit

    return options, attachments, tires

@lru_cache(maxsize=1)
def load_catalog_rows() -> List[Dict[str,Any]]:
    log.info("[ai_logic] Using catalog: %s (exists=%s)", _OPTIONS_XLSX, os.path.exists(_OPTIONS_XLSX))
    return _read_excel_rows(_OPTIONS_XLSX)

@lru_cache(maxsize=1)
def load_catalogs() -> Tuple[Dict[str,str], Dict[str,str], Dict[str,str]]:
    rows = load_catalog_rows()
    options, attachments, tires = _bucketize_rows(rows)
    log.info("[ai_logic] Loaded buckets: tires=%d attachments=%d options=%d", len(tires), len(attachments), len(options))
    return options, attachments, tires

def refresh_catalog_caches() -> None:
    load_catalog_rows.cache_clear()
    load_catalogs.cache_clear()
    log.info("[ai_logic] Catalog caches refreshed")

# Back-compat shims some callers import
def load_options() -> Dict[str,str]:
    return load_catalogs()[0]

def load_attachments() -> Dict[str,str]:
    return load_catalogs()[1]

def load_tires_as_options() -> Dict[str,str]:
    return load_catalogs()[2]

def load_tires() -> Dict[str,str]:
    return load_catalogs()[2]

def options_lookup_by_name(name: str) -> Optional[str]:
    opt = load_options()
    return opt.get(name)

def option_benefit(name: str) -> str:
    return options_lookup_by_name(name) or ""

# ─────────────────────────────────────────────────────────────────────────────
# Intent helpers & scoring
# ─────────────────────────────────────────────────────────────────────────────
_TIRES_PAT    = re.compile(r"\b(tires?|tyres?|tire\s*types?)\b", re.I)
_ATTACH_PAT   = re.compile(r"\b(attach(ment)?s?)\b", re.I)
_OPTIONS_PAT  = re.compile(r"\b(option|options)\b", re.I)
_TELEM_PAT    = re.compile(r"\b(fics|fleet\s*management|telemetry|portal)\b", re.I)
_ATTACH_HINT  = re.compile(r"\b(attach(ment)?|clamp|sideshift|positioner|fork|boom|pole|ram|push\s*/?\s*pull|slip[-\s]?sheet|paper\s*roll)\b", re.I)

def _wants_sections(q: str) -> Dict[str,bool]:
    t = q or ""
    return {
        "tires": bool(_TIRES_PAT.search(t)),
        "attachments": bool(_ATTACH_PAT.search(t)),
        "options": bool(_OPTIONS_PAT.search(t)),
        "telemetry": bool(_TELEM_PAT.search(t)),
        "any": any([
            _TIRES_PAT.search(t), _ATTACH_PAT.search(t), _OPTIONS_PAT.search(t), _TELEM_PAT.search(t)
        ])
    }

def _env_flags(q: str) -> Dict[str,bool]:
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

def _kw_score(ql: str, name: str, benefit: str, env: Dict[str,bool]) -> float:
    text = _lower(name + " " + (benefit or ""))
    s = 0.01  # baseline

    # generic alignments
    for w in ("indoor","warehouse","outdoor","yard","dust","debris","visibility",
              "lighting","safety","cold","freezer","rain","snow","cab","comfort",
              "vibration","filtration","radiator","screen","pre air cleaner",
              "dual air filter","heater","wiper","windshield","work light","led"):
        if w in ql and w in text:
            s += 0.7

    # environment signals
    if env["cold"] and any(k in text for k in ("cab","heater","defrost","wiper","rain-proof","glass","windshield","work light","led")):
        s += 2.0
    if env["cold"] and ("air conditioner" in text or "a/c" in text):
        s -= 2.0
    if env["dark"] and any(k in text for k in ("light","led","beacon","blue light","work light")):
        s += 1.6
    if env["indoor"]:
        if "sideshifter" in text or "side shifter" in text:
            s += 1.6
        if "fork positioner" in text:
            s += 1.4
    if any(k in ql for k in ("debris","yard","gravel","dirty","recycling","foundry","sawmill")) \
       and any(k in text for k in ("radiator","screen","pre air cleaner","dual air filter","filtration","belly pan","protection")):
        s += 1.3
    if _TELEM_PAT.search(ql) and _TELEM_PAT.search(text):
        s += 2.2
    return s

def _rank_bucket(user_q: str, bucket: Dict[str,str], limit: Optional[int]) -> List[Dict[str,str]]:
    if not bucket:
        return []
    ql = (user_q or "").lower()
    env = _env_flags(user_q)
    scored: List[Tuple[float, Dict[str,str]]] = []
    for name, benefit in bucket.items():
        s = _kw_score(ql, name, benefit, env)
        scored.append((s, {"name": name, "benefit": benefit}))
    scored.sort(key=lambda t: t[0], reverse=True)
    ranked = [row for _, row in scored]
    if isinstance(limit, int) and limit > 0:
        return ranked[:limit]
    return ranked

def _prioritize_lighting(items: List[Dict[str,str]], q_lower: str) -> List[Dict[str,str]]:
    """If the query is about dark/low light, float lighting to the top."""
    if not any(k in q_lower for k in ("dark","dim","night","poor lighting","low light")):
        return items
    def is_light(x: Dict[str,str]) -> bool:
        t = (x.get("name","") + " " + x.get("benefit","")).lower()
        return any(w in t for w in ("light","led","beacon","work light","blue light","rear working light"))
    lights   = [x for x in items if is_light(x)]
    nonlight = [x for x in items if not is_light(x)]
    return lights + nonlight

def _drop_ac_when_cold(items: List[Dict[str,str]], q_lower: str) -> List[Dict[str,str]]:
    """In cold-only contexts, remove A/C so heater/cab/wipers rise."""
    if not any(k in q_lower for k in ("cold","freezer","subzero","winter")):
        return items
    return [x for x in items if "air conditioner" not in (x.get("name","") + " " + x.get("benefit","")).lower()]

# ─────────────────────────────────────────────────────────────────────────────
# Public: intent + selector + renderers
# ─────────────────────────────────────────────────────────────────────────────
def parse_catalog_intent(user_q: str) -> Dict[str, Any]:
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

def recommend_options_from_sheet(user_q: str, limit: int = 6) -> Dict[str, List[Dict[str,str]]]:
    """
    Scenario-aware selector. Shows ONLY the sections the user requested
    (unless nothing requested, then returns a sensible default mix).
    Cold: boost cab/heater/defrost/lights; hide A/C.
    Indoor: prefer Sideshifter/Fork Positioner; suppress clamps unless mentioned.
    """
    wants = _wants_sections(user_q)
    env   = _env_flags(user_q)
    ql    = (user_q or "").lower()

    options, attachments, tires = load_catalogs()
    telemetry = {n: b for n, b in options.items() if _TELEM_PAT.search((n + " " + (b or "")).lower())}

    result: Dict[str, List[Dict[str,str]]] = {}

    # If the user explicitly asked, show only those sections
    if wants["any"]:
        if wants["tires"]:
            ranked_tires = _rank_bucket(user_q, tires, limit=None)
            if env["asks_non_mark"]:
                ranked_tires = [t for t in ranked_tires if "non-mark" in (t["name"] + " " + t.get("benefit","")).lower()] or ranked_tires
            result["tires"] = ranked_tires[:limit] if limit else ranked_tires

        if wants["attachments"]:
            ranked_atts = _rank_bucket(user_q, attachments, limit=None)
            if env["indoor"] and not env["mentions_clamp"]:
                ranked_atts = [a for a in ranked_atts if not re.search(r"\bclamp\b", a["name"].lower())]
                ranked_atts.sort(key=lambda a: int(("sideshifter" in a["name"].lower()) or ("fork positioner" in a["name"].lower())), reverse=True)
            result["attachments"] = ranked_atts[:limit] if limit else ranked_atts

        if wants["options"]:
            ranked_opts = _rank_bucket(user_q, options, limit=None)
            ranked_opts = _drop_ac_when_cold(ranked_opts, ql)
            ranked_opts = _prioritize_lighting(ranked_opts, ql)
            result["options"] = ranked_opts[:limit] if limit else ranked_opts

        if wants["telemetry"]:
            result["telemetry"] = _rank_bucket(user_q, telemetry, limit=limit)

        return _normalize_catalog_result(result)

    # Broad question (no explicit section words): default to options + a few atts
    ranked_opts = _rank_bucket(user_q, options, limit=None)
    ranked_atts = _rank_bucket(user_q, attachments, limit=None)

    # Indoor default: prefer alignment, cut random clamps if not asked
    if env["indoor"] and not env["mentions_clamp"]:
        ranked_atts = [a for a in ranked_atts if not re.search(r"\bclamp\b", a["name"].lower())]
        ranked_atts.sort(key=lambda a: int(("sideshifter" in a["name"].lower()) or ("fork positioner" in a["name"].lower())), reverse=True)

    ranked_opts = _drop_ac_when_cold(ranked_opts, ql)
    ranked_opts = _prioritize_lighting(ranked_opts, ql)

    result["options"] = ranked_opts[:limit]
    result["attachments"] = ranked_atts[:4]

    # Only add tires/telemetry if hinted
    if _TIRES_PAT.search(user_q or ""):
        result["tires"] = _rank_bucket(user_q, tires, limit=4)
    if _TELEM_PAT.search(user_q or ""):
        result["telemetry"] = _rank_bucket(user_q, telemetry, limit=3)

    return _normalize_catalog_result(result)

def render_sections_markdown(result: Dict[str, Any]) -> str:
    """Render only non-empty sections; accepts dicts or strings per item."""
    data = _normalize_catalog_result(result)
    order  = ["tires","attachments","options","telemetry"]
    labels = {"tires":"Tires","attachments":"Attachments","options":"Options","telemetry":"Telemetry"}
    lines: List[str] = []
    for key in order:
        arr = data.get(key, [])
        if not arr:
            continue
        lines.append(f"**{labels[key]}:**")
        for item in arr:
            name = item["name"]
            ben  = item["benefit"].replace("\n"," ").strip()
            lines.append(f"- {name}" + (f" — {ben}" if ben else ""))
    return "\n".join(lines) if lines else "(no matching items)"

def render_catalog_sections(result: Dict[str, Any], max_per_section: Optional[int] = None) -> str:
    """
    Safe fallback renderer used by callers that log:
    'Options/Attachments (fallback): ...'
    """
    data = _normalize_catalog_result(result)
    if isinstance(max_per_section, int) and max_per_section > 0:
        for k in data:
            data[k] = data[k][:max_per_section]
    return render_sections_markdown(data)

def list_all_from_excel() -> Dict[str, List[str]]:
    """Returns every item grouped by section for admin/debug screens."""
    options, attachments, tires = load_catalogs()
    return {
        "tires": sorted(list(tires.keys())),
        "attachments": sorted(list(attachments.keys())),
        "options": sorted(list(options.keys()))
    }

# ─────────────────────────────────────────────────────────────────────────────
# Forklift recommendation helpers (kept simple but stable)
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_models() -> List[Dict[str,Any]]:
    if not os.path.exists(_MODELS_JSON):
        log.info("[ai_logic] models.json not found at %s", _MODELS_JSON)
        return []
    try:
        with open(_MODELS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "models" in data:
                return data["models"]
            if isinstance(data, list):
                return data
            return []
    except Exception as e:
        log.exception("Failed reading models.json: %s", e)
        return []

@lru_cache(maxsize=1)
def _load_accounts() -> List[Dict[str,Any]]:
    if not os.path.exists(_ACCOUNTS_JSON):
        log.info("[ai_logic] accounts.json not found at %s", _ACCOUNTS_JSON)
        return []
    try:
        with open(_ACCOUNTS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "accounts" in data:
                return data["accounts"]
            if isinstance(data, list):
                return data
            return []
    except Exception as e:
        log.exception("Failed reading accounts.json: %s", e)
        return []

def model_meta_for(m: Dict[str,Any]) -> Dict[str,Any]:
    """Small wrapper to return normalized meta fields used in UIs."""
    return {
        "name": _safe_model_name(m),
        "capacity_lbs": _capacity_of(m),
        "power": _power_of(m),
        "aisle_in": _aisle_of(m),
        "lift_height_in": _height_of(m),
        "tire": _tire_of(m),
        "raw": m,
    }

def filter_models(allowed: Optional[List[str]] = None) -> List[Dict[str,Any]]:
    """Return all models, optionally filtering by list of codes/names."""
    models = _load_models()
    if not allowed:
        return models
    s = set(_lower(x) for x in allowed)
    out = []
    for m in models:
        nm = _lower(_safe_model_name(m))
        if nm in s:
            out.append(m)
    return out

def _score_model_for_q(m: Dict[str,Any], q: str) -> float:
    """Super lightweight text + capacity matching for a fallback recommender."""
    ql = _lower(q)
    cap = _capacity_of(m) or 0
    power = _power_of(m)
    tire = _tire_of(m)
    nm = _lower(_safe_model_name(m))

    s = 0.0
    # If question mentions 5,000 lb, 6000, etc., try to align
    cap_num = _num(ql)
    if cap_num:
        # prefer models within ±20%
        if cap and 0.8*cap_num <= cap <= 1.2*cap_num:
            s += 2.0
    # Indoor/outdoor heuristic
    if any(k in ql for k in ("indoor","warehouse")):
        if "cushion" in tire or ("electric" in power):
            s += 1.0
    if any(k in ql for k in ("outdoor","yard","rough","gravel","dock")):
        if "pneumatic" in tire:
            s += 1.0
    # reach/VNA keywords
    if any(k in ql for k in ("narrow aisle","reach","order picker","vna","turret")):
        if _is_reach_or_vna(m):
            s += 1.5
    # small text match with model name
    for w in ("lithium","diesel","lpg","lp","electric","4-wheel","3-wheel","pneumatic","cushion"):
        if w in ql and w in (nm + " " + power + " " + tire):
            s += 0.6

    return s

def select_models_for_question(user_q: str, k: int = 5) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]]]:
    """
    Returns (hits, allowed). 'allowed' is currently all models (stub kept for compatibility).
    Never returns zero values to avoid "not enough values to unpack" crashes.
    """
    models = _load_models()
    if not models:
        return ([], [])
    scored: List[Tuple[float, Dict[str,Any]]] = []
    for m in models:
        scored.append((_score_model_for_q(m, user_q), m))
    scored.sort(key=lambda t: t[0], reverse=True)
    hits = [m for _, m in scored[:max(k, 1)]]
    return (hits, models)

def top_pick_meta(m: Dict[str,Any]) -> Dict[str,Any]:
    mm = model_meta_for(m)
    return {
        "title": mm["name"],
        "bullets": [
            f"Capacity ~ {int(mm['capacity_lbs'])} lb" if mm["capacity_lbs"] else "Capacity: n/a",
            f"Power: {mm['power'] or 'n/a'}",
            f"Tires: {mm['tire'] or 'n/a'}",
            f"Lift Height: {mm['lift_height_in']} in" if mm["lift_height_in"] else "Lift Height: n/a",
        ]
    }

def allowed_models_block(allowed: List[Dict[str,Any]]) -> str:
    """Render a short 'allowed models' paragraph for UI."""
    if not allowed:
        return ""
    names = ", ".join(_safe_model_name(m) for m in allowed[:12])
    return f"Models considered: {names}"

def generate_forklift_context(hits: List[Dict[str,Any]]) -> str:
    """Small, readable context block about the top picks."""
    if not hits:
        return "No matching models found."
    lines = []
    for m in hits:
        mm = model_meta_for(m)
        line = f"- {mm['name']} — {int(mm['capacity_lbs'])} lb" if mm["capacity_lbs"] else f"- {mm['name']}"
        if mm["power"]:
            line += f", {mm['power']}"
        if mm["tire"]:
            line += f", {mm['tire']} tires"
        if mm["lift_height_in"]:
            line += f", {int(mm['lift_height_in'])} in lift"
        lines.append(line)
    return "\n".join(lines)

def debug_parse_and_rank(q: str) -> Dict[str, Any]:
    """Quick debug hook to inspect intent & top-ranked items."""
    wants = _wants_sections(q)
    env = _env_flags(q)
    options, attachments, tires = load_catalogs()
    return {
        "wants": wants,
        "env": env,
        "sample": {
            "options": _rank_bucket(q, options, limit=5),
            "attachments": _rank_bucket(q, attachments, limit=5),
            "tires": _rank_bucket(q, tires, limit=5),
        }
    }

# Legacy aliases some modules import
def generate_catalog_mode_response(user_q: str) -> str:
    """Simple wrapper used by older endpoints; returns markdown."""
    res = recommend_options_from_sheet(user_q, limit=6)
    return render_sections_markdown(res)

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

    # Model filtering & context (recommendation mode)
    "filter_models", "generate_forklift_context", "select_models_for_question",
    "allowed_models_block", "model_meta_for", "top_pick_meta",

    # Debug
    "debug_parse_and_rank",

    # Intentional small helpers imported elsewhere
    "_num_from_keys",
]

# Keep Pylance happy for legacy imports that expect these to exist by name
_LEGACY_EXPORTS: Dict[str, Any] = {
    "list_all_from_excel": list_all_from_excel,
    "num_from_keys": _num_from_keys,
}
if hashlib.md5(str(sorted(_LEGACY_EXPORTS.keys())).encode()).hexdigest():
    # no-op; prevents "defined but not accessed" warnings in some linters
    pass
