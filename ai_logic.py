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

# Keys used to read models.json (flexible sheets)
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
# Catalog loader (Excel -> tires / attachments / options) and intent routing
# ─────────────────────────────────────────────────────────────────────────────

# Patterns we’ll reuse
_TIRES_PAT   = re.compile(r"\b(tires?|tyres?|tire\s*types?)\b", re.I)
_ATTACH_PAT  = re.compile(r"\b(attach(ment)?s?)\b", re.I)
_OPTIONS_PAT = re.compile(r"\b(option|options)\b", re.I)
_TELEM_PAT   = re.compile(r"\b(fics|fleet\s*management|telemetry|portal)\b", re.I)

def _wants_sections(q: str) -> dict:
    t = (q or "")
    return {
        "tires": bool(_TIRES_PAT.search(t)),
        "attachments": bool(_ATTACH_PAT.search(t)),
        "options": bool(_OPTIONS_PAT.search(t)),
        "telemetry": bool(_TELEM_PAT.search(t)),
        "any": bool(_TIRES_PAT.search(t) or _ATTACH_PAT.search(t) or _OPTIONS_PAT.search(t) or _TELEM_PAT.search(t)),
    }

def _env_flags(q: str) -> dict:
    ql = (q or "").lower()
    return {
        "cold": any(k in ql for k in ("cold","freezer","subzero","winter")),
        "indoor": any(k in ql for k in ("indoor","warehouse","inside","epoxy","polished","concrete")),
        "outdoor": any(k in ql for k in ("outdoor","yard","rain","snow","dust","gravel","dirt")),
        "dark": any(k in ql for k in ("dark","dim","night","poor lighting","low light")),
        "mentions_clamp": bool(re.search(r"\bclamp|paper\s*roll|bale|drum|carton|block\b", ql)),
        "mentions_align": bool(re.search(r"\balign|tight\s*aisle|narrow|staging\b", ql)),
        "asks_non_mark": bool(re.search(r"non[-\s]?mark", ql)),
    }

@lru_cache(maxsize=1)
def load_catalog_rows() -> List[Dict[str, str]]:
    """
    Load rows from the Excel catalog (Option / Benefit / Type / Subcategory).
    Returns a list of dicts with normalized keys.
    """
    path = _OPTIONS_XLSX
    exists = os.path.exists(path)
    log.info("[ai_logic] Using catalog: %s (exists=%s)", path, bool(exists))
    rows: List[Dict[str, str]] = []
    if not exists:
        return rows

    if _pd is None:
        # Minimal CSV/TSV fallback (if someone dropped a CSV in its place)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                header = None
                for line in f:
                    parts = [p.strip() for p in line.strip().split(",")]
                    if not header:
                        header = parts
                        continue
                    row = {header[i]: parts[i] if i < len(parts) else "" for i in range(len(header))}
                    rows.append({
                        "Option": row.get("Option",""),
                        "Benefit": row.get("Benefit",""),
                        "Type": row.get("Type",""),
                        "Subcategory": row.get("Subcategory",""),
                    })
        except Exception as e:
            log.error("Catalog fallback load failed: %s", e)
        return rows

    try:
        df = _pd.read_excel(path)
        for _, r in df.iterrows():
            rows.append({
                "Option": str(r.get("Option", "")).strip(),
                "Benefit": str(r.get("Benefit", "")).strip(),
                "Type": str(r.get("Type", "")).strip(),
                "Subcategory": str(r.get("Subcategory", "")).strip(),
            })
    except Exception as e:
        log.error("Failed reading Excel: %s", e)
    return rows

@lru_cache(maxsize=1)
def load_catalogs() -> Tuple[Dict[str,str], Dict[str,str], Dict[str,str]]:
    """
    Returns (options_dict, attachments_dict, tires_dict) mapping name -> benefit
    """
    rows = load_catalog_rows()
    opts: Dict[str,str] = {}
    atts: Dict[str,str] = {}
    tires: Dict[str,str] = {}
    for r in rows:
        name = str(r.get("Option","")).strip()
        ben  = str(r.get("Benefit","")).strip()
        typ  = str(r.get("Type","")).strip().lower()
        if not name:
            continue
        if typ == "tires" or typ == "tire":
            tires[name] = ben
        elif typ == "attachments" or typ == "attachment":
            atts[name] = ben
        else:
            # Treat everything else as option (includes Telemetry rows)
            opts[name] = ben
    # Log counts for visibility
    log.info("[ai_logic] Loaded buckets: tires=%d attachments=%d options=%d", len(tires), len(atts), len(opts))
    return opts, atts, tires

def refresh_catalog_caches() -> None:
    """Clear caches so a fresh Excel upload is picked up."""
    load_catalog_rows.cache_clear()
    load_catalogs.cache_clear()

# Convenience wrappers kept for legacy imports
def load_options() -> Dict[str,str]:
    return load_catalogs()[0]

def load_attachments() -> Dict[str,str]:
    return load_catalogs()[1]

def load_tires_as_options() -> Dict[str,str]:
    return load_catalogs()[2]

def load_tires() -> Dict[str,str]:
    return load_catalogs()[2]

def options_lookup_by_name(name: str) -> Optional[Dict[str,str]]:
    n = (name or "").strip().lower()
    opts, atts, tires = load_catalogs()
    for src in (opts, atts, tires):
        for k, v in src.items():
            if k.strip().lower() == n:
                return {"name": k, "benefit": v}
    return None

def option_benefit(name: str) -> str:
    rec = options_lookup_by_name(name)
    return rec["benefit"] if rec else ""

# ─────────────────────────────────────────────────────────────────────────────
# Ranking helpers for options/attachments/tires
# ─────────────────────────────────────────────────────────────────────────────
def _kw_score(q: str, name: str, benefit: str) -> float:
    ql = _lower(q)
    text = _lower(name + " " + (benefit or ""))
    score = 0.01  # small baseline

    for w in ("indoor","warehouse","outdoor","yard","dust","debris","visibility",
              "lighting","safety","cold","freezer","rain","snow","cab","comfort",
              "vibration","filtration","cooling","radiator","screen","pre air cleaner",
              "dual air filter","non-mark","pneumatic","cushion","heater","wiper","windshield","work light","led"):
        if w in ql and w in text:
            score += 0.7

    # cold boosts; hide AC
    if any(k in ql for k in ("cold","freezer","subzero","winter")):
        if any(k in text for k in ("cab","heater","defrost","wiper","rain-proof","glass","windshield","work light","led")):
            score += 2.0
        if ("air conditioner" in text) or ("a/c" in text):
            score -= 2.0

    # dark boosts
    if any(k in ql for k in ("dark","dim","night","poor lighting","low light")):
        if any(k in text for k in ("light","led","beacon","blue light","work light")):
            score += 1.6

    # indoor alignment
    if "indoor" in ql or "warehouse" in ql:
        if ("sideshifter" in text) or ("side shifter" in text):
            score += 1.6
        if "fork positioner" in text:
            score += 1.4

    # telemetry direct ask
    if _TELEM_PAT.search(ql) and _TELEM_PAT.search(text):
        score += 2.2

    return score

def _rank_bucket(q: str, bucket: Dict[str,str], limit: int = 6) -> List[Dict[str,str]]:
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
# Public: Parse + recommend + render
# ─────────────────────────────────────────────────────────────────────────────
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

def recommend_options_from_sheet(user_q: str, limit: int = 6) -> dict:
    """
    Scenario-aware selector. Shows ONLY the sections the user requested
    (unless nothing requested, then returns a sensible default mix).
    Cold: boost cab/heater/defrost/lights; hide A/C.
    Indoor: prefer Sideshifter/Fork Positioner; suppress random clamps unless mentioned.
    """
    wants = _wants_sections(user_q)
    env   = _env_flags(user_q)

    options, attachments, tires = load_catalogs()

    # carve telemetry out of options by name/benefit text
    telemetry = {
        n: b for n, b in options.items()
        if _TELEM_PAT.search((n + " " + (b or "")).lower())
    }

    result: Dict[str, List[Dict[str,str]]] = {}

    def _post_filter(items: List[Dict[str,str]]) -> List[Dict[str,str]]:
        if not items:
            return items
        ql = (user_q or "").lower()

        # cold-only asks: remove A/C so heater/etc. rise
        if env["cold"]:
            items = [x for x in items if "air conditioner" not in (x.get("name","") + " " + x.get("benefit","")).lower()]

        # dark asks: float lighting to top
        if env["dark"]:
            def is_light(x: Dict[str,str]) -> bool:
                t = (x.get("name","") + " " + x.get("benefit","")).lower()
                return any(w in t for w in ("light","led","beacon","work light","blue light","rear working light"))
            lights   = [x for x in items if is_light(x)]
            nonlight = [x for x in items if not is_light(x)]
            items = lights + nonlight

        # indoor: suppress clamps unless mentioned
        if env["indoor"] and not env["mentions_clamp"]:
            items = [a for a in items if not re.search(r"\bclamp\b", (a.get("name","") or "").lower())]

        return items

    # If the user explicitly asked, show *only* those sections.
    if wants["any"]:
        if wants["tires"]:
            ranked_tires = _rank_bucket(user_q, tires, limit=limit if limit else 0)
            if env["asks_non_mark"]:
                ranked_tires = [t for t in ranked_tires if "non-mark" in (t["name"] + " " + t.get("benefit","")).lower()] or ranked_tires
            result["tires"] = ranked_tires

        if wants["attachments"]:
            ranked_atts = _post_filter(_rank_bucket(user_q, attachments, limit=0))
            result["attachments"] = ranked_atts[:limit] if limit else ranked_atts

        if wants["options"]:
            ranked_opts = _post_filter(_rank_bucket(user_q, options, limit=0))
            result["options"] = ranked_opts[:limit] if limit else ranked_opts

        if wants["telemetry"]:
            ranked_tel = _rank_bucket(user_q, telemetry, limit=limit if limit else 0)
            result["telemetry"] = ranked_tel

        return result

    # Broad question (no explicit section words): default to options + a few atts
    ranked_opts = _post_filter(_rank_bucket(user_q, options, limit=0))
    ranked_atts = _post_filter(_rank_bucket(user_q, attachments, limit=0))

    result["options"] = ranked_opts[: (limit if limit else 6)]
    result["attachments"] = ranked_atts[:4]

    # Only add tires/telemetry if hinted
    if _TIRES_PAT.search(user_q or ""):
        result["tires"] = _rank_bucket(user_q, tires, limit=4)
    if _TELEM_PAT.search(user_q or ""):
        result["telemetry"] = _rank_bucket(user_q, telemetry, limit=3)

    return result

def _coerce_item(x: Any) -> Dict[str, str]:
    """Turn plain strings into {'name','benefit'} dicts. Pass dicts through."""
    if isinstance(x, dict):
        # Ensure keys exist
        return {"name": str(x.get("name","")).strip(),
                "benefit": str(x.get("benefit","")).strip()}
    # Plain string -> look up benefit if we can
    name = str(x).strip()
    ben = option_benefit(name) if name else ""
    return {"name": name, "benefit": ben}

def _coerce_list(items: Any) -> List[Dict[str, str]]:
    """Ensure a list of dicts; tolerate None/singletons/strings."""
    if items is None:
        return []
    if isinstance(items, (str, dict)):
        return [_coerce_item(items)]
    try:
        return [_coerce_item(i) for i in items]
    except Exception:
        # Last-resort: stringify the entire object
        return [_coerce_item(str(items))]

def normalize_catalog_result(result: Any) -> Dict[str, List[Dict[str, str]]]:
    """Make sure every section maps to a list of {'name','benefit'} dicts."""
    out: Dict[str, List[Dict[str, str]]] = {}
    if not isinstance(result, dict):
        # If some caller passed a raw list, treat it as 'options'
        out["options"] = _coerce_list(result)
        return out
    for key in ("tires","attachments","options","telemetry"):
        out[key] = _coerce_list(result.get(key))
    # Drop empty sections for cleanliness
    return {k: v for k, v in out.items() if v}

def render_sections_markdown(result: dict) -> str:
    """
    Render only sections that have items. Skips empty/absent sections.
    Sections: 'tires', 'attachments', 'options', 'telemetry'
    """
    result = normalize_catalog_result(result)  # ← defensive: coerce strings -> dicts

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
            ben  = (item.get("benefit") or "").strip().replace("\n", " ")
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            section_lines.append(f"- {name}" + (f" — {ben}" if ben else ""))

        if section_lines:
            lines.append(f"**{labels[key]}:**")
            lines.extend(section_lines)

    return "\n".join(lines) if lines else "(no matching items)"

def render_catalog_sections(result: dict, **kwargs) -> str:
    """
    Backwards-compatible wrapper used by older routes.
    Accepts and ignores unexpected kwargs (e.g., max_per_section) safely.
    """
    return render_sections_markdown(result)

def render_catalog_sections(result: dict, **kwargs) -> str:
    """
    Backwards-compatible wrapper used by older routes.
    Accepts and ignores unexpected kwargs (e.g., max_per_section) safely.
    """
    return render_sections_markdown(result)

def list_all_from_excel(section: str) -> List[Dict[str,str]]:
    """
    Utility for debugging / listing. section ∈ {'tires','attachments','options','telemetry'}
    """
    options, attachments, tires = load_catalogs()
    if section == "tires":
        return [{"name": k, "benefit": v} for k, v in tires.items()]
    if section == "attachments":
        return [{"name": k, "benefit": v} for k, v in attachments.items()]
    if section == "options":
        return [{"name": k, "benefit": v} for k, v in options.items()]
    if section == "telemetry":
        telemetry = {
            n: b for n, b in options.items()
            if _TELEM_PAT.search((n + " " + (b or "")).lower())
        }
        return [{"name": k, "benefit": v} for k, v in telemetry.items()]
    return []

def generate_catalog_mode_response(user_q: str, limit: int = 6) -> str:
    picks = recommend_options_from_sheet(user_q, limit=limit)
    picks = normalize_catalog_result(picks)  # ← extra safety
    return render_sections_markdown(picks)

# ─────────────────────────────────────────────────────────────────────────────
# Models loading + simple recommendation flow
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_models_raw() -> List[Dict[str,Any]]:
    path = _MODELS_JSON
    models: List[Dict[str,Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                models = data
            elif isinstance(data, dict) and "models" in data:
                models = data["models"]
    except Exception as e:
        log.warning("Could not load models.json at %s: %s", path, e)
    return models

@lru_cache(maxsize=1)
def _load_accounts_raw() -> List[Dict[str,Any]]:
    path = _ACCOUNTS_JSON
    accounts: List[Dict[str,Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                accounts = data
            elif isinstance(data, dict) and "accounts" in data:
                accounts = data["accounts"]
    except Exception as e:
        log.warning("Could not load accounts.json at %s: %s", path, e)
    return accounts

def filter_models(user_q: str, k: int = 5) -> List[Dict[str,Any]]:
    """
    Lightweight keyword filter across models. Robust to missing fields.
    """
    ql = (user_q or "").lower()
    models = _load_models_raw()
    if not models:
        return []

    scored: List[Tuple[float, Dict[str,Any]]] = []
    for m in models:
        text = " ".join(str(m.get(x, "")) for x in ("Model","Model Name","Description","Class","Type","Power","Drive"))
        tl = text.lower()
        s = 0.0
        # coarse alignment
        if any(k in ql for k in ("rough", "outdoor", "yard")) and any(x in tl for x in ("pneumatic","diesel","lpg","4x4","rough")):
            s += 2.0
        if any(k in ql for k in ("indoor", "warehouse")) and any(x in tl for x in ("cushion","electric","three wheel","3-wheel")):
            s += 2.0
        cap = _capacity_of(m) or 0
        m_height = _height_of(m) or 0
        # crude capacity match
        want = _num(re.search(r"(\d{3,6})\s*(lb|lbs|pounds?)", ql) or "")
        if want and cap:
            # closer capacity => higher score
            s += max(0.0, 2.5 - abs(cap - want)/max(want,1) * 2.5)
        # aisle hints
        if "narrow" in ql or "tight" in ql:
            if "reach" in tl or "vna" in tl:
                s += 1.2
        # penalty if opposite environment word
        if "indoor" in ql and "pneumatic" in tl:
            s -= 0.5
        if "outdoor" in ql and "cushion" in tl:
            s -= 0.5

        scored.append((s, m))

    scored.sort(key=lambda t: t[0], reverse=True)
    return [m for _, m in scored[:max(1,k)]]

def select_models_for_question(user_q: str, k: int = 5) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]]]:
    """
    Returns (hits, allowed). `allowed` is just the same as hits for now, but this
    preserves your previous function signature to avoid unpack errors.
    """
    hits = filter_models(user_q, k=k)
    allowed = hits[:]  # placeholder for any future allowlist logic
    return hits, allowed

def model_meta_for(m: Dict[str,Any]) -> Dict[str,Any]:
    return {
        "name": _safe_model_name(m),
        "capacity_lbs": _capacity_of(m),
        "lift_height_in": _height_of(m),
        "aisle_in": _aisle_of(m),
        "power": _power_of(m),
        "tire": _tire_of(m),
        "class": (_lower(m.get("Class")) if m.get("Class") else ""),
    }

def top_pick_meta(hits: List[Dict[str,Any]]) -> Dict[str,Any]:
    if not hits:
        return {}
    return model_meta_for(hits[0])

def allowed_models_block(hits: List[Dict[str,Any]]) -> str:
    if not hits:
        return "_No matching models found._"
    lines = ["**Recommended Models:**"]
    for m in hits:
        meta = model_meta_for(m)
        bits = []
        if meta.get("capacity_lbs"): bits.append(f'{int(meta["capacity_lbs"]):,} lb')
        if meta.get("power"): bits.append(meta["power"])
        if meta.get("tire"): bits.append(meta["tire"])
        lines.append(f"- {meta['name']} ({', '.join(bits)})")
    return "\n".join(lines)

def generate_forklift_context(hits: List[Dict[str,Any]]) -> str:
    """
    Returns a short natural-language context block describing the top picks.
    """
    if not hits:
        return "No suitable models matched the request."
    lines = []
    for m in hits[:5]:
        meta = model_meta_for(m)
        line = f"{meta['name']}"
        extras = []
        if meta.get("capacity_lbs"): extras.append(f"{int(meta['capacity_lbs']):,} lb")
        if meta.get("lift_height_in"): extras.append(f"{int(meta['lift_height_in'])} in lift")
        if meta.get("power"): extras.append(meta["power"])
        if meta.get("tire"): extras.append(meta["tire"])
        if extras:
            line += " — " + ", ".join(extras)
        lines.append(line)
    return "\n".join(lines)

def debug_parse_and_rank(user_q: str) -> Dict[str,Any]:
    wants = _wants_sections(user_q)
    env = _env_flags(user_q)
    opts, atts, tires = load_catalogs()
    picks = recommend_options_from_sheet(user_q)
    return {
        "wants": wants,
        "env": env,
        "counts": {"options": len(opts), "attachments": len(atts), "tires": len(tires)},
        "picks": picks
    }

# ─────────────────────────────────────────────────────────────────────────────
# Exports (to keep older imports stable)
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
