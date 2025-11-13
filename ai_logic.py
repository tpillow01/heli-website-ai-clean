"""
ai_logic.py
Single source of truth for:
- Catalog Q&A (Tires / Attachments / Options / Telemetry) from Excel
- Forklift Model Recommendation helpers used by heli_backup_ai.py
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Imports / logging / typing
# ─────────────────────────────────────────────────────────────────────────────
import os, json, re, logging, hashlib
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("ai_logic")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

try:
    import pandas as _pd  # Optional; file will still run without it
except Exception:
    _pd = None

# ─────────────────────────────────────────────────────────────────────────────
# Paths / constants
# ─────────────────────────────────────────────────────────────────────────────
_OPTIONS_XLSX = os.environ.get(
    "HELI_CATALOG_XLSX",
    os.path.join(os.path.dirname(__file__), "data", "forklift_options_benefits.xlsx")
)

_MODELS_JSON_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "data", "models.json"),
    os.path.join(os.path.dirname(__file__), "models.json"),
]

# Field tolerance for models.json variants
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

# Intent patterns
_TIRES_PAT   = re.compile(r"\b(tires?|tyres?|tire\s*types?)\b", re.I)
_ATTACH_PAT  = re.compile(r"\b(attach(ment)?s?)\b", re.I)
_OPTIONS_PAT = re.compile(r"\b(option|options)\b", re.I)
_TELEM_PAT   = re.compile(r"\b(fics|fleet\s*management|telemetry|portal)\b", re.I)

# ─────────────────────────────────────────────────────────────────────────────
# Small utils
# ─────────────────────────────────────────────────────────────────────────────
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
# Catalog loader (Excel) — robust to column name variations
# ─────────────────────────────────────────────────────────────────────────────
def _catalog_exists() -> bool:
    ok = os.path.exists(_OPTIONS_XLSX)
    log.info("Catalog path resolved: %s (exists=%s)", _OPTIONS_XLSX, ok)
    return ok

def _normalize_cols(cols: List[str]) -> Dict[str, str]:
    """Return mapping of lower->original for tolerant lookups."""
    mapping: Dict[str,str] = {}
    for c in cols:
        mapping[_lower(re.sub(r"\s+", " ", c))] = c
    return mapping

@lru_cache(maxsize=1)
def _load_catalog_df() -> Optional[Any]:
    if not _catalog_exists():
        return None
    if _pd is None:
        log.warning("pandas not available; Excel catalog disabled")
        return None
    try:
        df = _pd.read_excel(_OPTIONS_XLSX)
        return df
    except Exception as e:
        log.warning("Excel load failed: %s", e)
        return None

def refresh_catalog_caches() -> None:
    """Hot reload after Excel changes."""
    try:
        _load_catalog_df.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        pass

def _rows_from_df() -> List[Dict[str, Any]]:
    df = _load_catalog_df()
    if df is None:
        return []
    # Normalize column names
    cols_map = _normalize_cols(list(df.columns))
    def col(name_variants: List[str]) -> Optional[str]:
        for v in name_variants:
            key = _lower(v)
            if key in cols_map:
                return cols_map[key]
        return None

    col_option = col(["Option"])
    col_benefit = col(["Benefit"])
    col_type = col(["Type"])
    col_subcat = col(["Subcategory","Sub-category","Sub category"])
    out: List[Dict[str,Any]] = []
    for _, r in df.iterrows():
        name = str(r.get(col_option, "")).strip()
        ben  = str(r.get(col_benefit, "")).strip()
        typ  = str(r.get(col_type, "")).strip()
        sub  = str(r.get(col_subcat, "")).strip()
        if not name:
            continue
        out.append({"name": name, "benefit": ben, "type": typ, "subcategory": sub})
    return out

def load_catalog_rows() -> List[Dict[str,Any]]:
    return _rows_from_df()

def _bucket(rows: List[Dict[str,Any]], kind: str) -> Dict[str,str]:
    out: Dict[str,str] = {}
    for r in rows:
        if _lower(r.get("type")) == _lower(kind):
            out[r["name"]] = r.get("benefit","")
    return out

def load_options() -> Dict[str,str]:
    return _bucket(load_catalog_rows(), "Options")

def load_attachments() -> Dict[str,str]:
    return _bucket(load_catalog_rows(), "Attachments")

def load_tires() -> Dict[str,str]:
    return _bucket(load_catalog_rows(), "Tires")

def load_tires_as_options() -> Dict[str,str]:
    """For callers that unify buckets."""
    return load_tires()

def load_catalogs() -> Tuple[Dict[str,str], Dict[str,str], Dict[str,str]]:
    """Return (options, attachments, tires)."""
    return load_options(), load_attachments(), load_tires()

def options_lookup_by_name(name: str) -> Optional[Dict[str,str]]:
    nm = (name or "").strip().lower()
    rows = load_catalog_rows()
    for r in rows:
        if r["name"].strip().lower() == nm:
            return r
    return None

def option_benefit(name: str) -> str:
    r = options_lookup_by_name(name)
    return r.get("benefit","") if r else ""

# ─────────────────────────────────────────────────────────────────────────────
# Intent helpers for catalog mode
# ─────────────────────────────────────────────────────────────────────────────
def _wants_sections(q: str) -> Dict[str, bool]:
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

# ─────────────────────────────────────────────────────────────────────────────
# Catalog ranking & rendering
# ─────────────────────────────────────────────────────────────────────────────
def _kw_score(q: str, name: str, benefit: str) -> float:
    ql = _lower(q)
    text = _lower(name + " " + (benefit or ""))
    score = 0.01

    for w in ("indoor","warehouse","outdoor","yard","dust","debris","visibility",
              "lighting","safety","cold","freezer","rain","snow","cab","comfort",
              "vibration","filtration","cooling","radiator","screen","pre air cleaner",
              "dual air filter","non-mark","pneumatic","cushion","heater","wiper"):
        if w in ql and w in text:
            score += 0.7

    if any(k in ql for k in ("cold","freezer","subzero","winter")):
        if any(k in text for k in ("cab","heater","defrost","wiper","rain-proof","glass","windshield","work light","led")):
            score += 2.0
        if "air conditioner" in text or "a/c" in text:
            score -= 2.0
        if any(k in ql for k in ("dark","dim","poor lighting","night")) and any(k in text for k in ("light","led","beacon","work light")):
            score += 1.2

    if any(k in ql for k in ("dust","debris","recycling","sawmill","dirty","foundry","yard","gravel")):
        if any(k in text for k in ("radiator","screen","pre air cleaner","dual air filter","filtration","belly pan","protection")):
            score += 1.3

    if _TELEM_PAT.search(ql) and _TELEM_PAT.search(text):
        score += 2.0

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

def _prioritize_lighting(items: List[Dict[str,str]], q_lower: str) -> List[Dict[str,str]]:
    if not any(k in q_lower for k in ("dark","dim","night","poor lighting","low light")):
        return items
    def _is_light(x: Dict[str,str]) -> bool:
        t = (x.get("name","") + " " + x.get("benefit","")).lower()
        return any(w in t for w in ("light","led","beacon","work light","blue light","rear working light"))
    lights   = [x for x in items if _is_light(x)]
    nonlight = [x for x in items if not _is_light(x)]
    return lights + nonlight

def _drop_ac_when_cold(items: List[Dict[str,str]], q_lower: str) -> List[Dict[str,str]]:
    if not any(k in q_lower for k in ("cold","freezer","subzero","winter")):
        return items
    return [x for x in items if "air conditioner" not in (x.get("name","") + " " + x.get("benefit","")).lower()]

def recommend_options_from_sheet(user_q: str, limit: int = 6) -> Dict[str, List[Dict[str,str]]]:
    """
    Scenario-aware selector. Shows ONLY the sections the user requested
    (unless nothing is requested, then returns a default mix).
    """
    wants = _wants_sections(user_q)
    env   = _env_flags(user_q)
    options, attachments, tires = load_catalogs()
    telemetry = {n:b for n,b in options.items() if _TELEM_PAT.search((n + " " + b).lower())}
    ql = (user_q or "").lower()

    result: Dict[str, List[Dict[str,str]]] = {"attachments": [], "options": [], "telemetry": [], "tires": []}

    if wants["any"]:
        if wants["tires"]:
            ranked = _rank_bucket(user_q, tires, 0)
            if env["asks_non_mark"]:
                filtered = [t for t in ranked if "non-mark" in (t["name"] + " " + t.get("benefit","")).lower()]
                ranked = filtered or ranked
            result["tires"] = ranked[:limit] if limit else ranked

        if wants["attachments"]:
            atts = _rank_bucket(user_q, attachments, 0)
            if env["indoor"] and not env["mentions_clamp"]:
                atts = [a for a in atts if not re.search(r"\bclamp\b", a["name"].lower())]
            if env["indoor"]:
                atts.sort(key=lambda a: int(("sideshifter" in a["name"].lower()) or ("fork positioner" in a["name"].lower())), reverse=True)
            result["attachments"] = atts[:limit] if limit else atts

        if wants["options"]:
            opts = _rank_bucket(user_q, options, 0)
            opts = _drop_ac_when_cold(opts, ql)
            opts = _prioritize_lighting(opts, ql)
            result["options"] = opts[:limit] if limit else opts

        if wants["telemetry"]:
            result["telemetry"] = _rank_bucket(user_q, telemetry, limit or 6)
        return result

    # Broad question: produce a sensible default (no tires unless hinted)
    opts = _rank_bucket(user_q, options, 0)
    opts = _drop_ac_when_cold(opts, ql)
    opts = _prioritize_lighting(opts, ql)
    atts = _rank_bucket(user_q, attachments, 0)
    if env["indoor"] and not env["mentions_clamp"]:
        atts = [a for a in atts if not re.search(r"\bclamp\b", a["name"].lower())]
        atts.sort(key=lambda a: int(("sideshifter" in a["name"].lower()) or ("fork positioner" in a["name"].lower())), reverse=True)

    result["options"] = opts[: (limit or 6)]
    result["attachments"] = atts[: 4]

    if _TIRES_PAT.search(user_q or ""):
        result["tires"] = _rank_bucket(user_q, tires, 4)
    if _TELEM_PAT.search(user_q or ""):
        result["telemetry"] = _rank_bucket(user_q, telemetry, 3)
    return result

def render_sections_markdown(result: Dict[str, List[Dict[str,str]]]) -> str:
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
            if not isinstance(item, dict):
                # defensive guard against accidental string entries
                continue
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

def parse_catalog_intent(user_q: str) -> Dict[str, Any]:
    t = (user_q or "").strip().lower()
    which: Optional[str] = None
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

def list_all_from_excel(kind: str) -> List[Dict[str,str]]:
    """Utility for callers that want a raw list from the sheet."""
    kind_l = (kind or "").lower()
    if kind_l.startswith("att"):
        b = load_attachments()
    elif kind_l.startswith("opt"):
        b = load_options()
    elif kind_l.startswith("tir"):
        b = load_tires()
    else:
        b = {}
    return [{"name": k, "benefit": v} for k,v in b.items()]

def render_catalog_sections(user_text: str, max_per_section: int = 6) -> str:
    """
    Reactive entrypoint used by options_attachments_router.py.
    - If user asks to list/show all X, dump the whole category from Excel.
    - Otherwise, return scenario-aware picks for the requested sections only.
    """
    q = (user_text or "").strip()
    intent = parse_catalog_intent(q)
    which = intent["which"]
    if intent["list_all"] and which in ("tires","attachments","options","telemetry"):
        # Telemetry is a subset of options
        if which == "telemetry":
            options = load_options()
            telem = {n:b for n,b in options.items() if _TELEM_PAT.search((n + " " + b).lower())}
            rows = [{"name": k, "benefit": v} for k,v in telem.items()]
            return render_sections_markdown({"telemetry": rows})
        rows = list_all_from_excel(which)
        return render_sections_markdown({which: rows})

    picks = recommend_options_from_sheet(q, limit=max_per_section)
    # Only include sections that have items (function already does so)
    return render_sections_markdown(picks)

def generate_catalog_mode_response(user_text: str) -> str:
    """Alias for older callers."""
    return render_catalog_sections(user_text, max_per_section=6)

# ─────────────────────────────────────────────────────────────────────────────
# Forklift Recommendation Core (signature compatible with heli_backup_ai.py)
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_models_cache() -> List[Dict[str, Any]]:
    for p in _MODELS_JSON_CANDIDATES:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "models" in data:
                    return list(data["models"])
                if isinstance(data, list):
                    return data
            except Exception as e:
                log.warning("models.json load error at %s: %s", p, e)
    log.warning("models.json not found; returning empty list")
    return []

def model_meta_for(m: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": _safe_model_name(m),
        "capacity_lbs": _capacity_of(m),
        "lift_height_in": _height_of(m),
        "aisle_in": _aisle_of(m),
        "power": _power_of(m),
        "tire": _tire_of(m),
        "raw": m,
    }

def allowed_models_block(models: List[Dict[str, Any]]) -> str:
    if not models:
        return "_No candidate models found._"
    lines: List[str] = []
    for m in models:
        meta = model_meta_for(m)
        cap   = f'{int(meta["capacity_lbs"]):,} lb' if meta.get("capacity_lbs") else "—"
        ht    = f'{int(meta["lift_height_in"])} in' if meta.get("lift_height_in") else "—"
        aisle = f'{int(meta["aisle_in"])} in'      if meta.get("aisle_in") else "—"
        pwr   = (meta.get("power") or "—").title()
        tire  = (meta.get("tire") or "—").title()
        lines.append(f'- **{meta["name"]}** — {cap}, Lift {ht}, Aisle {aisle}, Power {pwr}, Tires {tire}')
    return "\n".join(lines)

def filter_models(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Hook for downstream hard filters; currently passthrough."""
    return models

def _soft_contains(hay: str, needles: List[str]) -> bool:
    h = (hay or "").lower()
    return any((n or "").lower() in h for n in needles if n)

def _model_score(user_q: str, row: Dict[str, Any]) -> float:
    ql = (user_q or "").lower()
    score = 0.0
    cap   = _capacity_of(row) or 0.0
    h_in  = _height_of(row) or 0.0
    aisle = _aisle_of(row) or 0.0
    pwr   = _power_of(row) or ""
    tire  = _tire_of(row) or ""

    # Capacity hints
    if any(k in ql for k in ("lb","lbs","pound","ton","capacity","lift")):
        score += min(cap / 10000.0, 2.0)

    # Aisle / tight
    if any(k in ql for k in ("aisle","tight","narrow","vna","right-angle")):
        score += (1.0 if aisle and aisle < 100 else 0.0)

    # Height
    if any(k in ql for k in ("height","stack","racking","high")):
        score += min(h_in / 300.0, 1.5)

    # Power/fuel matches
    if _soft_contains(pwr, ["electric","lithium","diesel","lpg","gas"]):
        if _soft_contains(ql, ["electric","lithium"]) and "electric" in pwr:
            score += 1.0
        if _soft_contains(ql, ["diesel"]) and "diesel" in pwr:
            score += 1.0
        if _soft_contains(ql, ["lpg","gas","propane"]) and ("lpg" in pwr or "gas" in pwr or "propane" in pwr):
            score += 1.0

    # Tires (indoor/outdoor)
    if any(k in ql for k in ("indoor","warehouse","epoxy","polished")):
        if "cushion" in tire:
            score += 0.8
    if any(k in ql for k in ("outdoor","yard","gravel","rough","debris")):
        if "pneumatic" in tire or "super elastic" in tire:
            score += 1.0

    # Reach/VNA preference in tight aisles
    name_text = (_lower(row.get("Model","")) + " " + _lower(row.get("Model Name","")))
    if any(k in ql for k in ("reach","order picker","vna","turret")):
        if any(w in name_text for w in ("reach","rq","vna","order","turret")):
            score += 2.0

    return score

def select_models_for_question(user_q: str, k: int = 5) -> Tuple[List[Dict[str, Any]], str]:
    """
    Returns (hits, allowed_block_markdown).
    """
    models = filter_models(_load_models_cache())
    if not models:
        return [], "_No candidate models found._"
    scored: List[Tuple[float, Dict[str,Any]]] = []
    for m in models:
        s = _model_score(user_q, m)
        scored.append((s, m))
    scored.sort(key=lambda t: t[0], reverse=True)
    hits = [m for _, m in scored[: max(1, k)]]
    return hits, allowed_models_block(hits)

def top_pick_meta(model: Dict[str, Any]) -> Dict[str, Any]:
    meta = model_meta_for(model)
    return {
        "name": meta["name"],
        "capacity_lbs": meta["capacity_lbs"],
        "lift_height_in": meta["lift_height_in"],
        "aisle_in": meta["aisle_in"],
        "power": meta["power"],
        "tire": meta["tire"],
    }

def generate_forklift_context(user_q: str, acct: Optional[Dict[str, Any]] = None) -> str:
    """
    Small prompt/context helper used by heli_backup_ai.py.
    You can enrich this with account/industry later.
    """
    acct_line = ""
    if acct and isinstance(acct, dict):
        name = str(acct.get("name") or acct.get("Sold to Name") or "").strip()
        if name:
            acct_line = f"\nAccount: {name}"
    return f"User question: {user_q.strip()}{acct_line}\nProvide a concise, practical forklift recommendation with reasons."

# Debug convenience for testing in REPL
def debug_parse_and_rank(q: str) -> str:
    picks = recommend_options_from_sheet(q, limit=6)
    return render_sections_markdown(picks)

# ─────────────────────────────────────────────────────────────────────────────
# Legacy-friendly exports (some code imports these symbols explicitly)
# ─────────────────────────────────────────────────────────────────────────────
def _is_attachment(name: str) -> bool:
    nl = _lower(name)
    return any(k in nl for k in (
        "clamp","sideshift","positioner","rotator","boom","pole","ram",
        "fork extension","extensions","push/ pull","push/pull",
        "slip-sheet","slipsheet","bale","carton","drum","bag push","load stabilizer"
    ))

def _num_from_keys_legacy(row: Dict[str,Any], keys: List[str]) -> Optional[float]:
    return _num_from_keys(row, keys)

# ─────────────────────────────────────────────────────────────────────────────
# __all__ for explicit imports elsewhere
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

    # Model filtering & context (forklift recommendation)
    "filter_models", "generate_forklift_context", "select_models_for_question",
    "allowed_models_block", "model_meta_for", "top_pick_meta",

    # Debug
    "debug_parse_and_rank",

    # Intentional small helpers sometimes imported
    "_num_from_keys",
    "_is_attachment",
]
