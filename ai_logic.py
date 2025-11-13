"""
ai_logic.py — clean rebuild
Catalog helpers (tires / attachments / options), intent→picks, and
model recommendation utilities used across the Heli AI app.

Goals:
- Zero missing-symbol errors (Pylance clean).
- Safe fallbacks when XLSX/JSON is missing.
- Deterministic, keyword‑aware ranking for Options/Attachments/Tires.
- Returns ONLY the sections the user asked for (no noisy headers).
- Simple, reliable model recommendation for the /api/chat “recommendation” mode.
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
    import pandas as _pd  # optional
except Exception:
    _pd = None

# ─────────────────────────────────────────────────────────────────────────────
# Paths / constants
# ─────────────────────────────────────────────────────────────────────────────
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_OPTIONS_XLSX = os.environ.get("HELI_CATALOG_XLSX", os.path.join(_DATA_DIR, "forklift_options_benefits.xlsx"))
_MODELS_JSON  = os.environ.get("HELI_MODELS_JSON",  os.path.join(_DATA_DIR, "models.json"))
_ACCTS_JSON   = os.environ.get("HELI_ACCOUNTS_JSON", os.path.join(_DATA_DIR, "accounts.json"))

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
def _lower(s: Any) -> str:
    return str(s or "").lower()

def _norm_spaces(s: str) -> str:
    return " ".join((s or "").split())

def _num(s: Any) -> Optional[float]:
    if s is None:
        return None
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

# ─────────────────────────────────────────────────────────────────────────────
# Model field normalizers
# ─────────────────────────────────────────────────────────────────────────────
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

def _power_of(row: Dict[str,Any]) -> str:
    return _text_from_keys(row, POWER_KEYS).lower()

# ─────────────────────────────────────────────────────────────────────────────
# Catalog loading (Options / Attachments / Tires)
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def load_catalog_rows() -> List[Dict[str, str]]:
    rows: List[Dict[str,str]] = []
    if _pd is None or not os.path.exists(_OPTIONS_XLSX):
        log.info("[ai_logic] Using catalog: %s (exists=%s)", _OPTIONS_XLSX, os.path.exists(_OPTIONS_XLSX))
        return rows
    try:
        df = _pd.read_excel(_OPTIONS_XLSX)
        for _, r in df.fillna("").iterrows():
            rows.append({
                "Option": str(r.get("Option", "")).strip(),
                "Benefit": str(r.get("Benefit", "")).strip(),
                "Type": str(r.get("Type", "")).strip(),
                "Subcategory": str(r.get("Subcategory", "")).strip(),
            })
    except Exception as e:
        log.exception("[ai_logic] Failed to read catalog XLSX: %s", e)
    return rows

@lru_cache(maxsize=1)
def load_options() -> Dict[str, str]:
    out: Dict[str,str] = {}
    for r in load_catalog_rows():
        if (r.get("Type") or "").lower() in ("options","option"):
            name = r.get("Option") or ""
            if name:
                out[name] = r.get("Benefit", "")
    return out

@lru_cache(maxsize=1)
def load_attachments() -> Dict[str, str]:
    out: Dict[str,str] = {}
    for r in load_catalog_rows():
        if (r.get("Type") or "").lower() in ("attachments","attachment"):
            name = r.get("Option") or ""
            if name:
                out[name] = r.get("Benefit", "")
    return out

@lru_cache(maxsize=1)
def load_tires() -> Dict[str, str]:
    out: Dict[str,str] = {}
    for r in load_catalog_rows():
        t = (r.get("Type") or "").lower()
        sub = (r.get("Subcategory") or "").lower()
        if t == "tires" or sub == "tire" or sub == "tires":
            name = r.get("Option") or ""
            if name:
                out[name] = r.get("Benefit", "")
    return out

@lru_cache(maxsize=1)
def load_tires_as_options() -> Dict[str, str]:
    return load_tires().copy()

@lru_cache(maxsize=1)
def load_catalogs() -> Tuple[Dict[str,str], Dict[str,str], Dict[str,str]]:
    opts, atts, tires = load_options(), load_attachments(), load_tires()
    log.info("[ai_logic] Loaded buckets: tires=%d attachments=%d options=%d", len(tires), len(atts), len(opts))
    return opts, atts, tires

# quick lookups
def options_lookup_by_name(name: str) -> Optional[str]:
    bucket = {**load_options(), **load_attachments(), **load_tires()}
    return bucket.get(name)

def option_benefit(name: str) -> str:
    return options_lookup_by_name(name) or ""

# ─────────────────────────────────────────────────────────────────────────────
# Intent helpers for catalog Q&A
# ─────────────────────────────────────────────────────────────────────────────
_TIRES_PAT    = re.compile(r"\b(tires?|tyres?|tire\s*types?)\b", re.I)
_ATTACH_PAT   = re.compile(r"\b(attach(ment)?s?)\b", re.I)
_OPTIONS_PAT  = re.compile(r"\b(options?|option)\b", re.I)
_TELEM_PAT    = re.compile(r"\b(fics|fleet\s*management|telemetry|portal)\b", re.I)
_ATTACH_HINT  = re.compile(r"\b(attach(ment)?|clamp|sideshift|positioner|fork|boom|pole|ram|push\s*/?\s*pull|slip[-\s]?sheet|paper\s*roll)\b", re.I)


def _wants_sections(q: str) -> Dict[str, bool]:
    t = q or ""
    return {
        "tires": bool(_TIRES_PAT.search(t)),
        "attachments": bool(_ATTACH_PAT.search(t)),
        "options": bool(_OPTIONS_PAT.search(t)),
        "telemetry": bool(_TELEM_PAT.search(t)),
        "any": any([
            _TIRES_PAT.search(t), _ATTACH_PAT.search(t), _OPTIONS_PAT.search(t), _TELEM_PAT.search(t)
        ]),
    }


def _env_flags(q: str) -> Dict[str, bool]:
    ql = (q or "").lower()
    return {
        "cold": any(k in ql for k in ("cold","freezer","subzero","winter")),
        "indoor": any(k in ql for k in ("indoor","warehouse","inside","epoxy","polished","concrete")),
        "dark": any(k in ql for k in ("dark","dim","night","poor lighting","low light")),
        "mentions_clamp": bool(re.search(r"\bclamp|paper\s*roll|bale|drum|carton|block\b", ql)),
        "mentions_align": bool(re.search(r"\balign|tight\s*aisle|narrow|staging\b", ql)),
        "asks_non_mark": bool(re.search(r"non[-\s]?mark", ql)),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Catalog ranking & selection
# ─────────────────────────────────────────────────────────────────────────────
def _prioritize_lighting(items: List[Dict[str,Any]], q_lower: str) -> List[Dict[str,Any]]:
    """If the query is about dark/low light, float lighting to the top."""
    if not any(k in q_lower for k in ("dark","dim","night","poor lighting","low light")):
        return items
    def is_light(x: Dict[str,Any]) -> bool:
        t = (x.get("name","") + " " + x.get("benefit",""))
        tl = t.lower()
        return any(w in tl for w in ("light","led","beacon","work light","blue light","rear working light"))
    lights   = [x for x in items if is_light(x)]
    nonlight = [x for x in items if not is_light(x)]
    return lights + nonlight


def _drop_ac_when_cold(items: List[Dict[str,Any]], q_lower: str) -> List[Dict[str,Any]]:
    """In cold-only contexts, remove A/C so heater/cab/wipers rise."""
    if not any(k in q_lower for k in ("cold","freezer","subzero","winter")):
        return items
    out: List[Dict[str,Any]] = []
    for x in items:
        t = (x.get("name","") + " " + x.get("benefit",""))
        if "air conditioner" in t.lower() or "a/c" in t.lower():
            continue
        out.append(x)
    return out


def recommend_options_from_sheet(user_q: str, limit: int = 6) -> Dict[str, List[Dict[str,str]]]:
    """
    Scenario-aware selector. Returns ONLY the sections the user requested.
    Cold: boost cab/heater/defrost/wipers/glass; hide A/C.
    Indoor: prefer Sideshifter/Fork Positioner; suppress random clamps unless mentioned.
    Dark: prioritize lights/LED/beacons.
    """
    wants = _wants_sections(user_q)
    env   = _env_flags(user_q)

    options, attachments, tires = load_catalogs()

    # carve telemetry out of options by name/benefit text
    telemetry = {
        n: b for n, b in options.items()
        if _TELEM_PAT.search((n + " " + (b or "")).lower())
    }

    def rank(bucket: Dict[str,str]) -> List[Dict[str,str]]:
        if not bucket:
            return []
        ql = (user_q or "").lower()
        scored: List[Tuple[float, Dict[str,str]]] = []
        for name, benefit in bucket.items():
            text = (name + " " + (benefit or "")).lower()
            s = 0.01
            # generic alignment
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
        # post filters
        ranked = _prioritize_lighting(ranked, (user_q or "").lower())
        ranked = _drop_ac_when_cold(ranked, (user_q or "").lower())
        return ranked

    result: Dict[str, List[Dict[str,str]]] = {}

    # If the user explicitly asked, output only those sections
    if wants["any"]:
        if wants["tires"]:
            ranked_tires = rank(tires)
            if env["asks_non_mark"]:
                ranked_tires = [t for t in ranked_tires if "non-mark" in (t["name"] + " " + t.get("benefit","" )).lower()] or ranked_tires
            result["tires"] = ranked_tires[:limit] if limit else ranked_tires
        if wants["attachments"]:
            ranked_atts = rank(attachments)
            if env["indoor"] and not env["mentions_clamp"]:
                ranked_atts = [a for a in ranked_atts if not re.search(r"\bclamp\b", a["name"].lower())]
                # float alignment tools
                ranked_atts.sort(key=lambda a: int("sideshifter" in a["name"].lower() or "fork positioner" in a["name"].lower()), reverse=True)
            result["attachments"] = ranked_atts[:limit] if limit else ranked_atts
        if wants["options"]:
            ranked_opts = rank(options)
            result["options"] = ranked_opts[:limit] if limit else ranked_opts
        if wants["telemetry"]:
            r = rank(telemetry)
            result["telemetry"] = r[:limit] if limit else r
        return result

    # Broad question fallback: Options + a few Attachments; add Tires/Telemetry only if hinted
    ranked_opts = rank(options)
    ranked_atts = rank(attachments)
    if env["indoor"] and not env["mentions_clamp"]:
        ranked_atts = [a for a in ranked_atts if not re.search(r"\bclamp\b", a["name"].lower())]
        ranked_atts.sort(key=lambda a: int("sideshifter" in a["name"].lower() or "fork positioner" in a["name"].lower()), reverse=True)
    result["options"] = ranked_opts[:limit]
    result["attachments"] = ranked_atts[:4]
    if _TIRES_PAT.search(user_q or ""):
        result["tires"] = rank(tires)[:4]
    if _TELEM_PAT.search(user_q or ""):
        result["telemetry"] = rank(telemetry)[:3]
    return result


def render_sections_markdown(result: Dict[str, List[Dict[str,str]]]) -> str:
    """Render only sections that have items; skip empty/absent sections."""
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

# Convenience wrapper used by routes
_def_limit = 6

def render_catalog_sections(user_q: str, limit: Optional[int] = None) -> str:
    picks = recommend_options_from_sheet(user_q, limit or _def_limit)
    return render_sections_markdown(picks)


def generate_catalog_mode_response(user_q: str) -> Dict[str, Any]:
    picks = recommend_options_from_sheet(user_q, _def_limit)
    return {"markdown": render_sections_markdown(picks), "raw": picks}


def list_all_from_excel() -> Dict[str, List[str]]:
    opts, atts, tires = load_catalogs()
    return {
        "options": sorted(list(opts.keys())),
        "attachments": sorted(list(atts.keys())),
        "tires": sorted(list(tires.keys())),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Models / recommendation mode (safe, minimal, deterministic)
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_json(path: str) -> Any:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        log.exception("[ai_logic] Failed to load JSON: %s", path)
        return None

@lru_cache(maxsize=1)
def load_models() -> List[Dict[str,Any]]:
    data = _load_json(_MODELS_JSON) or []
    if isinstance(data, dict) and "models" in data:
        data = data.get("models", [])
    return data if isinstance(data, list) else []

@lru_cache(maxsize=1)
def load_accounts() -> List[Dict[str,Any]]:
    data = _load_json(_ACCTS_JSON) or []
    if isinstance(data, dict) and "accounts" in data:
        data = data.get("accounts", [])
    return data if isinstance(data, list) else []


def filter_models(q: str, k: int = 10) -> List[Dict[str,Any]]:
    """Very light heuristic filter by capacity / environment keywords."""
    models = load_models()
    ql = _lower(q)
    want_cap = None
    m = re.search(r"(\d[\d,\.]{2,})\s*(lb|lbs|pounds|#)", ql)
    if m:
        want_cap = _num(m.group(1))
    # alt phrasing: "5000 pounds" or just a bare number
    if want_cap is None:
        m2 = re.search(r"\b(\d{3,6})\b\s*(pound|pounds|lbs?|#)?", ql)
        if m2:
            want_cap = _num(m2.group(1))
    scored: List[Tuple[float, Dict[str,Any]]] = []
    for row in models:
        cap = _normalize_capacity_lbs(row) or 0
        score = 0.0
        if want_cap:
            # prefer models meeting/exceeding; small penalty if lower
            if cap >= (want_cap * 0.98):
                score += 2.0
            else:
                score -= 5.0
        # indoor preferences
        if any(w in ql for w in ("indoor","warehouse","epoxy","non-mark")):
            if "cushion" in _lower(row.get("Tire Type") or ""):
                score += 0.8
        # outdoor preferences
        if any(w in ql for w in ("outdoor","yard","gravel","dirt")):
            if "pneumatic" in _lower(row.get("Tire Type") or ""):
                score += 0.8
        # power hints
        pw = _power_of(row)
        if "electric" in ql and "electric" in pw:
            score += 0.6
        if any(w in ql for w in ("diesel","lpg","gas")) and any(w in pw for w in ("diesel","lpg","gas")):
            score += 0.6
        # aisle hint
        want_aisle = None
        ma = re.search(r"(aisle|raa)\s*(\d{2,3})", ql)
        if ma:
            want_aisle = _num(ma.group(2))
        ais = _normalize_aisle_in(row) or 0
        if want_aisle and ais and ais <= want_aisle:
            score += 0.7
        scored.append((score, row))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [m for _, m in scored[:max(1,k)]]


def select_models_for_question(q: str, k: int = 5) -> Tuple[List[Dict[str,Any]], str]:
    """Return (hits, allowed_description) to satisfy existing caller expectations."""
    hits = filter_models(q, k=k)
    allowed = f"selected_top_{len(hits)}_by_capacity_and_keywords"
    return hits, allowed


def allowed_models_block(q: str) -> str:
    hits, _ = select_models_for_question(q, k=3)
    names = [safe_model_name(m) for m in hits]
    return "\n".join(f"- {n}" for n in names) or "(no models)"


def safe_model_name(m: Dict[str, Any]) -> str:
    for k in ("Model","Model Name","model","code","name","Code"):
        if m.get(k):
            return str(m[k]).strip()
    return "N/A"


def model_meta_for(m: Dict[str,Any]) -> Dict[str,Any]:
    return {
        "name": safe_model_name(m),
        "capacity_lbs": _normalize_capacity_lbs(m),
        "aisle_in": _normalize_aisle_in(m),
        "lift_height_in": _normalize_height_in(m),
        "power": _power_of(m),
        "tire": _lower(m.get("Tire Type") or m.get("Tires") or m.get("Tire") or ""),
    }


def top_pick_meta(q: str) -> Dict[str,Any]:
    hits, _ = select_models_for_question(q, k=1)
    return model_meta_for(hits[0]) if hits else {}


def generate_forklift_context(q: str) -> str:
    hits, allowed = select_models_for_question(q, k=3)
    if not hits:
        return "No matching models found."
    lines = [f"Allowed set: {allowed}"]
    for i, m in enumerate(hits, 1):
        meta = model_meta_for(m)
        lines.append(
            f"{i}. {meta['name']} — {int(meta['capacity_lbs'] or 0)} lbs, "
            f"aisle≈{int(meta['aisle_in'] or 0)} in, power={meta['power'] or 'n/a'}, tires={meta['tire'] or 'n/a'}"
        )
    return "\n".join(lines)


def debug_parse_and_rank(q: str) -> Dict[str, Any]:
    return {
        "wants": _wants_sections(q),
        "env": _env_flags(q),
        "catalog_preview": recommend_options_from_sheet(q, limit=5),
        "models_preview": [model_meta_for(m) for m in filter_models(q, k=3)],
    }

# ─────────────────────────────────────────────────────────────────────────────
# Legacy shim names to silence old imports
# ─────────────────────────────────────────────────────────────────────────────
_LEGACY_EXPORTS: Dict[str, Any] = {
    "list_all_from_excel": list_all_from_excel,
    "num_from_keys": _num_from_keys,
}
if hashlib.md5(str(sorted(_LEGACY_EXPORTS.keys())).encode()).hexdigest():
    pass

# ─────────────────────────────────────────────────────────────────────────────
# Exports
# ─────────────────────────────────────────────────────────────────────────────
__all__ = [
    # Catalog IO / caches
    "load_catalogs", "load_catalog_rows", "load_options", "load_attachments", "load_tires_as_options", "load_tires",
    "options_lookup_by_name", "option_benefit",

    # Scenario picks & catalog renderers
    "recommend_options_from_sheet", "render_sections_markdown",
    "render_catalog_sections", "parse_catalog_intent", "generate_catalog_mode_response", "list_all_from_excel",

    # Model filtering & context
    "filter_models", "generate_forklift_context", "select_models_for_question",
    "allowed_models_block", "model_meta_for", "top_pick_meta",

    # Debug
    "debug_parse_and_rank",

    # Intentional small helpers sometimes imported
    "_num_from_keys",
]

# Provide a very small parse_catalog_intent for compatibility (some routes import it)
def parse_catalog_intent(user_q: str) -> Dict[str, Any]:
    wants = _wants_sections(user_q)
    which = None
    if wants["tires"] and not any([wants["attachments"], wants["options"], wants["telemetry"]]):
        which = "tires"
    elif wants["telemetry"] and not any([wants["tires"], wants["attachments"], wants["options"]]):
        which = "telemetry"
    elif wants["attachments"] and wants["options"]:
        which = "both"
    elif wants["attachments"]:
        which = "attachments"
    elif wants["options"]:
        which = "options"
    return {"which": which, "list_all": bool(re.search(r"\b(list|show|give|display)\b.*\b(all|full|everything)\b", (user_q or "").lower()))}
