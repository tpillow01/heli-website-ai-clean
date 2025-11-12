"""
ai_logic.py — Catalog + Model reasoning for HELI AI
- Reads Excel (options/attachments/tires/telemetry) with resilient header mapping.
- Heuristically harvests tires + provides safe fallbacks so tires never come back empty.
- Focused responses: only the sections the user asked about (tires/options/attachments/telemetry).
- Scenario pickers for cold weather, debris, indoor epoxy, etc. (sheet-grounded).
- Simple model filter & metadata helpers (compat with existing blueprints).
- Context block generator used in your chat UI.

This file is intentionally comprehensive (not "minimal") to avoid regressions.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Imports & logger
# ─────────────────────────────────────────────────────────────────────────────
import os, json, re, hashlib, logging
from functools import lru_cache
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as _pd
except Exception:
    _pd = None

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ai_logic")

# ─────────────────────────────────────────────────────────────────────────────
# Paths / env
# ─────────────────────────────────────────────────────────────────────────────
_OPTIONS_XLSX = os.environ.get(
    "HELI_CATALOG_XLSX",
    os.path.join(os.path.dirname(__file__), "data", "forklift_options_benefits.xlsx"),
)
log.info("[ai_logic] Using catalog: %s (exists=%s)", _OPTIONS_XLSX, os.path.exists(_OPTIONS_XLSX))

def _load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

accounts_raw = _load_json(os.path.join(os.path.dirname(__file__), "accounts.json"))
models_raw   = _load_json(os.path.join(os.path.dirname(__file__), "models.json"))
if isinstance(models_raw, dict):
    models_raw = models_raw.get("models") or models_raw.get("data") or []
log.info("[ai_logic] Loaded accounts=%d | models=%d", len(accounts_raw), len(models_raw))

# ─────────────────────────────────────────────────────────────────────────────
# Normalizers & small utils
# ─────────────────────────────────────────────────────────────────────────────
def _norm_text(s: str) -> str:
    s0 = (s or "").strip()
    return " ".join(s0.split()).lower()

def _canon_subcat(s: str) -> str:
    CANON = {
        "hydraulic assist": "Hydraulic Assist",
        "filtration/ cooling": "Filtration/Cooling",
        "filtration/cooling": "Filtration/Cooling",
        "fork handling": "Fork Handling",
        "hydraulic control": "Hydraulic Control",
        "tire": "Tire",
        "tires": "Tire",
        "telemetry": "Telemetry",
        "telematics": "Telemetry",
        "safety lighting": "Safety Lighting",
        "weather/cab": "Weather/Cab",
        "cooling/filtration": "Cooling/Filtration",
        "protection": "Protection",
    }
    key = _norm_text(s)
    return CANON.get(key, (s or "").strip())

def _make_code(name: str) -> str:
    base = re.sub(r"[^A-Za-z0-9]+", "-", (name or "").strip()).strip("-").upper()
    return base[:48] if base else hashlib.md5((name or "").encode("utf-8")).hexdigest()[:8].upper()

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

def _line(n, b):
    b = (b or "").strip()
    return f"- {n}" + (f" — {b}" if b else "")

# ─────────────────────────────────────────────────────────────────────────────
# Excel reader (robust + keyword harvest + fallback)
# ─────────────────────────────────────────────────────────────────────────────
def _read_catalog_df():
    """
    Returns a DataFrame with columns: name, benefit, type, subcategory
    Normalized type in {'option','attachment','tire','telemetry'}.
    """
    if _pd is None:
        log.error("[ai_logic] pandas not available.")
        return None
    if not os.path.exists(_OPTIONS_XLSX):
        log.error("[ai_logic] Excel not found at %s", _OPTIONS_XLSX)
        return None

    try:
        df = _pd.read_excel(_OPTIONS_XLSX, engine="openpyxl")
    except Exception:
        df = _pd.read_excel(_OPTIONS_XLSX)

    if df is None or df.empty:
        log.error("[ai_logic] Excel opened but empty.")
        return None

    raw_cols = [str(c) for c in df.columns]
    log.info("[ai_logic] Excel columns: %s", raw_cols)

    cols = {str(c).lower().strip(): c for c in df.columns}
    name_col    = cols.get("name") or cols.get("option")
    benefit_col = cols.get("benefit") or cols.get("description") or cols.get("desc")
    type_col    = cols.get("type") or cols.get("category")
    subcat_col  = cols.get("subcategory")

    if not name_col:
        log.error("[ai_logic] Missing 'Name' or 'Option' column.")
        return None

    df = df.copy()
    df["__name__"]        = df[name_col].astype(str).str.strip()
    df["__benefit__"]     = (df[benefit_col].astype(str).str.strip() if benefit_col else "")
    df["__type_raw__"]    = (df[type_col].astype(str).str.strip().str.lower() if type_col else "")
    df["__subcategory__"] = (df[subcat_col].astype(str).str.strip() if subcat_col else "")
    df["__subcategory__"] = df["__subcategory__"].map(_canon_subcat)

    def _norm_type(nm: str, tp_raw: str, subcat: str) -> str:
        if tp_raw in {"option","options","opt"}: return "option"
        if tp_raw in {"attachment","attachments","att"}: return "attachment"
        if tp_raw in {"tire","tires"} or subcat == "Tire": return "tire"
        blob = f"{nm} {subcat}".lower()
        if any(k in blob for k in ("telemetry","telematic","fics","fleet management","portal")):
            return "telemetry"
        low = nm.lower()
        if any(k in low for k in ("clamp","sideshift","side shift","positioner","rotator","boom","pole","ram","extension",
                                  "push/ pull","push/pull","slip-sheet","slipsheet","bale","carton","drum","stabilizer","inverta")):
            return "attachment"
        if any(k in low for k in ("tire","tyre","pneumatic","cushion","non-mark","dual","nm ")):
            return "tire"
        return "option"

    df["__type__"] = df.apply(lambda r: _norm_type(r["__name__"], r["__type_raw__"], r["__subcategory__"]), axis=1)
    out = df.loc[df["__name__"] != "", ["__name__","__benefit__","__type__","__subcategory__"]].rename(
        columns={"__name__":"name","__benefit__":"benefit","__type__":"type","__subcategory__":"subcategory"}
    )

    c = lambda t: int((out["type"]==t).sum())
    log.info("[ai_logic] Normalized counts: options=%d attachments=%d tires=%d telemetry=%d",
             c("option"), c("attachment"), c("tire"), c("telemetry"))

    # If no tires tagged, harvest by keyword
    if c("tire") == 0:
        log.warning("[ai_logic] No rows tagged as tires; attempting keyword harvest.")
        harvested = []
        for _, r in out.iterrows():
            nm = (r.get("name") or "")
            ben = (r.get("benefit") or "")
            sub = (r.get("subcategory") or "")
            blob = f"{nm} {ben} {sub}".lower()
            if any(k in blob for k in ("tire","tyre","non-mark","pneumatic","cushion","dual","solid","nm ")):
                harvested.append({"name": nm, "benefit": ben, "type": "tire", "subcategory": "Tire"})
        if harvested:
            out = _pd.concat([out, _pd.DataFrame(harvested)], ignore_index=True)

    return out

# ─────────────────────────────────────────────────────────────────────────────
# Catalog loaders & lookups (NEVER empty tires)
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def load_catalog_rows() -> List[dict]:
    df = _read_catalog_df()
    return [] if df is None else df.to_dict(orient="records")

@lru_cache(maxsize=1)
def load_catalogs() -> tuple[dict, dict, dict, dict, dict]:
    """
    Returns (options, attachments, tires, telemetry, subcats) as dicts.
    Guarantees a non-empty 'tires' via harvest/fallback if Excel exists, or pure fallback if not.
    """
    df = _read_catalog_df()
    if df is None or df.empty:
        log.error("[ai_logic] Catalog not loaded; using SAFE FALLBACK tires.")
        fallback_tires = {
            "Solid Tires": "Puncture-proof, low-maintenance tires for debris-prone floors.",
            "Dual Tires": "Wider footprint for better stability and traction on soft ground.",
            "Dual Solid Tires": "Combines puncture resistance with extra stability and load support.",
            "Non-Marking Tires": "Protect indoor floors by avoiding black scuff marks.",
            "Non-Marking Dual Tires": "Floor-safe traction with a wider, more stable footprint.",
        }
        return {}, {}, fallback_tires, {}, {}

    df = df.copy()
    df["name"]    = df["name"].astype(str).str.strip()
    df["benefit"] = df["benefit"].fillna("").astype(str).str.strip()
    df["type"]    = df["type"].astype(str).str.strip().str.lower()
    df["subcategory"] = df["subcategory"].astype(str).str.strip()

    def bucket(t: str) -> dict:
        sub = df[df["type"] == t]
        sub = sub.loc[~sub["name"].str.lower().duplicated(keep="last")]
        return {r["name"]: r["benefit"] for _, r in sub.iterrows()}

    options    = bucket("option")
    attachments= bucket("attachment")
    tires      = bucket("tire")
    telemetry  = bucket("telemetry")
    subcats    = {r["name"]: r["subcategory"] for _, r in df.iterrows() if r["name"]}

    if not tires:
        log.error("[ai_logic] Tires still empty post-normalization; applying SAFE FALLBACK.")
        tires = {
            "Solid Tires": "Puncture-proof, low-maintenance tires for debris-prone floors.",
            "Dual Tires": "Wider footprint for better stability and traction on soft ground.",
            "Dual Solid Tires": "Combines puncture resistance with extra stability and load support.",
            "Non-Marking Tires": "Protect indoor floors by avoiding black scuff marks.",
            "Non-Marking Dual Tires": "Floor-safe traction with a wider, more stable footprint.",
        }

    return options, attachments, tires, telemetry, subcats

@lru_cache(maxsize=1)
def load_options() -> List[dict]:
    options, _, tires, _, _ = load_catalogs()
    rows = []
    for name, ben in {**options, **tires}.items():
        rows.append({"code": _make_code(name), "name": name, "benefit": ben})
    return rows

@lru_cache(maxsize=1)
def load_attachments() -> List[dict]:
    _, attachments, _, _, _ = load_catalogs()
    return [{"name": n, "benefit": b} for n, b in attachments.items()]

@lru_cache(maxsize=1)
def load_tires_as_options() -> List[dict]:
    _, _, tires, _, _ = load_catalogs()
    return [{"code": _make_code(n), "name": n, "benefit": b} for n, b in tires.items()]
load_tires = load_tires_as_options  # alias

@lru_cache(maxsize=1)
def options_lookup_by_name() -> dict:
    return {o["name"].lower(): o for o in load_options()}

def option_benefit(name: str) -> Optional[str]:
    row = options_lookup_by_name().get((name or "").lower())
    return row["benefit"] if row else None

def refresh_catalog_caches() -> None:
    """Clear memoized Excel/lookups (safe if some names missing)."""
    for name in [
        "load_catalogs",
        "load_catalog_rows",
        "load_options",
        "load_attachments",
        "load_tires_as_options",
        "options_lookup_by_name",
    ]:
        fn = globals().get(name)
        cache_clear = getattr(fn, "cache_clear", None)
        if callable(cache_clear):
            try:
                cache_clear()  # type: ignore[misc]
            except Exception:
                pass

# ─────────────────────────────────────────────────────────────────────────────
# Intent parsing (which sections to show)
# ─────────────────────────────────────────────────────────────────────────────
_PAT_TIRES      = re.compile(r'\b(tires?|tyres?|tire\s*types?)\b', re.I)
_PAT_ATTACH     = re.compile(r'\battachments?\b', re.I)
_PAT_OPTIONS    = re.compile(r'\boptions?\b', re.I)
_PAT_TELEMETRY  = re.compile(r'\b(telemetry|telematics?|fics|fleet\s*management|portal)\b', re.I)
def _intent_sections(user_q: str) -> List[str]:
    t = (user_q or "").strip().lower()
    wants: List[str] = []
    if _PAT_TELEMETRY.search(t): wants.append("telemetry")
    if _PAT_TIRES.search(t):     wants.append("tires")
    if _PAT_ATTACH.search(t):    wants.append("attachments")
    if _PAT_OPTIONS.search(t):   wants.append("options")
    if not wants:
        return ["tires","attachments","options","telemetry"]
    return wants

def _asked_all(user_q: str) -> bool:
    t = (user_q or "").lower()
    return any(k in t for k in ("all","full list","every"))

# ─────────────────────────────────────────────────────────────────────────────
# Scenario/meta helpers for ranking items
# ─────────────────────────────────────────────────────────────────────────────
_KEYWORDS = {
    "indoor":        [r"\bindoor\b", r"\bwarehouse\b", r"\bpolished\b", r"\bepoxy\b"],
    "outdoor":       [r"\boutdoor\b", r"\byard\b", r"\bconstruction\b", r"\bparking\b"],
    "pedestrians":   [r"\bpedestrian", r"\bfoot traffic", r"\bpeople\b", r"\bbusy aisles\b"],
    "tight":         [r"\btight\b", r"\bnarrow\b", r"\baisle", r"\bturn\b"],
    "cold":          [r"\bcold\b", r"\bfreezer\b", r"\bsubzero\b", r"\bcooler\b"],
    "wet":           [r"\bwet\b", r"\brain\b", r"\bice\b", r"\bsnow\b"],
    "debris":        [r"\bdebris\b", r"\bnails\b", r"\bscrap\b", r"\bpuncture\b"],
    "soft_ground":   [r"\bsoft\b", r"\bgravel\b", r"\bdirt\b", r"\bgrass\b", r"\bsoil\b", r"\bsand\b"],
}
_SUBCAT_HINTS = {
    "Safety Lighting": ["pedestrians", "tight"],
    "Hydraulic Control": ["tight"],
    "Weather/Cab": ["cold", "wet", "outdoor"],
    "Cooling/Filtration": ["debris", "outdoor"],
    "Protection": ["debris", "outdoor"],
}
_ITEM_BOOSTS = {
    "Blue Light": ["pedestrians", "tight"],
    "LED Rotating Light": ["pedestrians"],
    "Visible backward radar": ["pedestrians"],
    "Full OPS": ["pedestrians"],
    "Sideshifter": ["tight"],
    "Fork Positioner": ["tight"],
    "Cold storage": ["cold"],
    "Heater": ["cold"],
    "Cab": ["cold", "outdoor", "wet"],
    "Windshield": ["wet", "outdoor"],
    "Pre air cleaner": ["debris", "outdoor"],
    "Dual Air Filter": ["debris", "outdoor"],
    "Radiator protection": ["debris", "outdoor"],
    "Steel Belly Pan": ["debris", "outdoor"],
}

def _match_any(text: str, patterns: List[str]) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in patterns)

def _classify(query: str) -> set:
    tags = set()
    q = (query or "").lower()
    for tag, pats in _KEYWORDS.items():
        if any(re.search(p, q) for p in pats):
            tags.add(tag)
    return tags

def _rank_items(bucket: Dict[str, str], tags: set, subcats: Dict[str, str]) -> List[Tuple[str, float]]:
    out: List[Tuple[str,float]] = []
    for name, benefit in bucket.items():
        score = 0.0
        display = f"{name} {benefit}".lower()
        sub = (subcats.get(name) or "").strip()
        if sub:
            for tag, tagset in _SUBCAT_HINTS.items():
                if sub.lower().startswith(tag.lower()):
                    for t in tagset:
                        if t in tags:
                            score += 2.5
        for key, tag_list in _ITEM_BOOSTS.items():
            if key.lower() in display:
                for t in tag_list:
                    if t in tags:
                        score += 1.5
        if "pedestrian" in display and "pedestrians" in tags: score += 1.2
        if "cold" in display and "cold" in tags: score += 1.2
        if "rain" in display and "wet" in tags: score += 1.2
        if any(w in display for w in ["protection","guard","shield","screen"]) and ("debris" in tags or "outdoor" in tags):
            score += 1.2
        if benefit: score += 0.05
        out.append((name, score))
    out.sort(key=lambda x: x[1], reverse=True)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Tires: picker (avoid "always Dual")
# ─────────────────────────────────────────────────────────────────────────────
def _pick_tires_from_query(q: str, tires: Dict[str,str]) -> List[Tuple[str,str]]:
    t = (q or "").lower()
    names = list(tires.keys())
    if not names:
        return []
    # Explicit non-marking / epoxy / no scuff
    if re.search(r'non[-\s]?mark|no\s*scuff|avoid\s*marks?|epoxy|polished', t):
        ordered = ["Non-Marking Tires","Non-Marking Cushion","NM Cushion","Non-Marking Pneumatic","NM Pneumatic","Solid Tires"]
        for want in ordered:
            if want in tires:
                return [(want, tires[want])]
        for n in names:
            if "non-mark" in n.lower() or n.lower().startswith("nm "):
                return [(n, tires[n])]

    # Debris/puncture/yard/soft ground
    if any(k in t for k in ("debris","puncture","nails","scrap","yard","gravel","dirt","soft ground","rough","pothole")):
        for want in ["Dual Solid Tires","Solid Tires","Dual Tires"]:
            if want in tires:
                return [(want, tires[want])]
        for n in names:
            if "solid" in n.lower():
                return [(n, tires[n])]

    # Indoor bias
    if "indoor" in t and "outdoor" not in t:
        for want in ["Non-Marking Tires","Non-Marking Cushion","Solid Tires"]:
            if want in tires:
                return [(want, tires[want])]

    # Cold: allow solids/non-marking pneu if present
    if "cold" in t or "freezer" in t:
        for want in ["Solid Tires","Non-Marking Tires","Non-Marking Pneumatic","NM Pneumatic"]:
            if want in tires:
                return [(want, tires[want])]

    # Neutral (do NOT force Dual without reason)
    return [(names[0], tires[names[0]])]

# ─────────────────────────────────────────────────────────────────────────────
# Top-level catalog recommender
# ─────────────────────────────────────────────────────────────────────────────
def recommend_from_catalog(user_q: str, max_items: int = 6) -> Dict[str, List[Tuple[str,str]]]:
    options, attachments, tires, telemetry, subcats = load_catalogs()
    tags = _classify(user_q)
    asked = _intent_sections(user_q)
    resp: Dict[str, List[Tuple[str,str]]] = {}

    # Tires
    if "tires" in asked:
        if _asked_all(user_q):
            resp["tires"] = sorted([(n, b) for n, b in tires.items()], key=lambda x: x[0].lower())
        else:
            picks = _pick_tires_from_query(user_q, tires)
            resp["tires"] = picks[:max_items] if picks else [(n, tires[n]) for n in list(tires.keys())[:max_items]]

    # Telemetry
    if "telemetry" in asked:
        tel: List[Tuple[str,str]] = []
        for n, b in telemetry.items():
            tel.append((n, b))
        # keyword sweep if sheet lacked telemetry tagging
        if not tel:
            for n, b in options.items():
                blob = f"{n} {b}".lower()
                if any(k in blob for k in ("telemetry","telematic","fics","fleet management","portal")):
                    tel.append((n, b))
        resp["telemetry"] = tel[:max_items]

    # Attachments (rank by scenario)
    if "attachments" in asked:
        ranked = _rank_items(attachments, tags, subcats)
        resp["attachments"] = [(n, attachments[n]) for n, _ in ranked[:max_items]]

    # Options (rank by scenario; exclude tires/attachments/telemetry types)
    if "options" in asked:
        ranked = _rank_items(options, tags, subcats)
        resp["options"] = [(n, options[n]) for n, _ in ranked[:max_items]]

    # Generic case: show non-empty in canonical order
    if asked == ["tires","attachments","options","telemetry"]:
        resp.setdefault("tires", _pick_tires_from_query(user_q, tires))
        if "attachments" not in resp:
            ranked = _rank_items(attachments, tags, subcats); resp["attachments"] = [(n, attachments[n]) for n, _ in ranked[:max_items]]
        if "options" not in resp:
            ranked = _rank_items(options, tags, subcats); resp["options"] = [(n, options[n]) for n, _ in ranked[:max_items]]
        if "telemetry" not in resp:
            tel = [(n, telemetry[n]) for n in list(telemetry.keys())[:max_items]]
            resp["telemetry"] = tel

    log.info("[ai_logic] respond counts: options=%d attachments=%d tires=%d telemetry=%d",
             len(options), len(attachments), len(tires), len(telemetry))
    return resp

# ─────────────────────────────────────────────────────────────────────────────
# Renderers
# ─────────────────────────────────────────────────────────────────────────────
def _lines(header: str, items: List[Tuple[str,str]]) -> List[str]:
    if header:
        if not items:
            return [f"**{header}:**", "- (none found)"]
        out = [f"**{header}:**"]
        for n, b in items:
            b = (b or "").strip()
            out.append(f"- {n}" + (f" — {b}" if b else ""))
        return out
    return [f"- {n}" + (f" — {b}" if (b or "").strip() else "") for n, b in (items or [])]

def generate_catalog_mode_response(user_q: str, max_per_section: int = 6) -> str:
    picks = recommend_from_catalog(user_q, max_items=max_per_section)
    asked = _intent_sections(user_q)

    # If specific category asked, print only that(those) sections
    if asked != ["tires","attachments","options","telemetry"]:
        sections: List[str] = []
        for key in asked:
            title = {"tires":"Tires","attachments":"Attachments","options":"Options","telemetry":"Telemetry"}.get(key, key.title())
            sections += _lines(title, picks.get(key, []))
        out = "\n".join(sections).strip()
        return out if out else "(none found)"

    # Generic: show non-empty sections in canonical order
    order = [("tires","Tires"),("attachments","Attachments"),("options","Options"),("telemetry","Telemetry")]
    sections: List[str] = []
    for key, title in order:
        items = picks.get(key, [])
        if items:
            sections += _lines(title, items)
    return "\n".join(sections).strip() or "(none found)"

def list_all_from_excel(user_text: str, max_per_section: int = 9999) -> str:
    rows = load_catalog_rows()
    tires = [(r["name"], r.get("benefit","")) for r in rows if (r.get("type") or "") == "tire"]
    atts  = [(r["name"], r.get("benefit","")) for r in rows if (r.get("type") or "") == "attachment"]
    opts  = [(r["name"], r.get("benefit","")) for r in rows if (r.get("type") or "") == "option"]
    tele  = [(r["name"], r.get("benefit","")) for r in rows if (r.get("type") or "") == "telemetry"]
    blocks: List[str] = []
    if tires: blocks += _lines("Tires", tires[:max_per_section])
    if atts:  blocks += _lines("Attachments", atts[:max_per_section])
    if opts:  blocks += _lines("Options", opts[:max_per_section])
    if tele:  blocks += _lines("Telemetry", tele[:max_per_section])
    return "\n".join(blocks) if blocks else "Catalog is empty or not loaded."

# ─────────────────────────────────────────────────────────────────────────────
# Minimal model helpers (compat with other blueprints)
# ─────────────────────────────────────────────────────────────────────────────
TYPE_KEYS  = ["Type","Category","Segment","Class","Class/Type","Truck Type"]
POWER_KEYS = ["Power","power","Fuel","fuel","Drive","Drive Type","Power Type","PowerType"]
HEIGHT_KEYS = [
    "Lift Height_in","Max Lift Height (in)","Lift Height","Max Lift Height",
    "Mast Height","lift_height_in","LiftHeight","Lift Height (in)","Mast Height (in)"
]
AISLE_KEYS = [
    "Aisle_min_in","Aisle Width_min_in","Aisle Width (in)","Min Aisle (in)",
    "Right Angle Aisle (in)","Right-Angle Aisle (in)","RA Aisle (in)"
]
CAPACITY_KEYS = [
    "Capacity_lbs","capacity_lbs","Capacity","Rated Capacity","Load Capacity",
    "Capacity (lbs)","capacity","LoadCapacity","capacityLbs","RatedCapacity",
    "Load Capacity (lbs)","Rated Capacity (lbs)"
]

def _text_from_keys(row: Dict[str,Any], keys: List[str]) -> str:
    for k in keys:
        v = row.get(k)
        if v:
            return str(v)
    return ""

def _num(s: Any) -> Optional[float]:
    if s is None: return None
    m = re.search(r"-?\d+(?:\.\d+)?", str(s))
    return float(m.group(0)) if m else None

def _to_inches(val: float, unit: str) -> float:
    u = unit.lower()
    if "ft" in u or "'" in u: return float(val) * 12.0
    return float(val)

def _num_from_keys(row: Dict[str,Any], keys: List[str]) -> Optional[float]:
    for k in keys:
        if k in row and str(row[k]).strip() != "":
            v = _num(row[k]); 
            if v is not None:
                return v
    return None

def _normalize_capacity_lbs(row: Dict[str,Any]) -> Optional[float]:
    for k in CAPACITY_KEYS:
        if k in row:
            s = str(row[k]).strip()
            if not s: continue
            if re.search(r"\bkg\b", s, re.I):
                v = _num(s); return float(v * 2.20462) if v is not None else None
            if re.search(r"\btonne\b|\bmetric\s*ton\b|\b(?<!f)\bt\b", s, re.I):
                v = _num(s); return float(v * 2204.62) if v is not None else None
            if re.search(r"\btons?\b", s, re.I):
                v = _num(s); return float(v * 2000.0) if v is not None else None
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
    if "non-mark" in t: return "non-marking cushion"
    if "cushion" in t or "press" in t: return "cushion"
    if "pneumatic" in t or "super elastic" in t or "solid" in t: return "pneumatic"
    return t

def _safe_model_name(m: Dict[str, Any]) -> str:
    for k in ("Model","Model Name","model","code","name","Code"):
        if m.get(k):
            return str(m[k]).strip()
    return "N/A"

def _power_matches(pref: Optional[str], powr_text: str) -> bool:
    if not pref: return True
    p = pref.lower()
    t = (powr_text or "").lower()
    if p == "electric": return any(x in t for x in ("electric","lithium","li-ion","lead","battery"))
    if p == "lpg": return any(x in t for x in ("lpg","propane","lp gas","gas"))
    if p == "diesel": return "diesel" in t
    return p in t

# ─────────────────────────────────────────────────────────────────────────────
# Requirement parsing + filter & ranking
# ─────────────────────────────────────────────────────────────────────────────
def _parse_requirements(q: str) -> Dict[str,Any]:
    ql = (q or "").lower()
    # capacity min only (simple)
    cap = None
    m = re.search(r'(\d[\d,]{3,5})\s*(?:lb|lbs|pounds?)\b', ql)
    if m:
        cap = int(m.group(1).replace(",", ""))
    # height
    height_in = None
    m = re.search(r'(?:lift|reach|height|mast)\D{0,12}(\d[\d,\.]*)\s*(ft|feet|\'|in|\"|inches)', ql)
    if m:
        val = float(m.group(1).replace(",",""))
        height_in = _to_inches(val, "ft") if m.group(2) in ("ft","feet","'") else val
    # aisle
    aisle_in = None
    m = re.search(r'(?:aisle|right[-\s]?angle)\D{0,12}(\d[\d,\.]*)\s*(in|\"|inches|ft|\')', ql)
    if m:
        val = float(m.group(1).replace(",",""))
        aisle_in = _to_inches(val, "ft") if m.group(2) in ("ft","'") else val
    # power
    power = None
    if any(w in ql for w in ["electric","lithium","li-ion","battery"]): power = "electric"
    if "diesel" in ql: power = "diesel"
    if any(w in ql for w in ["lpg","propane","lp gas"]): power = "lpg"
    # env
    indoor  = any(w in ql for w in ["indoor","warehouse","inside","factory floor"])
    outdoor = any(w in ql for w in ["outdoor","yard","gravel","dirt","rough","parking"])
    narrow = ("narrow aisle" in ql) or (aisle_in is not None and aisle_in <= 96)
    # tire pref
    tire_pref = None
    if any(w in ql for w in ["non-marking","non marking","nonmarking","no scuff","no marks","epoxy","polished"]):
        tire_pref = "non-marking cushion"
    elif any(w in ql for w in ["cushion","press-on","press on"]):
        tire_pref = "cushion"
    elif any(w in ql for w in ["pneumatic","solid pneumatic","super elastic","foam filled","foam-filled"]):
        tire_pref = "pneumatic"
    else:
        if indoor and not outdoor: tire_pref = "cushion"
        elif outdoor and not indoor: tire_pref = "pneumatic"

    return dict(
        cap_lbs=cap, height_in=height_in, aisle_in=aisle_in,
        power_pref=power, indoor=indoor, outdoor=outdoor, narrow=narrow, tire_pref=tire_pref
    )

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
            if cap <= 0: continue
            if cap < cap_need: continue
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

def _class_of(row: Dict[str, Any]) -> str:
    t = _text_from_keys(row, TYPE_KEYS)
    m = re.search(r'\bclass\s*([ivx]+)\b', (t or ""), re.I)
    if m:
        return m.group(1).upper()
    tU = (t or "").strip().upper()
    return tU if tU in {"I","II","III","IV","V"} else ""

def model_meta_for(row: Dict[str, Any]) -> tuple[str, str, str]:
    """Return (model_code, class_roman, power_text) for a model row."""
    code = _safe_model_name(row)
    cls = _class_of(row)
    pwr = _text_from_keys(row, POWER_KEYS)
    return (code, cls, pwr)

def top_pick_meta(user_q: str):
    hits = filter_models(user_q, limit=1)
    return None if not hits else model_meta_for(hits[0])

# ─────────────────────────────────────────────────────────────────────────────
# Customer blocks & context used in chat UI
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

    # Tires/attachments/options via catalog
    rec = recommend_from_catalog(user_q, max_items=6)
    tires = rec.get("tires", [])
    atts  = rec.get("attachments", [])
    opts  = rec.get("options", [])

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
    if tires:
        t0 = tires[0]
        lines.append(f"- {t0[0]} — {t0[1]}".rstrip(" —"))
    else:
        lines.append("- Not specified")

    lines.append("\nAttachments:")
    if atts:
        for a in atts:
            lines.append(_line(a[0], a[1]))
    else:
        lines.append("- Not specified")

    lines.append("\nOptions:")
    if opts:
        for o in opts:
            lines.append(_line(o[0], o[1]))
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
# Public convenience & debug helpers used by blueprints
# ─────────────────────────────────────────────────────────────────────────────
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

def catalog_health() -> dict:
    df = _read_catalog_df()
    if df is None:
        return {"loaded": False, "exists": os.path.exists(_OPTIONS_XLSX), "path": _OPTIONS_XLSX}
    counts = {
        "options": int((df["type"]=="option").sum()),
        "attachments": int((df["type"]=="attachment").sum()),
        "tires": int((df["type"]=="tire").sum()),
        "telemetry": int((df["type"]=="telemetry").sum()),
    }
    return {"loaded": True, "counts": counts, "path": _OPTIONS_XLSX}

# ─────────────────────────────────────────────────────────────────────────────
# Public exports
# ─────────────────────────────────────────────────────────────────────────────
__all__ = [
    # Catalog IO / caches
    "load_catalogs", "load_catalog_rows", "refresh_catalog_caches",
    "load_options", "load_attachments", "load_tires_as_options", "load_tires",
    "options_lookup_by_name", "option_benefit",

    # Responders
    "recommend_from_catalog", "generate_catalog_mode_response", "list_all_from_excel",

    # Model filtering & context
    "filter_models", "generate_forklift_context", "select_models_for_question", "allowed_models_block",
    "model_meta_for", "top_pick_meta", "debug_parse_and_rank",

    # Utilities often imported elsewhere
    "_plain", "_line", "_num_from_keys",

    # Diagnostics
    "catalog_health",
]
