"""
ai_logic.py
Pure helper module: account lookup + model filtering + prompt context builder.
Grounds model picks strictly on models.json and parses user needs robustly.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import os, json, re, difflib, hashlib, time, logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as _pd
except Exception:
    _pd = None

# ─────────────────────────────────────────────────────────────────────────────
# Catalog (Excel) path
# ─────────────────────────────────────────────────────────────────────────────
_OPTIONS_XLSX = os.environ.get(
    "HELI_CATALOG_XLSX",
    os.path.join(os.path.dirname(__file__), "data", "forklift_options_benefits.xlsx")
)

# ─────────────────────────────────────────────────────────────────────────────
# Normalization helpers (shared)
# ─────────────────────────────────────────────────────────────────────────────
def _norm(s: str) -> str:
    """Lowercase trim + collapse spaces, normalize punctuation for matching."""
    s0 = (s or "").lower().strip()
    s0 = re.sub(r"[\/\-–—_]+", " ", s0)
    s0 = re.sub(r"[()]+", " ", s0)
    s0 = re.sub(r"\bslip\s*[- ]?\s*sheet\b", "slipsheet", s0)  # unify "slip sheet"
    s0 = re.sub(r"\s+", " ", s0)
    return s0

def _make_code(s: str) -> str:
    """
    Stable, uppercase code for keys/IDs.
    - Collapse whitespace to single underscores
    - Keep A–Z, 0–9; strip others
    - Trim leading/trailing underscores
    """
    s0 = (s or "").strip().upper()
    s1 = re.sub(r"[^A-Z0-9]+", "_", s0)
    s2 = re.sub(r"_+", "_", s1).strip("_")
    return s2

# ─────────────────────────────────────────────────────────────────────────────
# Canon & lightweight auto-tags (used by some helpers)
# ─────────────────────────────────────────────────────────────────────────────
# Canonicalize Subcategory values (fixes typos & spacing)
CANON: dict[str, str] = {
    "hydraulic assist": "Hydraulic Assist",
    "filtration/ cooling": "Filtration/Cooling",
    "filtration/cooling": "Filtration/Cooling",
    "fork handling": "Fork Handling",
    "hydraulic control": "Hydraulic Control",
    "tire": "Tire",
    "tires": "Tire",
}

def _canon_subcat(s: str) -> str:
    key = _norm(s or "")
    return CANON.get(key, (s or "").strip())

# Reusable regex pieces
_TIRE_WORD = r"(?:tire|tyre)s?"

# Compile once
_RE_NON_MARKING = re.compile(r"(?i)\bnon[-\s]?marking\b|\bnm\s+(?:cushion|pneumatic)\b")
_RE_TIRE_CORE  = re.compile(rf"(?i)\b{_TIRE_WORD}\b(?!\s*clamp)")  # avoid tire/tyre clamp
_RE_TIRE_VARIANTS = re.compile(
    r"(?i)\b(?:dual(?:\s+solid)?\s+tires?|solid\s+tires?|non[-\s]?marking\s+tires?)\b"
)

# Light tag rules: (tag, regex) — keep minimal & safe
TAG_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("cold", re.compile(r"(?i)\bcold\s+storage\b|\bfreezer\b|\bsubzero\b")),
    ("visibility", re.compile(r"(?i)\b(blue|red)\s+light\b|\bled\b|\bbeacon\b")),
    ("safety", re.compile(r"(?i)\b(full\s+ops|operator\s+presence|seatbelt|backup\s+alarm)\b")),
    ("hydraulic", re.compile(r"(?i)\b(?:3|4|5)\s*valve\b|\bfinger\s*tip|\bfingertip\b")),
    ("tires", _RE_TIRE_CORE),
    ("non-marking", _RE_NON_MARKING),
]

def _is_tire(row: Dict[str, Any]) -> bool:
    """Returns True if this row describes a tire, avoiding 'tire/tyre clamp'."""
    name = (row.get("Option") or row.get("Name") or "").strip()
    benefit = (row.get("Benefit") or "").strip()
    typ = _norm(row.get("Type") or "")
    subcat = _canon_subcat(row.get("Subcategory"))
    text = f"{name} {benefit}"

    # Trust explicit Type/Subcategory first
    if typ in {"tire", "tires"} or subcat == "Tire":
        return True

    # Textual match for tire variants, but exclude clamps/attachments
    if re.search(r"(?i)\bclamp\b", text):
        return False

    return bool(_RE_TIRE_CORE.search(text) or _RE_TIRE_VARIANTS.search(text) or _RE_NON_MARKING.search(text))

# Attachment detection — conservative & readable
_ATTACHMENT_TERMS = re.compile(
    r"(?i)\b("
    r"clamp|carton\s*clamp|paper\s*roll\s*clamp|bale\s*clamp|"
    r"fork\s*positioner|sideshift(?:er)?|side[-\s]?shift|"
    r"rotator|push\s*/?\s*pull|push[-\s]?pull|"
    r"fork\s*spreader|fork\s*extensions?|"
    r"load\s*stabilizer|integral\s*carriage"
    r")\b"
)

def _is_attachment(row: Dict[str, Any]) -> bool:
    """
    Returns True when the row clearly describes an ATTACHMENT (not a tire, not a generic option).
    """
    typ = _norm(row.get("Type") or "")
    subcat = _norm(row.get("Subcategory") or "")
    name = (row.get("Option") or row.get("Name") or "")
    benefit = (row.get("Benefit") or "")
    text = f"{name} {benefit}"

    # 1) Trust explicit schema first
    if typ in {"attachment", "attachments"}:
        return True
    if subcat in {"attachment", "attachments"}:
        return True

    # 2) Avoid false positives: if it's clearly tires, bail out
    if typ in {"tire", "tires"} or _canon_subcat(row.get("Subcategory")) == "Tire":
        return False
    if re.search(r"(?i)\btires?\b|\btyres?\b", text):
        return False

    # 3) Keyword fallback
    return bool(_ATTACHMENT_TERMS.search(text))

def _is_option(row: Dict[str, Any]) -> bool:
    """Treat as 'Option' when clearly not an attachment or tire."""
    typ = _norm(row.get("Type") or "")
    subcat = _canon_subcat(row.get("Subcategory"))
    if typ in {"attachments", "attachment"}:
        return False
    if typ in {"tires", "tire"} or subcat == "Tire":
        return False
    return True

def _auto_tags(name: str, benefit: str, subcat: str) -> set[str]:
    """Lightweight tag inference for routing and UX chips."""
    text = f"{name or ''} {benefit or ''}".strip()
    tags: set[str] = set()

    if _RE_TIRE_CORE.search(text) or _RE_TIRE_VARIANTS.search(text) or _canon_subcat(subcat) == "Tire":
        # Avoid mis-tagging 'tire/tyre clamp'
        if not re.search(r"(?i)\bclamp\b", text):
            tags.add("tires")

    if _RE_NON_MARKING.search(text):
        tags.add("non-marking")

    for tag, pattern in TAG_RULES:
        if tag in {"tires", "non-marking"}:
            continue
        if pattern.search(text):
            tags.add(tag)

    return tags

# ─────────────────────────────────────────────────────────────────────────────
# Excel reader & normalizer
# ─────────────────────────────────────────────────────────────────────────────
def _read_catalog_df():
    """
    Returns a DataFrame with standardized columns: name, benefit, type, subcategory
    Types normalized to: option | attachment | tire
    """
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
        "tires": "tire", "tire": "tire"
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
    """
    Returns: (options: {name: benefit}, attachments: {name: benefit}, tires: {name: benefit})
    """
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
# Quick “legacy-shaped” accessors used around the app
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def load_options() -> List[dict]:
    options, _, tires = load_catalogs()
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
load_tires = load_tires_as_options  # alias

@lru_cache(maxsize=1)
def options_lookup_by_name() -> dict:
    return {o["name"].lower(): o for o in load_options()}

def option_benefit(name: str) -> Optional[str]:
    row = options_lookup_by_name().get((name or "").lower())
    return row["benefit"] if row else None

def refresh_catalog_caches():
    for fn in (load_catalogs, load_options, load_attachments,
               load_tires_as_options, options_lookup_by_name, load_catalog_rows):
        try:
            fn.cache_clear()
        except Exception:
            pass

# ─────────────────────────────────────────────────────────────────────────────
# Intent parsing for “list all …” & catalog intent
# ─────────────────────────────────────────────────────────────────────────────
_BOTH_PAT  = re.compile(r'\b(both\s+lists?|attachments\s+and\s+options|options\s+and\s+attachments)\b', re.I)
_ATT_PAT   = re.compile(r'\b(attachments?\s+only|attachments?)\b', re.I)
_OPT_PAT   = re.compile(r'\b(options?\s+only|options?)\b', re.I)
_TIRES_PAT = re.compile(r'\b(tires?|tyres?|tire\s*types?)\b', re.I)

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
    list_all = (
        bool(re.search(r'\b(list|show|give|display)\b.*\b(all|full|everything)\b', t)) or
        "all attachments" in t or "all options" in t or "all tires" in t or
        "full list of attachments" in t or "full list of options" in t or "full list of tires" in t or
        "tire types" in t or "types of tires" in t
    )
    return {"which": which, "list_all": list_all}

def _line(n, b):
    b = (b or "").strip()
    return f"- {n}" + (f" — {b}" if b else "")

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

def list_all_from_excel(user_text: str, max_per_section: int = 9999) -> str:
    """Public alias expected by some routes."""
    return _list_all_from_excel(user_text, max_per_section=max_per_section)

def _list_all_from_excel(user_text: str, max_per_section: int = 9999) -> str:
    df = _read_catalog_df()
    if df is None or df.empty:
        return "Catalog is empty or not loaded."
    def _dump(df_sub, header):
        out = [f"**{header}:**"]
        for _, r in df_sub.iterrows():
            nm = (r.get("name") or "").strip()
            b  = (r.get("benefit") or "").strip()
            out.append(f"- {nm}" + (f" — {b}" if b else ""))
        return out
    atts = df[df["type"] == "attachment"]
    opts = df[df["type"] == "option"]
    tires = df[df["type"] == "tire"]
    lines: List[str] = []
    if not tires.empty: lines += _dump(tires, "Tires")
    if not atts.empty: lines += _dump(atts, "Attachments")
    if not opts.empty: lines += _dump(opts, "Options")
    return "\n".join(lines) if lines else "No items found in the catalog."

# ─────────────────────────────────────────────────────────────────────────────
# Scenario classifiers (for attachments/options ranking)
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

def _pick_any(available_names: List[str], preferred_order: List[str]) -> List[str]:
    avail_lower = {n.lower(): n for n in available_names}
    for want in preferred_order:
        hit = avail_lower.get(want.lower())
        if hit:
            return [hit]
    return [available_names[0]] if available_names else []

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

@dataclass
class SelectorOutput:
    tire_primary: List[Tuple[str, str]]
    attachments_top: List[Tuple[str, str]]
    options_top: List[Tuple[str, str]]
    debug: Dict[str, Any]

def recommend_from_query(query: str, *, top_attachments: int = 5, top_options: int = 5) -> SelectorOutput:
    options, attachments, tires = load_catalogs()
    df = _read_catalog_df() or None
    subcats: Dict[str,str] = {}
    if df is not None and not df.empty and "name" in df.columns and "subcategory" in df.columns:
        sub = df[["name","subcategory"]].copy()
        sub["name"] = sub["name"].astype(str).str.strip()
        subcats = {r["name"]: (r["subcategory"] or "") for _, r in sub.iterrows()}

    tags = _classify(query)

    tire_names = list(tires.keys())
    _TIRE_RULES = [
        ({"indoor"}, lambda names: _pick_any(names, ["Non-Marking Tires","Non-Marking Cushion","NM Cushion","Non-Marking Pneumatic","NM Pneumatic","Solid Tires"])),
        ({"indoor","pedestrians"}, lambda names: _pick_any(names, ["Non-Marking Tires","Non-Marking Cushion","NM Cushion","Solid Tires"])),
        ({"tight"}, lambda names: _pick_any(names, ["Non-Marking Tires","Solid Tires","Non-Marking Cushion","NM Cushion"])),
        ({"debris"}, lambda names: _pick_any(names, ["Solid Tires","Dual Solid Tires"])),
        ({"soft_ground"}, lambda names: _pick_any(names, ["Dual Tires","Dual Solid Tires"])),
        ({"cold"}, lambda names: _pick_any(names, ["Solid Tires","Non-Marking Tires","Non-Marking Pneumatic","NM Pneumatic"])),
        (set(), lambda names: _pick_any(names, ["Solid Tires","Non-Marking Tires","Dual Tires","Dual Solid Tires"]))
    ]
    best_tire_list: List[str] = []
    for req_tags, chooser in _TIRE_RULES:
        if req_tags and not req_tags.issubset(tags):
            continue
        best_tire_list = chooser(tire_names)
        if best_tire_list:
            break
    if not best_tire_list and tire_names:
        best_tire_list = [tire_names[0]]
    tire_primary = [(n, tires.get(n, "")) for n in best_tire_list]

    att_ranked = _rank_items(attachments, tags, subcats)
    opt_ranked = _rank_items(options, tags, subcats)

    attachments_top = [(n, attachments[n]) for n, _ in att_ranked[:top_attachments]]
    options_top     = [(n, options[n])     for n, _ in opt_ranked[:top_options]]

    return SelectorOutput(
        tire_primary=tire_primary,
        attachments_top=attachments_top,
        options_top=options_top,
        debug={"tags": sorted(list(tags))}
    )

# ─────────────────────────────────────────────────────────────────────────────
# Text → flags (environment/work content) used by Excel pickers
# ─────────────────────────────────────────────────────────────────────────────
def _need_flags_from_text(user_q: str) -> dict:
    t = (user_q or "").lower()
    f: Dict[str, Any] = {}

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

    f["non_marking"]   = bool(re.search(r'non[-\s]?mark|no\s*marks?|black\s*marks?|avoid\s*marks?|scuff', t))

    f["alignment_frequent"] = re.search(r'align|line\s*up|tight\s*aisles|staging', t) is not None
    f["varied_width"]       = bool(re.search(r'vary|mixed\s*pallet|different\s*width|multiple\s*widths|mix\s*of\s*\d+\s*["in]?\s*(and|&)\s*\d+\s*["in]?\s*pallets?', t))
    f["paper_rolls"]        = re.search(r'paper\s*roll|newsprint|tissue', t) is not None
    f["slip_sheets"]        = re.search(r'slip[-\s]?sheet', t) is not None
    f["carpet"]             = "carpet" in t or "textile" in t
    f["long_loads"]         = bool(re.search(r'long|oversize|over[-\s]?length|overhang|\b\d+\s*[- ]?ft\b|\b\d+\s*foot\b|\b\d+\s*feet\b|crate[s]?', t))
    f["weighing"]           = re.search(r'weigh|scale|check\s*weight', t) is not None

    f["pedestrian_heavy"]   = re.search(r'pedestrian|foot\s*traffic|busy|congested|blind\s*corner|walkway', t) is not None
    f["poor_visibility"]    = re.search(r'low\s*light|dim|night|second\s*shift|poor\s*lighting', t) is not None

    f["extra_hydraulics"]   = "4th function" in t or "fourth function" in t
    f["multi_function"]     = "multiple clamp" in t or "multiple attachments" in t
    f["ergonomics"]         = re.search(r'ergonomic|fatigue|wrist|reach|comfort', t) is not None

    f["cold"]               = re.search(r'cold|freezer|refrigerated|winter', t) is not None
    f["hot"]                = re.search(r'hot|heat|summer|foundry|high\s*ambient', t) is not None

    f["speed_control"]      = re.search(r'limit\s*speed|speeding|zoned\s*speed', t) is not None
    f["ops_required"]       = re.search(r'ops|operator\s*presence|osha|insurance|audit|policy', t) is not None

    f["tall_operator"]      = "tall operator" in t or "headroom" in t
    f["high_loads"]         = re.search(r'high\s*mast|tall\s*stacks|top\s*heavy|elevated', t) is not None
    f["special_color"]      = "special color" in t or "paint" in t
    f["rigging"]            = "rigging" in t or "lift with crane" in t
    f["telematics"]         = "fics" in t or "fleet management" in t or "telematics" in t

    f["power_lpg"]          = re.search(r'\b(lpg|propane|lp[-\s]?gas)\b', t) is not None
    f["electric"]           = re.search(r'\b(lithium|li[-\s]?ion|electric|battery)\b', t) is not None
    f["_raw_text"]          = user_q
    return f

# ─────────────────────────────────────────────────────────────────────────────
# Name LUT helpers (Excel rows)
# ─────────────────────────────────────────────────────────────────────────────
def _lut_by_name(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lut: Dict[str, Dict[str, Any]] = {}
    for it in items or []:
        nm_sheet = (it.get("name") or it.get("Name") or it.get("option") or "").strip()
        if nm_sheet:
            lut[_norm(nm_sheet)] = it
    canonical = next((it for it in items if (it.get("name") or it.get("option")) == "Push/ Pull (Slip-Sheet)"), None)
    if canonical:
        lut["push pull slipsheet"] = canonical
    return lut

def _get_with_default(lut: Dict[str, Dict[str, Any]], name: str, default_benefit: str) -> Dict[str, Any]:
    row = lut.get(_norm(name))
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
# Excel-driven pickers
# ─────────────────────────────────────────────────────────────────────────────
def _pick_tire_from_flags(flags: Dict[str, Any], excel_lut: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    f = {k: bool(flags.get(k)) for k in [
        "non_marking", "rough", "debris", "puncture",
        "outdoor", "indoor", "soft_ground", "yard", "gravel", "mixed",
        "heavy_loads", "long_runs", "high_loads"
    ]}
    heavy_or_stability = f["heavy_loads"] or f["high_loads"]
    if f["non_marking"]:
        if f["outdoor"] or f["mixed"] or f["yard"] or heavy_or_stability:
            return _get_with_default(excel_lut, "Non-Marking Dual Tires",
                                     "Dual non-marking tread — keeps floors clean with extra footprint for stability.")
        return _get_with_default(excel_lut, "Non-Marking Tires",
                                 "Non-marking compound — prevents black marks on painted/epoxy floors.")
    if f["puncture"] or f["debris"] or f["rough"] or f["gravel"]:
        if f["soft_ground"] or f["heavy_loads"]:
            return _get_with_default(excel_lut, "Dual Solid Tires",
                                     "Puncture-proof dual solids — added footprint and stability on rough/soft ground.")
        return _get_with_default(excel_lut, "Solid Tires",
                                 "Puncture-proof solid tires — best for debris-prone or rough surfaces.")
    if f["yard"] or f["soft_ground"] or f["outdoor"] or f["mixed"]:
        return _get_with_default(excel_lut, "Dual Tires",
                                 "Wider footprint for traction and stability on soft or uneven ground.")
    if f["indoor"] and not f["outdoor"]:
        return _get_with_default(excel_lut, "Non-Marking Tires",
                                 "Indoor warehouse floors — non-marking avoids scuffs on concrete/epoxy.")
    if f["outdoor"] and not f["indoor"]:
        return _get_with_default(excel_lut, "Solid Tires",
                                 "Outdoor pavement/yard — reduced flats and lower maintenance.")
    return _get_with_default(excel_lut, "Dual Tires",
                             "Mixed or unspecified environment — dual improves footprint and stability.")

def _pick_attachments_from_excel(flags: Dict[str, Any], excel_lut: Dict[str, Dict[str, Any]], max_items: int = 6) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    t = (flags.get("_raw_text") or "").lower()
    pallets_mentioned = bool(re.search(r'\bpallet(s)?\b', t))
    def maybe_add(names: List[Tuple[str, str]]):
        for nm, default_ben in names:
            row = excel_lut.get(_norm(nm))
            if row:
                item = _get_with_default(excel_lut, nm, default_ben)
                if all(_norm(x["name"]) != _norm(item["name"]) for x in out):
                    out.append(item)
    if flags.get("cold"):
        if flags.get("alignment_frequent") or "tight aisle" in t or pallets_mentioned:
            maybe_add([("Sideshifter", "Aligns loads without repositioning — faster, cleaner placement.")])
        if flags.get("varied_width"):
            maybe_add([("Fork Positioner", "Adjust fork spread from the seat for mixed pallet sizes.")])
        return out[:max_items]
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
    if not out and (flags.get("indoor") or pallets_mentioned):
        maybe_add([("Sideshifter", "Aligns loads without repositioning — faster, cleaner placement.")])
        if flags.get("varied_width"):
            maybe_add([("Fork Positioner", "Adjust fork spread from the seat for mixed pallet sizes.")])
    return out[:max_items]

def _pick_options_from_excel(flags: Dict[str, Any], excel_lut: Dict[str, Dict[str, Any]], max_items: int = 6) -> List[Dict[str, Any]]:
    picks: List[Dict[str, Any]] = []
    def add_if_present(name: str, default_benefit: str):
        row = excel_lut.get(_norm(name))
        if not row: return
        ben_txt = (row.get("benefit") or row.get("Benefit") or "")
        if "suspend" in ben_txt.lower():  # ignore suspended lines
            return
        item = _get_with_default(excel_lut, name, default_benefit)
        if all(_norm(item["name"]) != _norm(x["name"]) for x in picks):
            picks.append(item)
    if flags.get("cold"):
        for nm, ben in [
            ("Panel mounted Cab", "Weather protection for outdoor duty; reduces operator fatigue."),
            ("Heater", "Keeps operator warm in cold environments; improves productivity."),
            ("Glass Windshield with Wiper", "Visibility in rain/snow; safer outdoor travel."),
            ("Top Rain-proof Glass", "Overhead visibility while shielding precipitation."),
            ("Rear Windshield Glass", "Reduces wind/snow ingress from the rear."),
        ]:
            add_if_present(nm, ben)
        if flags.get("electric"):
            add_if_present("Added cost for the cold storage package (for electric forklift)",
                           "Seals/heaters for freezer rooms; consistent performance in cold aisles.")
        for nm, ben in [
            ("LED Rear Working Light", "Bright, low-draw rear work lighting for dim winter shifts."),
            ("LED Rotating Light", "High-visibility 360° beacon to alert pedestrians."),
            ("Blue Light", "Directional warning for blind corners and intersections."),
            ("Visible backward radar, if applicable", "Audible/visual alerts while reversing."),
        ]:
            if len(picks) >= max_items: break
            add_if_present(nm, ben)
        if flags.get("ops_required"):
            add_if_present("Full OPS", "Operator presence system for safety interlocks.")
        if flags.get("speed_control") or flags.get("pedestrian_heavy"):
            add_if_present("Speed Control system (not for diesel engine)", "Limits travel speed to enhance safety.")
        return picks[:max_items]

    if flags.get("pedestrian_heavy") or flags.get("poor_visibility"):
        for nm, ben in [
            ("LED Rotating Light", "High-visibility 360° beacon to alert pedestrians."),
            ("Blue spot Light", "Projects a visible spot to warn pedestrians at intersections."),
            ("Red side line Light", "Creates a visual exclusion zone along the sides."),
            ("Visible backward radar, if applicable", "Audible/visual alerts while reversing."),
            ("Rear Working Light", "Improves rear visibility in dim aisles."),
            ("LED Rear Working Light", "Bright, low-draw rear lighting."),
            ("Blue Light", "Directional warning for blind corners."),
        ]:
            add_if_present(nm, ben)

    if flags.get("outdoor") or flags.get("hot"):
        for nm, ben in [
            ("Panel mounted Cab", "Weather protection for outdoor duty; reduces fatigue."),
            ("Air conditioner", "Comfort in hot conditions; keeps productivity steady."),
            ("Glass Windshield with Wiper", "Rain/dust visibility."),
            ("Top Rain-proof Glass", "Keeps overhead visibility while shielding precipitation."),
            ("Rear Windshield Glass", "Shields rear from wind/dust."),
        ]:
            add_if_present(nm, ben)

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
            add_if_present(nm, ben)

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
            add_if_present(nm, ben)

    if flags.get("heavy_loads") or flags.get("long_runs") or flags.get("outdoor"):
        add_if_present("Wet Disc Brake axle for CPCD50-100", "Lower maintenance braking under heavy duty cycles.")

    if flags.get("speed_control"):
        add_if_present("Speed Control system (not for diesel engine)", "Safer aisles via speed limiting.")
    if flags.get("ops_required"):
        add_if_present("Full OPS", "Operator presence system for OSHA-aligned safety interlocks.")

    if flags.get("ergonomics") or flags.get("long_runs"):
        add_if_present("Heli Suspension Seat with armrest", "Improves comfort and posture over long shifts.")
        add_if_present("Grammar Full Suspension Seat MSG65", "Premium suspension; pairs with fingertip controls.")

    if flags.get("power_lpg"):
        add_if_present("Swing Out Drop LPG Bracket", "Faster, safer tank changes; reduces strain.")
        add_if_present("LPG Tank", "Extra tank for quick swaps on multi-shift ops.")
        add_if_present("Low Fuel Indicator Light", "Prevents surprise stalls; prompt tank change.")

    if flags.get("electric") and flags.get("cold"):
        add_if_present("Added cost for the cold storage package (for electric forklift)",
                       "Seals/heaters for freezer rooms; consistent performance in cold aisles.")

    if flags.get("tall_operator") or flags.get("high_loads"):
        add_if_present("more 100mm higher overhead guard OHG as 2370mm (93”)",
                       "Extra headroom and load clearance where needed.")

    if flags.get("special_color"):
        add_if_present("Special Color Painting", "Matches facility requirement or corporate branding.")

    if flags.get("rigging"):
        add_if_present("Lifting eyes", "Simplifies hoisting the truck safely during site work or transport.")

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
    rec = recommend_options_from_sheet(user_text, max_total=max_per_section)
    lines: List[str] = []
    tire = rec.get("tire")
    lines.append("Tires (recommended):")
    lines.append(f"- {tire.get('name','')} — {tire.get('benefit','')}" if tire else "- (no specific tire triggered)")
    lines.append("Attachments (relevant):")
    atts = (rec.get("attachments") or [])[:max_per_section]
    if atts:
        for a in atts:
            lines.append(f"- {a.get('name','')} — {a.get('benefit','') or ''}".rstrip(" —"))
    else:
        lines.append("- (none triggered)")
    lines.append("Options (relevant):")
    opts = (rec.get("options") or [])[:max_per_section]
    if opts:
        for o in opts:
            lines.append(f"- {o.get('name','')} — {o.get('benefit','') or ''}".rstrip(" —"))
    else:
        lines.append("- (none triggered)")
    logging.info("[ai_logic] render_catalog_sections: scenario path, max=%s", max_per_section)
    return "\n".join(lines)

def generate_catalog_mode_response(user_q: str, max_per_section: int = 6) -> str:
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

logging.info("[ai_logic] Loaded accounts: %d | models: %d", len(accounts_raw), len(models_raw))

# ─────────────────────────────────────────────────────────────────────────────
# Model helpers
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
    if s is None: return None
    m = re.search(r"-?\d+(?:\.\d+)?", str(s))
    return float(m.group(0)) if m else None

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
# Requirement parsing (capacity/height/aisle/power/env)
# ─────────────────────────────────────────────────────────────────────────────
def _parse_capacity_lbs_intent(text: str) -> tuple[Optional[int], Optional[int]]:
    if not text: return (None, None)
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
    m = re.search(rf'between\s+{NUM}\s+and\s+{NUM}', t)
    if m:
        a, b = int(round(_n(m.group(1)))), int(round(_n(m.group(2))))
        return (min(a,b), max(a,b))
    m = re.search(rf'(?:up to|max(?:imum)?)\s+{NUM}\s*(?:{UNIT_LB})?', t)
    if m: return (None, int(round(_n(m.group(1)))))
    m = re.search(rf'(?:at least|minimum|min)\s+{NUM}\s*(?:{UNIT_LB})?', t)
    if m: return (int(round(_n(m.group(1)))), None)
    m = re.search(rf'{LOAD_WORDS}[^0-9k\-]*{KNUM}', t)
    if m: return (int(round(_n(m.group(1))*1000)), None)
    m = re.search(rf'{LOAD_WORDS}[^0-9\-]*{NUM}\s*(?:{UNIT_LB})?', t)
    if m: return (int(round(_n(m.group(1)))), None)
    m = re.search(rf'{KNUM}\s*(?:{UNIT_LB})?\b', t)
    if m: return (int(round(_n(m.group(1))*1000)), None)
    m = re.search(rf'{NUM}\s*{UNIT_LB}\b', t)
    if m: return (int(round(_n(m.group(1)))), None)
    m = re.search(rf'{NUM}\s*{UNIT_KG}\b', t)
    if m: return (int(round(_n(m.group(1))*2.20462)), None)
    m = re.search(rf'{NUM}\s*{UNIT_TONNE}\b', t)
    if m: return (int(round(_n(m.group(1))*2204.62)), None)
    m = re.search(rf'{NUM}\s*{UNIT_TON}\b', t)
    if m: return (int(round(_n(m.group(1))*2000)), None)
    m = re.search(rf'\b(\d[\d,]{3,5})\s*(?:{UNIT_LB})\b', t)
    if m: return (int(m.group(1).replace(",", "")), None)
    near = re.search(rf'(?:{LOAD_WORDS})\D{{0,12}}(\d{{4,5}})\b', t)
    if near: return (int(near.group(1)), None)
    return (None, None)

def _parse_requirements(q: str) -> Dict[str,Any]:
    ql = q.lower()
    cap_min, cap_max = _parse_capacity_lbs_intent(ql)
    cap_lbs = cap_min
    height_in = None
    for m in re.finditer(r'(\d[\d,\.]*)\s*(ft|feet|\'|in|\"|inches)\b', ql):
        raw, unit = m.group(1), m.group(2)
        ctx = ql[max(0, m.start()-18): m.end()+18]
        if re.search(r'\b(aisle|ra\s*aisle|right[-\s]?angle)\b', ctx): continue
        try: height_in = _to_inches(float(raw.replace(",","")), "ft") if unit in ("ft","feet","'") else float(raw.replace(",","")); break
        except: pass
    if height_in is None:
        m = re.search(r'(?:lift|raise|reach|height|clearance|mast)\D{0,12}(\d[\d,\.]*)\s*(ft|feet|\'|in|\"|inches)', ql)
        if m:
            raw, unit = m.group(1), m.group(2)
            try: height_in = _to_inches(float(raw.replace(",","")), "ft") if unit in ("ft","feet","'") else float(raw.replace(",",""))
            except: height_in = None
    aisle_in = None
    m = re.search(r'(?:aisle|aisles|aisle width)\D{0,12}(\d[\d,\.]*)\s*(?:in|\"|inches|ft|\')', ql)
    if m:
        raw, unitblob = m.group(1), m.group(0)
        try: aisle_in = _to_inches(float(raw.replace(",","")), "ft") if ("ft" in unitblob or "'" in unitblob) else float(raw.replace(",",""))
        except: pass
    if aisle_in is None:
        m = re.search(r'(?:right[-\s]?angle(?:\s+aisle|\s+stack(?:ing)?)?|ra\s*aisle|ras)\D{0,12}(\d[\d,\.]*)\s*(in|\"|inches|ft|\')', ql)
        if m:
            raw, unit = m.group(1), m.group(2)
            try: aisle_in = _to_inches(float(raw.replace(",","")), "ft") if unit in ("ft","'") else float(raw.replace(",",""))
            except: pass

    power_pref = None
    if any(w in ql for w in ["zero emission","zero-emission","emissions free","emissions-free","eco friendly","eco-friendly","green","battery powered","battery-powered","battery","lithium","li-ion","li ion","lead acid","lead-acid","electric"]): power_pref = "electric"
    if "diesel" in ql: power_pref = "diesel"
    if any(w in ql for w in ["lpg","propane","lp gas","gas (lpg)","gas-powered","gas powered"]): power_pref = "lpg"

    indoor  = any(w in ql for w in ["indoor","warehouse","inside","factory floor","distribution center","dc"])
    outdoor = any(w in ql for w in ["outdoor","yard","dock yard","construction","lumber yard","gravel","dirt","uneven","rough","pavement","parking lot","rough terrain","rough-terrain"])
    narrow  = ("narrow aisle" in ql) or ("very narrow" in ql) or ("vna" in ql) or ("turret" in ql) or ("reach truck" in ql) or ("stand-up reach" in ql) or (aisle_in is not None and aisle_in <= 96)

    tire_pref = None
    if any(w in ql for w in ["non-marking","non marking","nonmarking"]): tire_pref = "non-marking cushion"
    if tire_pref is None and any(w in ql for w in ["cushion","press-on","press on"]): tire_pref = "cushion"
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

# ─────────────────────────────────────────────────────────────────────────────
# Context block builder for your chat UI
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
        lines.append("- Not specified")

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

def _class_of(row: Dict[str, Any]) -> str:
    t = _text_from_keys(row, TYPE_KEYS)
    m = re.search(r'\bclass\s*([ivx]+)\b', (t or ""), re.I)
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
# Exports & Pylance “not accessed” silencer
# ─────────────────────────────────────────────────────────────────────────────
__all__ = [
    # Catalog IO / caches
    "load_catalogs", "load_catalog_rows", "refresh_catalog_caches",
    "load_options", "load_attachments", "load_tires_as_options", "load_tires",
    "options_lookup_by_name", "option_benefit",

    # Scenario picks & catalog renderers
    "recommend_options_from_sheet", "render_catalog_sections", "parse_catalog_intent",
    "generate_catalog_mode_response", "_list_all_from_excel", "list_all_from_excel",

    # Model filtering & context
    "filter_models", "generate_forklift_context", "select_models_for_question", "allowed_models_block",
    "model_meta_for", "top_pick_meta",

    # Debug
    "debug_parse_and_rank",

    # Intentional small helpers (sometimes imported in other places)
    "_plain", "_line", "_num_from_keys",
    "_is_attachment", "_is_tire", "_is_option", "_auto_tags", "_make_code",
]

# Touch-map so Pylance treats helpers as “used” without side effects.
_LEGACY_EXPORTS: Dict[str, Any] = {
    "list_all_from_excel": list_all_from_excel,
    "plain": _plain,
    "line_fmt": _line,
    "num_from_keys": _num_from_keys,
    "is_attachment": _is_attachment,
    "is_tire": _is_tire,
    "is_option": _is_option,
    "auto_tags": _auto_tags,
    "make_code": _make_code,
}
# Avoid “defined but not used”
if hashlib.md5(str(sorted(_LEGACY_EXPORTS.keys())).encode()).hexdigest():
    pass
