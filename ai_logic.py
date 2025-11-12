# pyright: reportUnusedFunction=false
# pyright: reportUnusedImport=false
"""
ai_logic.py
Pure helper module: account lookup + model filtering + prompt context builder.
Grounds model picks strictly on models.json and parses user needs robustly.
"""

from __future__ import annotations
import json, re, difflib, os, time, hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Dict, Any, Tuple, Optional

# --- Options / Attachments / Tires loader (Excel: ./data/forklift_options_benefits.xlsx) ---

try:
    import pandas as _pd  # uses your existing pandas from requirements.txt
except Exception:
    _pd = None

# Single source of truth: the typed catalog Excel
_OPTIONS_XLSX = os.environ.get(
    "HELI_CATALOG_XLSX",
    os.path.join(os.path.dirname(__file__), "data", "forklift_options_benefits.xlsx")
)

# ── Canonicalize subcategories + lightweight auto-tagging ─────────────────

# Fix common typos/spacing in Subcategory values coming from Excel
CANON = {
    "hyrdaulic assist": "Hydraulic Assist",
    "filtration/ cooling": "Filtration/Cooling",
    "fork handling": "Fork Handling",
}
def _canon_subcat(s: str) -> str:
    s0 = (s or "").strip()
    key = " ".join(s0.split()).lower()  # collapse multiple spaces
    return CANON.get(key, s0)

# Simple tag rules (used later to let responses react to context words)
TAG_RULES = [
    ("cold",       r"cold|freez|sub\s*zero|heater|wiper|windshield|cab|enclos|winter|snow|ice"),
    ("comfort",    r"comfort|seat|suspension|armrest|vibration|ergonom"),
    ("dust",       r"dust|debris|screen|pre\s*air|dual\s*air\s*filter|belly\s*pan|radiator\s*screen"),
    ("visibility", r"light|beacon|blue|spot|led|radar|camera|wiper|windshield"),
    ("safety",     r"ops|radar|camera|red\s*line|blue\s*spot"),
    ("hydraulic",  r"hydraulic|valve|finger"),
    ("outdoor",    r"pneumatic|dual|traction|snow|ice|rough\s*terrain|yard|gravel|dirt"),
    ("indoor",     r"non[-\s]?mark|cushion|solid|warehouse|smooth|polished"),
    ("handling",   r"clamp|carton|bale|drum|roll|fork\s*position|extensions?|rotator|push|pull|single\s*double|turnaload"),
]
def _auto_tags(name: str, benefit: str, subcat: str) -> set[str]:
    text = f"{name} {benefit} {subcat}".lower()
    tags = set(lbl for lbl, rx in TAG_RULES if re.search(rx, text))
    # Tires: enforce indoor/outdoor hints if the name itself says so
    if "tire" in text or "tyre" in text:
        if re.search(r"non[-\s]?mark|cushion|press[-\s]?on", text):
            tags.add("indoor")
        if re.search(r"pneumatic|solid|dual|rough|yard|gravel|dirt", text):
            tags.add("outdoor")
    return tags

def _make_code(name: str) -> str:
    s = (name or "").upper()
    s = re.sub(r"[^\w\+\s-]", " ", s)      # remove odd chars
    s = s.replace("+", " PLUS ")
    s = re.sub(r"[\s/-]+", "_", s)         # normalize separators
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:64] or "UNKNOWN_OPTION"

# ── Catalog intent regexes (also used by options_attachments_router) ──
_BOTH_PAT  = re.compile(r'\b(both\s+lists?|attachments\s+and\s+options|options\s+and\s+attachments)\b', re.I)
_ATT_PAT   = re.compile(r'\b(attachments?\s+only|attachments?)\b', re.I)
_OPT_PAT   = re.compile(r'\b(options?\s+only|options?)\b', re.I)
_TIRES_PAT = re.compile(r'\b(tires?|tyres?|tire\s*types?)\b', re.I)

# --- NEW: strict type guards so tires/attachments/options never bleed into each other ---
def _is_attachment(row: Dict[str, Any]) -> bool:
    return (row.get("Type", "") or "").strip().lower() == "attachments"

def _is_option(row: Dict[str, Any]) -> bool:
    return (row.get("Type", "") or "").strip().lower() == "options"

# ---------------- Core typed loader ----------------
def _read_catalog_df():
    """
    Reads Excel and normalizes columns to:
      - name         (from 'Name' or 'Option')
      - benefit      (from 'Benefit' | 'Desc' | 'Description' or empty)
      - type         (from 'Type' | 'Category' or inferred; values: option | attachment | tire)
      - subcategory  (optional, from 'Subcategory')

    Returns a pandas DataFrame with columns:
      ['name', 'benefit', 'type', 'subcategory']
    or None if missing.
    """
    if _pd is None or not os.path.exists(_OPTIONS_XLSX):
        print(f"[ai_logic] Excel not found or pandas missing: {_OPTIONS_XLSX}")
        return None

    try:
        df = _pd.read_excel(_OPTIONS_XLSX, engine="openpyxl")
    except Exception:
        df = _pd.read_excel(_OPTIONS_XLSX)

    if df is None or df.empty:
        print("[ai_logic] Excel read but empty.")
        return None

    # Map lowercase headers -> original headers
    cols = {str(c).lower().strip(): c for c in df.columns}
    name_col    = cols.get("name") or cols.get("option")
    benefit_col = cols.get("benefit") or cols.get("desc") or cols.get("description")
    type_col    = cols.get("type") or cols.get("category")
    subcat_col  = cols.get("subcategory")

    if not name_col:
        print("[ai_logic] Excel must have a 'Name' or 'Option' column.")
        return None

    # Work on a copy so we don't mutate the original df
    df = df.copy()

    # --- Standardize core columns ---------------------------------------
    df["__name__"] = df[name_col].astype(str).str.strip()

    if benefit_col:
        df["__benefit__"] = df[benefit_col].astype(str).str.strip()
    else:
        df["__benefit__"] = ""

    if type_col:
        df["__type__"] = df[type_col].astype(str).str.strip().str.lower()
    else:
        df["__type__"] = ""

    if subcat_col:
        df["__subcategory__"] = df[subcat_col].astype(str).str.strip()
    else:
        df["__subcategory__"] = ""

    # --- NEW: clean up Subcategory typos/spacing using _canon_subcat -----
    df["__subcategory__"] = df["__subcategory__"].map(_canon_subcat)

    # --- Normalize type labels from sheet --------------------------------
    df["__type__"] = df["__type__"].replace({
        "options": "option",
        "opt": "option",
        "option": "option",
        "attachments": "attachment",
        "att": "attachment",
        "attachment": "attachment",
        "tires": "tire",
        "tire": "tire"
    })

    # --- Fallback: infer type if missing/blank ---------------------------
    def _infer_type(nm: str, tp: str) -> str:
        if tp:
            return tp
        ln = (nm or "").lower()
        # treat common clamp/sideshift/etc. as attachments
        if any(k in ln for k in (
            "clamp","sideshift","side shift","side-shift","positioner","rotator",
            "boom","pole","ram","fork extension","extensions","push/ pull","push/pull",
            "slip-sheet","slipsheet","bale","carton","drum","load stabilizer","inverta"
        )):
            return "attachment"
        # treat anything with tire language as tire
        if any(k in ln for k in ("tire","tyre","pneumatic","cushion","non-mark","dual")):
            return "tire"
        # otherwise it's a regular option
        return "option"

    df["__type__"] = df.apply(
        lambda r: _infer_type(r["__name__"], r["__type__"]),
        axis=1
    )

    # --- Final shape: drop blanks, return normalized columns -------------
    out = df.loc[
        df["__name__"] != "",
        ["__name__", "__benefit__", "__type__", "__subcategory__"]
    ].rename(
        columns={
            "__name__": "name",
            "__benefit__": "benefit",
            "__type__": "type",
            "__subcategory__": "subcategory"
        }
    )

    return out

# ---------------- Query intent + filtering ------------------------------------

def _interpret_query(q: str) -> dict:
    """
    Pull lightweight intent from free text.
    Returns dict with:
      wanted_types: set of {'attachment','option','tire'} or empty for 'auto'
      subcategory: canonicalized subcategory string or ''
      keywords: set of free-text tokens (lowercased) used to bias selection
      list_all: True if user asked to list everything for a type
    """
    txt = (q or "").strip()
    low = txt.lower()

    wanted = set()
    # Hard type hints
    if re.search(r"\battachment(s)?\b", low):
        wanted.add("attachment")
    if re.search(r"\boption(s)?\b", low):
        wanted.add("option")
    if re.search(r"\btire(s)?\b|\btyre(s)?\b", low):
        wanted.add("tire")

    # Specific phrasing like "which attachment"
    if re.search(r"\bwhich\s+attachment\b", low):
        wanted = {"attachment"}

    # Subcategory only queries (e.g., "show visibility subcategory only")
    subcat = ""
    m = re.search(r"\bsubcategory\s*:\s*([^\n\r]+)", low)  # subcategory: X
    if m:
        subcat = m.group(1).strip()
    else:
        m2 = re.search(r"\b(show|only|just)\s+([a-z][a-z\s\-\/]+)\s+subcategory\b", low)
        if m2:
            subcat = m2.group(2).strip()

    # Clean/canonicalize subcategory with your helper
    try:
        subcat = _canon_subcat(subcat)
    except Exception:
        pass

    # "list all ..." intent
    list_all = bool(re.search(r"\blist\s+all\b", low))

    # Keywords for relevance bias (avoid single stopwords)
    tokens = set(t for t in re.findall(r"[a-z0-9\-+/]{3,}", low))

    return {
        "wanted_types": wanted,   # empty means auto
        "subcategory": subcat,    # '' means ignore
        "keywords": tokens,
        "list_all": list_all,
    }

def _score_row(name: str, benefit: str, kw: set[str]) -> float:
    """Very small relevance bump from keyword overlaps."""
    blob = f"{name} {benefit}".lower()
    score = 0.0
    for t in kw:
        if t in blob:
            score += 1.0
    # Prefer exact phrase “paper roll” when present, etc.
    if "paper" in kw and "roll" in kw and "roll" in blob:
        score += 2.0
    return score

@lru_cache(maxsize=1)
def load_catalogs() -> tuple[dict, dict, dict]:
    """
    Returns three dictionaries keyed by exact 'name':
      - options:     { name: benefit }
      - attachments: { name: benefit }
      - tires:       { name: benefit }

    Notes:
    - Enforces allowed types only ('option', 'attachment', 'tire')
    - Trims whitespace
    - De-dupes by name case-insensitively (last row wins)
    """
    df = _read_catalog_df()
    if df is None or df.empty:
        print("[ai_logic] load_catalogs(): 0/0/0 (Excel missing or empty)")
        return {}, {}, {}

    # Normalize columns we depend on
    required_cols = {"name", "type", "benefit"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"[ai_logic] Missing required columns in catalog: {missing}")

    # normalize/trim
    df = df.copy()
    df["name"]    = df["name"].astype(str).str.strip()
    df["benefit"] = df["benefit"].fillna("").astype(str).str.strip()
    df["type"]    = df["type"].astype(str).str.strip().str.lower()

    # allow only known buckets
    allowed = {"option", "attachment", "tire"}
    df = df[df["type"].isin(allowed)]

    # de-dupe per bucket by name (case-insensitive): keep last
    def bucket_dict(t: str) -> dict[str, str]:
        sub = df[df["type"] == t]
        sub = sub.loc[~sub["name"].str.lower().duplicated(keep="last")]
        return {row["name"]: row["benefit"] for _, row in sub.iterrows()}

    options     = bucket_dict("option")
    attachments = bucket_dict("attachment")
    tires       = bucket_dict("tire")

    print(f"[ai_logic] load_catalogs(): options={len(options)} attachments={len(attachments)} tires={len(tires)}")
    return options, attachments, tires

# ─────────────────────────────────────────────────────────────────────────────
# Context → Recommendation selector (DROP-IN)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SelectorOutput:
    tire_primary: List[Tuple[str, str]]          # [(name, benefit)]
    attachments_top: List[Tuple[str, str]]       # [(name, benefit)]
    options_top: List[Tuple[str, str]]           # [(name, benefit)]
    debug: Dict[str, Any]                        # trace (optional to show)

# Simple keyword buckets you can expand any time
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

# Map environment → tire rule (1 best pick).
_TIRE_RULES = [
    ({"indoor"}, lambda names: _pick_any(names, ["Non-Marking Tires","Non-Marking Cushion","NM Cushion","Non-Marking Pneumatic","NM Pneumatic","Solid Tires"])),
    ({"indoor","pedestrians"}, lambda names: _pick_any(names, ["Non-Marking Tires","Non-Marking Cushion","NM Cushion","Solid Tires"])),
    ({"tight"}, lambda names: _pick_any(names, ["Non-Marking Tires","Solid Tires","Non-Marking Cushion","NM Cushion"])),
    ({"debris"}, lambda names: _pick_any(names, ["Solid Tires","Dual Solid Tires"])),
    ({"soft_ground"}, lambda names: _pick_any(names, ["Dual Tires","Dual Solid Tires"])),
    ({"cold"}, lambda names: _pick_any(names, ["Solid Tires","Non-Marking Tires","Non-Marking Pneumatic","NM Pneumatic"])),
    (set(), lambda names: _pick_any(names, ["Solid Tires","Non-Marking Tires","Dual Tires","Dual Solid Tires"]))
]

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

def _pick_any(available_names: List[str], preferred_order: List[str]) -> List[str]:
    avail_lower = {n.lower(): n for n in available_names}
    for want in preferred_order:
        hit = avail_lower.get(want.lower())
        if hit:
            return [hit]
    return [available_names[0]] if available_names else []

def _rank_items(bucket: Dict[str, str], tags: set, subcats: Dict[str, str]) -> List[Tuple[str, float]]:
    out = []
    for name, benefit in bucket.items():
        score = 0.0
        display = f"{name} {benefit}".lower()

        # Subcategory signal (use normalized 'subcategory')
        sub = (subcats.get(name) or "").strip()
        if sub:
            for tag, tagset in _SUBCAT_HINTS.items():
                if sub.lower().startswith(tag.lower()):
                    for t in tagset:
                        if t in tags:
                            score += 2.5

        # Keyword/name/benefit boosts
        for key, tag_list in _ITEM_BOOSTS.items():
            if key.lower() in display:
                for t in tag_list:
                    if t in tags:
                        score += 1.5

        if "pedestrian" in display and "pedestrians" in tags:
            score += 1.2
        if "cold" in display and "cold" in tags:
            score += 1.2
        if "rain" in display and "wet" in tags:
            score += 1.2
        if any(w in display for w in ["protection","guard","shield","screen"]) and ("debris" in tags or "outdoor" in tags):
            score += 1.2

        if benefit:
            score += 0.05

        out.append((name, score))
    out.sort(key=lambda x: x[1], reverse=True)
    return out

def recommend_from_query(
    query: str,
    *,
    top_attachments: int = 5,
    top_options: int = 5,
) -> SelectorOutput:
    options, attachments, tires = load_catalogs()

    # We also need Subcategory to improve ranking → read once here
    df = _read_catalog_df() or None
    subcats: Dict[str,str] = {}
    if df is not None and not df.empty and "subcategory" in df.columns and "name" in df.columns:
        sub = df[["name","subcategory"]].copy()
        sub["name"] = sub["name"].astype(str).str.strip()
        subcats = {r["name"]: (r["subcategory"] or "") for _, r in sub.iterrows()}

    tags = _classify(query)

    # 1) Pick ONE best tire (rule-based)
    tire_names = list(tires.keys())
    best_tire_list: List[str] = []
    for req_tags, chooser in _TIRE_RULES:
        if req_tags and not req_tags.issubset(tags):
            continue
        best_tire_list = chooser(tire_names)
        if best_tire_list:
            break
    if not best_tire_list and tire_names:
        best_tire_list = [tire_names[0]]

    tire_primary: List[Tuple[str,str]] = [(n, tires.get(n,"")) for n in best_tire_list]

    # 2) Score & trim attachments and options
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

# ---------------- Legacy-compatible helpers (used by endpoints/router) --------
@lru_cache(maxsize=1)
def load_options() -> List[dict]:
    """
    Legacy shape used by /api/options:
      Returns list of dicts: [{code, name, benefit}]
      NOTE: This includes both Options and Tires so your UI can categorize tires
            with your existing 'infer_category' logic.
    """
    options, _, tires = load_catalogs()
    rows = []
    for name, ben in {**options, **tires}.items():
        rows.append({"code": _make_code(name), "name": name, "benefit": ben})
    return rows

@lru_cache(maxsize=1)
def load_attachments() -> List[dict]:
    """List of attachments as [{name, benefit}] for the router."""
    _, attachments, _ = load_catalogs()
    return [{"name": n, "benefit": b} for n, b in attachments.items()]

@lru_cache(maxsize=1)
def load_tires_as_options() -> List[dict]:
    """Tires shaped like options so the router can merge them if needed."""
    _, _, tires = load_catalogs()
    return [{"code": _make_code(n), "name": n, "benefit": b} for n, b in tires.items()]

# Some versions of the router import load_tires(), so keep a thin alias:
load_tires = load_tires_as_options

@lru_cache(maxsize=1)
def options_lookup_by_name() -> dict:
    """Lowercase name -> row dict, for quick lookups across options+tires."""
    return {o["name"].lower(): o for o in load_options()}

def option_benefit(name: str) -> Optional[str]:
    """Convenience: get the benefit sentence for an exact option/tire name."""
    row = options_lookup_by_name().get((name or "").lower())
    return row["benefit"] if row else None

@lru_cache(maxsize=1)
def load_catalog_rows() -> list[dict]:
    """
    Return every Excel row as dicts with keys:
      name, benefit, type, subcategory
    """
    df = _read_catalog_df()
    if df is None or df.empty:
        return []
    return df.to_dict(orient="records")

# === Sales Catalog helpers ===============================================

def _env_tags_for_name(nl: str) -> list[str]:
    """
    Best-use environment tags inferred from the option/attachment name.
    Deterministic so reps get predictable tags. Used later by the catalog mode.
    """
    nl = (nl or "").lower()
    tags = set()

    # Tires
    if "tire" in nl:
        if "non-mark" in nl:
            tags.add("Indoor • clean floors")
        if "dual" in nl:
            tags.add("Mixed • stability/ramps")
        if "solid" in nl:
            tags.add("Outdoor • rough/debris")
        if not tags:
            tags.add("General use")

    # Lighting / pedestrian awareness
    if "light" in nl or "beacon" in nl or "blue spot" in nl or "red side" in nl:
        tags.update(["Pedestrian-heavy", "Low-light / 2nd shift"])
    if "radar" in nl or "ops" in nl:
        tags.add("Safety/Policy")

    # Cab / climate
    if any(k in nl for k in ["heater","cab","windshield","wiper","air conditioner"]):
        tags.add("Outdoor weather")
    if "cold storage" in nl:
        tags.add("Cold storage / freezer")

    # Filtration / protection
    if any(k in nl for k in ["radiator","belly pan","screen","air filter","pre air"]):
        tags.add("Dust/debris sites")

    # Hydraulics / controls / seats
    if "valve" in nl or "finger control" in nl:
        tags.update(["Multi-function attachments", "Ergonomics"])
    if "seat" in nl:
        tags.add("Long shifts / ergonomics")

    # Fuel / LPG
    if "lpg" in nl or "fuel" in nl:
        tags.add("IC/LPG operations")

    # Attachments
    if "sideshift" in nl or "side shift" in nl:
        tags.update(["Tight aisles", "Frequent pallet alignment"])
    if "positioner" in nl:
        tags.update(["Mixed pallet widths", "Fast changeovers"])
    if "paper roll clamp" in nl:
        tags.add("Paper/converter plants")
    if "push" in nl and "slip" in nl:
        tags.add("Slip-sheeted goods (no pallets)")
    if "carpet" in nl or "ram" in nl:
        tags.add("Rolled goods")
    if "extension" in nl:
        tags.add("Occasional long loads")

    if not tags:
        tags.add("General use")
    return sorted(tags)

_ATTACHMENT_KEYS = [
    "sideshift","side shift","fork positioner","positioner","clamp","rotator",
    "push/pull","push pull","slip-sheet","slipsheet","bale","carton","appliance",
    "drum","jib","boom","fork extension","extensions","spreader","multi-pallet",
    "double pallet","triple pallet","roll clamp","paper roll","coil ram",
    "carpet pole","layer picker","pole"
]

def _infer_category_from_name(n: str) -> str:
    n = (n or "").lower()
    if "tire" in n: return "Tires"
    if "valve" in n or "finger control" in n: return "Hydraulics / Controls"
    if any(k in n for k in ["light", "beacon", "radar", "ops", "blue spot", "red side"]):
        return "Lighting / Safety"
    if any(k in n for k in ["seat", "cab", "windshield", "wiper", "heater", "air conditioner", "rain-proof"]):
        return "Cab / Comfort"
    if any(k in n for k in ["radiator", "screen", "belly pan", "protection bar", "fan"]):
        return "Protection / Cooling"
    if "brake" in n: return "Braking"
    if "fics" in n or "fleet" in n: return "Telematics"
    if "cold storage" in n: return "Environment"
    if "lpg" in n or "fuel" in n: return "Fuel / LPG"
    if any(k in n for k in ["overhead guard", "lifting eyes"]): return "Chassis / Structure"
    return "Other"

def _options_iter():
    """
    Yield normalized rows with a 'category' that prefers the Excel Subcategory,
    falling back to name-based inference when blank.
    """
    for r in load_catalog_rows():  # has: name, benefit, type, subcategory
        name = r.get("name", "")
        benefit = r.get("benefit", "")
        subcat = (r.get("subcategory") or "").strip()
        category = subcat if subcat else _infer_category_from_name(name)
        yield {
            "code": _make_code(name),
            "name": name,
            "benefit": benefit,
            "category": category,
            "lname": name.lower(),
        }

def _score_option_for_needs(opt: dict, want: dict) -> float:
    s = 0.0
    name = opt["lname"]
    indoor, outdoor = want.get("indoor"), want.get("outdoor")
    cap = (want.get("cap_lbs") or 0)
    power_pref = (want.get("power_pref") or "")

    if opt["category"] == "Tires":
        if indoor and outdoor:
            if "solid" in name and "dual" not in name and "non-mark" not in name: s += 5.0
            if "dual" in name: s += 2.0 if cap >= 8000 else -0.5
            if "non-mark" in name: s -= 0.8
        elif outdoor and not indoor:
            if "solid" in name: s += 4.0
            if "dual" in name and cap >= 8000: s += 2.5
            if "non-mark" in name: s -= 1.5
        elif indoor and not outdoor:
            if "non-mark" in name: s += 4.0
            if "cushion" in name or "press-on" in name: s += 2.0
            if "solid pneumatic" in name or "pneumatic" in name: s -= 1.0
        else:
            if "solid" in name and "non-mark" not in name: s += 2.0

    if opt["category"] == "Hydraulics / Controls":
        if any(k in name for k in ["4valve","4-valve","4 valve","5 valve","5-valve"]): s += 3.5
        if any(k in name for k in ["3valve","3-valve","3 valve"]): s += 2.0
        if "finger control" in name: s += 1.0
        if "msg65" in name: s += 0.8

    if opt["category"] == "Lighting / Safety":
        if "blue spot" in name or "red side" in name: s += 2.5 if indoor or (indoor and outdoor) else 1.0
        if "rotating" in name or "beacon" in name: s += 1.5
        if "rear working light" in name: s += 1.5 if outdoor or (indoor and outdoor) else 0.8
        if "radar" in name or "ops" in name: s += 1.2

    if opt["category"] == "Cab / Comfort":
        if "msg65" in name or "suspension seat" in name: s += 2.0 if cap >= 6000 else 1.0
        if "heater" in name or "air conditioner" in name: s += 1.5 if outdoor or (indoor and outdoor) else 0.5
        if "cab" in name or "windshield" in name or "rain-proof" in name: s += 1.2 if outdoor or (indoor and outdoor) else 0.3

    if opt["category"] == "Protection / Cooling":
        if outdoor or (indoor and outdoor): s += 1.8
        if "radiator" in name or "screen" in name or "fan" in name: s += 0.6
        if "belly pan" in name or "protection bar" in name: s += 0.6

    if opt["category"] == "Braking":
        if cap >= 8000: s += 2.0

    if "speed control" in name and power_pref == "diesel":
        s -= 5.0

    return s

# --- Fallback flag extractor (only if you don't already have one) ---
if '_need_flags_from_text' not in globals():
    def _need_flags_from_text(user_q: str) -> dict:
        t = (user_q or "").lower()
        f = {}

        # Environment / duty
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

        # Tires (one-line versions you asked for)
        f["non_marking"]   = bool(re.search(r'non[-\s]?mark|no\s*marks?|black\s*marks?|avoid\s*marks?|scuff', t))

        # Work content → attachments
        f["alignment_frequent"] = re.search(r'align|line\s*up|tight\s*aisles|staging', t) is not None
        f["varied_width"]       = bool(re.search(r'vary|mixed\s*pallet|different\s*width|multiple\s*widths|mix\s*of\s*\d+\s*["in]?\s*(and|&)\s*\d+\s*["in]?\s*pallets?', t))
        f["paper_rolls"]        = re.search(r'paper\s*roll|newsprint|tissue', t) is not None
        f["slip_sheets"]        = re.search(r'slip[-\s]?sheet', t) is not None
        f["carpet"]             = "carpet" in t or "textile" in t
        f["long_loads"]         = bool(re.search(r'long|oversize|over[-\s]?length|overhang|\b\d+\s*[- ]?ft\b|\b\d+\s*foot\b|\b\d+\s*feet\b|crate[s]?', t))
        f["weighing"]           = re.search(r'weigh|scale|check\s*weight', t) is not None

        # Visibility / safety
        f["pedestrian_heavy"]   = re.search(r'pedestrian|foot\s*traffic|busy|congested|blind\s*corner|walkway', t) is not None
        f["poor_visibility"]    = re.search(r'low\s*light|dim|night|second\s*shift|poor\s*lighting', t) is not None

        # Hydraulics / functions
        f["extra_hydraulics"]   = "4th function" in t or "fourth function" in t
        f["multi_function"]     = "multiple clamp" in t or "multiple attachments" in t
        f["ergonomics"]         = re.search(r'ergonomic|fatigue|wrist|reach|comfort', t) is not None

        # Climate / environment
        f["cold"]               = re.search(r'cold|freezer|refrigerated|winter', t) is not None
        f["hot"]                = re.search(r'hot|heat|summer|foundry|high\s*ambient', t) is not None

        # Policy / compliance
        f["speed_control"]      = re.search(r'limit\s*speed|speeding|zoned\s*speed', t) is not None
        f["ops_required"]       = re.search(r'ops|operator\s*presence|osha|insurance|audit|policy', t) is not None

        # Misc site/config
        f["tall_operator"]      = "tall operator" in t or "headroom" in t
        f["high_loads"]         = re.search(r'high\s*mast|tall\s*stacks|top\s*heavy|elevated', t) is not None
        f["special_color"]      = "special color" in t or "paint" in t
        f["rigging"]            = "rigging" in t or "lift with crane" in t
        f["telematics"]         = "fics" in t or "fleet management" in t or "telematics" in t

        # Power hints
        f["power_lpg"]          = re.search(r'\b(lpg|propane|lp[-\s]?gas)\b', t) is not None
        f["electric"]           = re.search(r'\b(lithium|li[-\s]?ion|electric|battery)\b', t) is not None
        return f

# === REPLACEMENT: Excel-driven tires / attachments / options recommender ===

def _norm(s: str) -> str:
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
        if not nm_sheet:
            continue
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
    elif "solid tire" in nlow or (("solid" in nlow) and ("tire" in nlow)):
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

def _pick_tire_from_flags(flags: Dict[str, Any], excel_lut: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    f = {k: bool(flags.get(k)) for k in [
        "non_marking", "rough", "debris", "puncture",
        "outdoor", "indoor", "soft_ground", "yard", "gravel", "mixed",
        "heavy_loads", "long_runs", "high_loads"
    ]}

    heavy_or_stability = f["heavy_loads"] or f["high_loads"]

    if f["non_marking"]:
        if f["outdoor"] or f["mixed"] or f["yard"] or heavy_or_stability:
            return _get_with_default(
                excel_lut, "Non-Marking Dual Tires",
                "Dual non-marking tread — keeps floors clean with extra footprint for stability."
            )
        return _get_with_default(
            excel_lut, "Non-Marking Tires",
            "Non-marking compound — prevents black marks on painted/epoxy floors."
        )

    if f["puncture"] or f["debris"] or f["rough"] or f["gravel"]:
        if f["soft_ground"] or f["heavy_loads"]:
            return _get_with_default(
                excel_lut, "Dual Solid Tires",
                "Puncture-proof dual solids — added footprint and stability on rough/soft ground."
            )
        return _get_with_default(
            excel_lut, "Solid Tires",
            "Puncture-proof solid tires — best for debris-prone or rough surfaces."
        )

    if f["yard"] or f["soft_ground"] or f["outdoor"] or f["mixed"]:
        return _get_with_default(
            excel_lut, "Dual Tires",
            "Wider footprint for traction and stability on soft or uneven ground."
        )

    if f["indoor"] and not f["outdoor"]:
        return _get_with_default(
            excel_lut, "Non-Marking Tires",
            "Indoor warehouse floors — non-marking avoids scuffs on concrete/epoxy."
        )
    if f["outdoor"] and not f["indoor"]:
        return _get_with_default(
            excel_lut, "Solid Tires",
            "Outdoor pavement/yard — reduced flats and lower maintenance."
        )
    return _get_with_default(
        excel_lut, "Dual Tires",
        "Mixed or unspecified environment — dual improves footprint and stability."
    )

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
            maybe_add([("Sideshifter", "Aligns loads without repositioning the truck—faster, cleaner placement.")])
        if flags.get("varied_width"):
            maybe_add([("Fork Positioner", "Adjust fork spread from the seat for mixed pallet sizes.")])
        return out[:max_items]

    if flags.get("alignment_frequent"):
        maybe_add([("Sideshifter", "Aligns loads without repositioning the truck—faster, cleaner placement.")])
    if flags.get("varied_width"):
        maybe_add([("Fork Positioner", "Adjust fork spread from the seat for mixed pallet sizes.")])
    if flags.get("paper_rolls"):
        maybe_add([("Paper Roll Clamp", "Secure, damage-reducing handling for paper rolls.")])
    if flags.get("slip_sheets"):
        maybe_add([("Push/ Pull (Slip-Sheet)", "Handles slip-sheeted cartons—eliminates pallets and cuts freight weight.")])
    if flags.get("carpet"):
        maybe_add([("Carpet Pole", "Safe handling of carpet or coil-like rolled goods.")])
    if flags.get("long_loads"):
        maybe_add([("Fork Extensions", "Supports longer or over-length loads safely.")])

    if not out and (flags.get("indoor") or pallets_mentioned):
        maybe_add([("Sideshifter", "Aligns loads without repositioning the truck—faster, cleaner placement.")])
        if flags.get("varied_width"):
            maybe_add([("Fork Positioner", "Adjust fork spread from the seat for mixed pallet sizes.")])

    return out[:max_items]

def _pick_options_from_excel(flags: Dict[str, Any], excel_lut: Dict[str, Dict[str, Any]], max_items: int = 6) -> List[Dict[str, Any]]:
    picks: List[Dict[str, Any]] = []

    def add_if_present(name: str, default_benefit: str):
        row = excel_lut.get(_norm(name))
        if not row:
            return
        ben_txt = (row.get("benefit") or row.get("Benefit") or "")
        if "suspend" in ben_txt.lower():
            return
        item = _get_with_default(excel_lut, name, default_benefit)
        if all(_norm(item["name"]) != _norm(x["name"]) for x in picks):
            picks.append(item)

    if flags.get("cold"):
        climate_priority = [
            ("Panel mounted Cab", "Weather protection for outdoor duty; reduces operator fatigue."),
            ("Heater", "Keeps operator warm in cold environments; improves productivity."),
            ("Glass Windshield with Wiper", "Visibility in rain/snow; safer outdoor travel."),
            ("Top Rain-proof Glass", "Overhead visibility while shielding from precipitation."),
            ("Rear Windshield Glass", "Reduces wind/snow ingress from the rear."),
        ]
        for nm, ben in climate_priority:
            add_if_present(nm, ben)

        if flags.get("electric"):
            add_if_present("Added cost for the cold storage package (for electric forklift)",
                           "Seals/heaters for freezer rooms; consistent performance in cold aisles.")

        vis_candidates = [
            ("LED Rear Working Light", "Bright, low-draw rear work lighting for dim winter shifts."),
            ("LED Rotating Light", "High-visibility 360° beacon to alert pedestrians."),
            ("Blue Light", "Directional warning for blind corners and intersections."),
            ("Visible backward radar, if applicable", "Audible/visual alerts while reversing."),
        ]
        for nm, ben in vis_candidates:
            if len(picks) >= max_items:
                break
            add_if_present(nm, ben)

        if flags.get("ops_required"):
            add_if_present("Full OPS", "Operator presence system for safety interlocks.")
        if flags.get("speed_control") or flags.get("pedestrian_heavy"):
            add_if_present("Speed Control system (not for diesel engine)", "Limits travel speed to enhance safety.")
        return picks[:max_items]

    if flags.get("pedestrian_heavy") or flags.get("poor_visibility"):
        add_if_present("LED Rotating Light", "High-visibility 360° beacon to alert pedestrians.")
        add_if_present("Blue spot Light", "Projects a visible spot to warn pedestrians at intersections.")
        add_if_present("Red side line Light", "Creates a visual exclusion zone along the sides.")
        add_if_present("Visible backward radar, if applicable", "Audible/visual alerts while reversing.")
        add_if_present("Rear Working Light", "Improves rear visibility in dim aisles.")
        add_if_present("LED Rear Working Light", "Bright, low-draw rear lighting.")
        add_if_present("Blue Light", "Directional warning for blind corners.")

    if flags.get("outdoor") or flags.get("hot"):
        add_if_present("Panel mounted Cab", "Weather protection for outdoor duty; reduces fatigue.")
        add_if_present("Air conditioner", "Comfort in hot conditions; keeps productivity steady.")
        add_if_present("Glass Windshield with Wiper", "Rain/dust visibility.")
        add_if_present("Top Rain-proof Glass", "Keeps overhead visibility while shielding precipitation.")
        add_if_present("Rear Windshield Glass", "Shields rear from wind/dust.")

    if flags.get("rough") or flags.get("debris") or flags.get("yard") or flags.get("gravel"):
        add_if_present("Radiator protection bar", "Shields radiator from impacts/debris.")
        add_if_present("Steel Belly Pan", "Protects the underside from debris and snags.")
        add_if_present("Removable radiator screen", "Keeps fins clear; easy cleaning.")
        add_if_present("Dual Air Filter", "Improved filtration for dusty yards.")
        add_if_present("Pre air cleaner", "Cyclonic pre-cleaning extends filter life.")
        add_if_present("Air filter service indicator", "Prompts timely filter service.")
        add_if_present("Tilt or Steering cylinder boot", "Protects cylinder rods from grit.")

    if flags.get("extra_hydraulics") or flags.get("multi_function"):
        add_if_present("3 Valve with Handle", "Adds third function for attachments; simple handle control.")
        add_if_present("4 Valve with Handle", "Adds fourth hydraulic circuit for complex tools.")
        add_if_present("5 Valve with Handle", "Maximum hydraulic flexibility for specialized attachments.")
        add_if_present("Finger control system(2valve),if applicable.should work together with MSG65 seat",
                       "Compact fingertip controls; less reach/wrist strain.")
        add_if_present("Finger control system(3valve),if applicable.should work together with MSG65 seat",
                       "Fingertip precision for three hydraulic functions; reduced fatigue.")
        add_if_present("Finger control system(4valve), if applicable, should work together with MSG65 seat",
                       "Fingertip controls for complex four-function attachments.")

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

# --- scenario wrapper / public helpers -----------------------------------

def _norm_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s/+-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _kw_in(text: str, *alts: str) -> bool:
    return any(a in text for a in alts)

@lru_cache(maxsize=256)
def _scenario_profiles() -> List[Dict[str, Any]]:
    return [
        {
            "name": "indoor+polished+tight_aisles",
            "match": lambda t: (_kw_in(t, "indoor", "inside")
                                and _kw_in(t, "polished", "smooth")
                                and _kw_in(t, "tight aisle", "tight aisles", "narrow")
                                ) or _kw_in(t, "pedestrian"),
            "tire_preference": ["non-marking", "solid"],
            "attach_hints": ["sideshifter", "fork positioner", "fork extension"],
            "option_hints": ["blue light", "blue spot", "red side line", "ops"],
        },
        {
            "name": "cold_storage",
            "match": lambda t: _kw_in(t, "cold", "freezer", "cold storage", "refrigerated"),
            "tire_preference": ["solid", "non-marking"],
            "attach_hints": [],
            "option_hints": ["cold storage", "heater", "cab", "windshield", "wiper"],
        },
        {
            "name": "rough_terrain",
            "match": lambda t: _kw_in(t, "rough", "gravel", "yard", "outdoor", "construction"),
            "tire_preference": ["dual", "solid"],
            "attach_hints": ["load stabilizer", "fork extension", "lifting arm"],
            "option_hints": ["steel belly", "radiator protection", "rear working light"],
        },
        {
            "name": "paper_packaging",
            "match": lambda t: _kw_in(t, "paper", "packaging", "appliance", "roll", "carton"),
            "tire_preference": ["solid", "non-marking"],
            "attach_hints": ["carton clamp", "paper roll clamp", "fork positioner"],
            "option_hints": ["blue light", "ops"],
        },
        {
            "name": "indoor_busy",
            "match": lambda t: _kw_in(t, "indoor", "warehouse", "polished", "smooth", "busy aisle", "busy aisles"),
            "tire_preference": ["non-marking", "solid"],
            "attach_hints": ["sideshifter", "fork positioner"],
            "option_hints": ["blue light", "blue spot", "ops", "speed control"],
        },
    ]

def _pick_tire(user_text: str, tires: Dict[str, str]) -> Tuple[str, str]:
    t = _norm_text(user_text)
    tire_names = list(tires.keys())

    for prof in _scenario_profiles():
        if prof["match"](t):
            for pref in prof["tire_preference"]:
                cand = next((n for n in tire_names if pref in n.lower()), None)
                if cand:
                    return cand, tires.get(cand, "")

    if _kw_in(t, "indoor", "polished", "warehouse", "pedestrian"):
        cand = next((n for n in tire_names if "non-marking" in n.lower()), None)
        if cand:
            return cand, tires.get(cand, "")
    if _kw_in(t, "rough", "gravel", "yard", "outdoor"):
        cand = next((n for n in tire_names if "dual" in n.lower()), None)
        if cand:
            return cand, tires.get(cand, "")

    if tire_names:
        name = tire_names[0]
        return name, tires.get(name, "")
    return "", ""

def _shortlist_by_hints(hints: List[str], pool: Dict[str, str], k: int) -> List[Dict[str, str]]:
    hits: List[str] = []
    lname = {n.lower(): n for n in pool.keys()}
    for h in hints:
        h = h.lower()
        for ln, orig in lname.items():
            if h in ln and orig not in hits:
                hits.append(orig)
                if len(hits) >= k:
                    break
        if len(hits) >= k:
            break

    if len(hits) < k:
        for n in pool.keys():
            if n not in hits:
                hits.append(n)
            if len(hits) >= k:
                break

    out = []
    for n in hits[:k]:
        out.append({"name": n, "benefit": pool.get(n, "")})
    return out

def recommend_options_from_sheet(user_text: str, max_total: int = 6) -> Dict[str, Any]:
    flags = _need_flags_from_text(user_text)
    flags["_raw_text"] = user_text

    rows = load_catalog_rows()  # [{name, benefit, type, subcategory}, ...]
    excel_lut = _lut_by_name(rows)

    tire = _pick_tire_from_flags(flags, excel_lut)

    k = max_total if isinstance(max_total, int) and max_total > 0 else 6
    attachments = _pick_attachments_from_excel(flags, excel_lut, max_items=k)
    options = _pick_options_from_excel(flags, excel_lut, max_items=k)

    return {
        "tire": tire,
        "attachments": attachments,
        "options": options,
    }

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

    # ask to enumerate everything
    list_all = (
        bool(re.search(r'\b(list|show|give|display)\b.*\ball\b', t)) or
        bool(re.search(r'\b(full|complete)\b.*\b(list|catalog)\b', t)) or
        "all attachments" in t or "all options" in t or "all tires" in t or
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

# --- Intent helpers -------------------------------------------------------
_LIST_ALL_PAT = re.compile(r'\b(list|show|give|display)\s+(all|full|everything)\b', re.I)
_LIST_ALL_CATS = re.compile(r'\b(all\s+)?(attachments?|options?|tires?|tyres?)\b', re.I)

def _list_all_requested(text: str) -> bool:
    t = (text or "").lower()
    return bool(_LIST_ALL_PAT.search(t)) and bool(_LIST_ALL_CATS.search(t))

def _asks_list_all(text: str) -> bool:
    t = (text or "").lower()
    if _LIST_ALL_PAT.search(t):
        return True
    return any(kw in t for kw in [
        "all attachments", "all options", "all tires", "full list of attachments",
        "full list of options", "full list of tires"
    ])

def render_catalog_sections(user_text: str, max_per_section: int = 6) -> str:
    rec = recommend_options_from_sheet(user_text, max_total=max_per_section)

    lines = []

    # Tires
    tire = rec.get("tire")
    lines.append("Tires (recommended):")
    if tire:
        lines.append(f"- {tire.get('name','')} — {tire.get('benefit','')}")
    else:
        lines.append("- (no specific tire triggered)")

    # Attachments
    atts = (rec.get("attachments") or [])[:max_per_section]
    lines.append("Attachments (relevant):")
    if atts:
        for a in atts:
            lines.append(f"- {a.get('name','')} — {a.get('benefit','')}")
    else:
        lines.append("- (none triggered)")

    # Options
    opts = (rec.get("options") or [])[:max_per_section]
    lines.append("Options (relevant):")
    if opts:
        for o in opts:
            lines.append(f"- {o.get('name','')} — {o.get('benefit','')}")
    else:
        lines.append("- (none triggered)")

    print("[ai_logic] render_catalog_sections: SCENARIO path is active, max_per_section=", max_per_section)
    return "\n".join(lines)

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
    opts  = df[df["type"] == "option"]
    tires = df[df["type"] == "tire"]

    lines = []
    if not tires.empty:
        lines += _dump(tires, "Tires")
    if not atts.empty:
        lines += _dump(atts, "Attachments")
    if not opts.empty:
        lines += _dump(opts, "Options")
    return "\n".join(lines) if lines else "No items found in the catalog."

# --- Back-compat shim so old imports keep working ------------------------
def generate_catalog_mode_response(user_q: str, max_per_section: int = 6) -> str:
    return render_catalog_sections(user_q, max_per_section=max_per_section)

# --- maintenance: refresh Excel-driven caches ----------------------------
def refresh_catalog_caches():
    try: load_catalogs.cache_clear()
    except Exception: pass
    try: load_options.cache_clear()
    except Exception: pass
    try: load_attachments.cache_clear()
    except Exception: pass
    try: load_tires_as_options.cache_clear()
    except Exception: pass
    try: options_lookup_by_name.cache_clear()
    except Exception: pass
    try: load_catalog_rows.cache_clear()
    except Exception: pass

# ── load JSON once -------------------------------------------------------
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

print(f"[ai_logic] Loaded accounts: {len(accounts_raw)} | models: {len(models_raw)}")

# Common key aliases we’ll look for in models.json
CAPACITY_KEYS = [
    "Capacity_lbs", "capacity_lbs", "Capacity", "Rated Capacity", "Load Capacity",
    "Capacity (lbs)", "capacity", "LoadCapacity", "capacityLbs", "RatedCapacity",
    "Load Capacity (lbs)", "Rated Capacity (lbs)"
]
HEIGHT_KEYS = [
    "Lift Height_in", "Max Lift Height (in)", "Lift Height", "Max Lift Height",
    "Mast Height", "lift_height_in", "LiftHeight", "Lift Height (in)", "Mast Height (in)"
]
AISLE_KEYS = [
    "Aisle_min_in", "Aisle Width_min_in", "Aisle Width (in)", "Min Aisle (in)",
    "Right Angle Aisle (in)", "Right-Angle Aisle (in)", "RA Aisle (in)"
]
POWER_KEYS = ["Power", "power", "Fuel", "fuel", "Drive", "Drive Type", "Power Type", "PowerType"]
TYPE_KEYS  = ["Type", "Category", "Segment", "Class", "Class/Type", "Truck Type"]

# ── account helpers ------------------------------------------------------
def get_account(text: str) -> Optional[Dict[str, Any]]:
    low = text.lower()
    for acct in accounts_raw:  # substring pass
        name = str(acct.get("Account Name","")).lower()
        if name and name in low:
            return acct
    names = [a.get("Account Name","") for a in accounts_raw if a.get("Account Name")]
    close = difflib.get_close_matches(text, names, n=1, cutoff=0.7)
    if close:
        return next(a for a in accounts_raw if a.get("Account Name") == close[0])
    return None

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

# ── parsing & normalization helpers -------------------------------------
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

def _num(s: Any) -> Optional[float]:
    if s is None: return None
    m = re.search(r"-?\d+(?:\.\d+)?", str(s))
    return float(m.group(0)) if m else None

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

def _text_from_keys(row: Dict[str,Any], keys: List[str]) -> str:
    for k in keys:
        v = row.get(k)
        if v:
            return str(v)
    return ""

def _is_reach_or_vna(row: Dict[str,Any]) -> bool:
    t = (_text_from_keys(row, TYPE_KEYS) + " " + str(row.get("Model","")) + " " + str(row.get("Model Name",""))).lower()
    if any(word in t for word in ["reach", "vna", "order picker", "turret"]):
        return True
    return bool(re.search(r"\b(cqd|rq|vna)\b", t))

def _is_three_wheel(row: Dict[str, Any]) -> bool:
    drive = (str(row.get("Drive Type", "")) + " " + str(row.get("Drive", ""))).lower()
    model = (str(row.get("Model","")) + " " + str(row.get("Model Name",""))).lower()
    return ("three wheel" in drive) or ("3-wheel" in drive) or (" 3 wheel" in drive) or ("sq" in model)

def _power_matches(pref: Optional[str], powr_text: str) -> bool:
    if not pref:
        return True
    p = pref.lower()
    t = (powr_text or "").lower()
    if p == "electric":
        return any(x in t for x in ("electric", "lithium", "li-ion", "lead", "battery"))
    if p == "lpg":
        return any(x in t for x in ("lpg", "propane", "lp gas", "gas"))
    if p == "diesel":
        return "diesel" in t
    return p in t

# --- robust capacity intent parser (improved) -----------------------------
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

    def _n(s: str) -> float: return float(s.replace(",", ""))

    m = re.search(rf'(?:{KNUM}|{NUM})\s*\+\s*(?:{UNIT_LB})?', t)
    if m:
        val = m.group(1) or m.group(2)
        v = float(val.replace(",", ""))
        if m.group(1) is not None:
            return (int(round(v*1000)), None)
        return (int(round(v)), None)

    m = re.search(rf'{KNUM}\s*-\s*{KNUM}', t)
    if m:
        lo, hi = int(round(_n(m.group(1))*1000)), int(round(_n(m.group(2))*1000))
        return (min(lo, hi), max(lo, hi))

    m = re.search(rf'{NUM}\s*-\s*{NUM}\s*(?:{UNIT_LB})?', t)
    if m:
        a, b = int(round(_n(m.group(1)))), int(round(_n(m.group(2))))
        return (min(a, b), max(a, b))

    m = re.search(rf'between\s+{NUM}\s+and\s+{NUM}', t)
    if m:
        a, b = int(round(_n(m.group(1)))), int(round(_n(m.group(2))))
        return (min(a, b), max(a, b))

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

    m = re.search(rf'\b(\d[\d,]{{3,5}})\s*(?:{UNIT_LB})\b', t)
    if m: return (int(m.group(1).replace(",", "")), None)

    near = re.search(rf'(?:{LOAD_WORDS})\D{{0,12}}(\d{{4,5}})\b', t)
    if near:
        return (int(near.group(1)), None)

    return (None, None)

# --- parse requirements ---------------------------------------------------
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
            height_in = _to_inches(float(raw.replace(",","")), "ft") if unit in ("ft","feet","'") \
                        else float(raw.replace(",",""))
            break
        except:
            pass
    if height_in is None:
        m = re.search(r'(?:lift|raise|reach|height|clearance|mast)\D{0,12}(\d[\d,\.]*)\s*(ft|feet|\'|in|\"|inches)', ql)
        if m:
            raw, unit = m.group(1), m.group(2)
            try:
                height_in = _to_inches(float(raw.replace(",","")), "ft") if unit in ("ft","feet","'") \
                            else float(raw.replace(",",""))
            except:
                height_in = None

    aisle_in = None
    m = re.search(r'(?:aisle|aisles|aisle width)\D{0,12}(\d[\d,\.]*)\s*(?:in|\"|inches|ft|\')', ql)
    if m:
        raw, unitblob = m.group(1), m.group(0)
        try:
            aisle_in = _to_inches(float(raw.replace(",","")), "ft") if ("ft" in unitblob or "'" in unitblob) \
                       else float(raw.replace(",",""))
        except: pass
    if aisle_in is None:
        m = re.search(r'(?:right[-\s]?angle(?:\s+aisle|\s+stack(?:ing)?)?|ra\s*aisle|ras)\D{0,12}(\d[\d,\.]*)\s*(in|\"|inches|ft|\')', ql)
        if m:
            raw, unit = m.group(1), m.group(2)
            try:
                aisle_in = _to_inches(float(raw.replace(",","")), "ft") if unit in ("ft","'") \
                           else float(raw.replace(",",""))
            except: pass

    power_pref = None
    if any(w in ql for w in ["zero emission","zero-emission","emissions free","emissions-free","eco friendly",
                             "eco-friendly","green","battery powered","battery-powered","battery","lithium",
                             "li-ion","li ion","lead acid","lead-acid","electric"]):
        power_pref = "electric"
    if "diesel" in ql: power_pref = "diesel"
    if any(w in ql for w in ["lpg","propane","lp gas","gas (lpg)","gas-powered","gas powered"]): power_pref = "lpg"

    indoor  = any(w in ql for w in ["indoor","warehouse","inside","factory floor","distribution center","dc"])
    outdoor = any(w in ql for w in ["outdoor","yard","dock yard","construction","lumber yard","gravel","dirt",
                                    "uneven","rough","pavement","parking lot","rough terrain","rough-terrain"])
    narrow  = ("narrow aisle" in ql) or ("very narrow" in ql) or ("vna" in ql) or ("turret" in ql) \
              or ("reach truck" in ql) or ("stand-up reach" in ql) \
              or (aisle_in is not None and aisle_in <= 96)

    tire_pref = None
    if any(w in ql for w in ["non-marking","non marking","nonmarking"]): tire_pref = "non-marking cushion"
    if tire_pref is None and any(w in ql for w in ["cushion","press-on","press on"]): tire_pref = "cushion"
    if any(w in ql for w in ["pneumatic","air filled","air-filled","rough terrain tires","rt tires","knobby",
                             "off-road","outdoor tires","solid pneumatic","super elastic","foam filled","foam-filled"]):
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

# ── accessors to raw row fields -----------------------------------------
def _capacity_of(row: Dict[str,Any]) -> Optional[float]:
    return _normalize_capacity_lbs(row)

def _height_of(row: Dict[str,Any]) -> Optional[float]:
    return _normalize_height_in(row)

def _aisle_of(row: Dict[str,Any]) -> Optional[float]:
    return _normalize_aisle_in(row)

def _power_of(row: Dict[str,Any]) -> str:
    return _text_from_keys(row, POWER_KEYS).lower()

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

def _tire_guidance(want: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    t = want.get("tire_pref")
    indoor, outdoor = want.get("indoor"), want.get("outdoor")
    if t:
        if "cushion" in t and indoor and not outdoor:
            return (t, "Indoor floors favor low rolling resistance and protect finished surfaces; non-marking avoids floor scuffs.")
        if "pneumatic" in t and outdoor and not indoor:
            return (t, "Outdoor/uneven surfaces need shock absorption and traction; (solid) pneumatic handles debris and rough pavement.")
        return (t, None)
    if indoor and not outdoor:
        return ("cushion", "Indoor warehouse use → cushion tires (non-marking where floor care matters).")
    if outdoor and not indoor:
        return ("pneumatic", "Outdoor/yard work → (solid) pneumatic for stability and grip on uneven ground.")
    return (None, None)

def pick_tire_advanced(user_q: str) -> tuple[str, str]:
    t = (user_q or "").lower()

    indoorish = any(k in t for k in [
        "indoor", "inside", "warehouse", "production line", "manufacturing floor",
        "painted", "epoxy", "sealed", "polished", "smooth concrete", "food", "pharma",
        "clean floor", "cleanroom"
    ])
    outdoorish = any(k in t for k in [
        "outdoor", "yard", "dock", "dock yard", "lumber yard", "construction",
        "pavement", "asphalt", "gravel", "dirt", "parking lot"
    ])
    rough = any(k in t for k in [
        "rough terrain", "rough-terrain", "uneven", "broken", "pothole",
        "ruts", "curbs", "thresholds", "rails", "speed bumps"
    ])
    debris = any(k in t for k in [
        "debris", "nails", "screws", "scrap", "metal", "glass", "shavings", "chip"
    ])
    mixed = any(k in t for k in [
        "indoor/outdoor", "inside and outside", "both indoor and outdoor",
        "dock to yard", "mixed environment", "go outside and inside"
    ])
    nonmark_need = any(k in t for k in [
        "non-mark", "non mark", "no marks", "avoid marks", "black marks", "scuff", "no scuffs"
    ])

    stability = any(k in t for k in [
        "ramp", "ramps", "slope", "incline", "grade", "dock plate",
        "high mast", "elevated", "tall stacks", "top heavy",
        "wide loads", "long loads", "coil", "paper", "rolls"
    ])

    heavy = False
    try:
        cap_min, _ = _parse_capacity_lbs_intent(t)
        heavy = bool(cap_min and cap_min >= 7000)
    except Exception:
        pass

    if nonmark_need or (indoorish and not outdoorish and any(k in t for k in ["concrete", "painted", "polished", "epoxy", "clean"])):
        if heavy or stability or mixed:
            return ("Non-Marking Dual Tires", "Clean/painted floors with heavier or mixed-duty usage — non-marking duals add stability and reduce scuffing.")
        return ("Non-Marking Tires", "Clean indoor floors — non-marking prevents black marks and scuffs.")

    if rough or debris or (outdoorish and not indoorish and any(k in t for k in ["gravel", "dirt", "pothole", "broken"])):
        if heavy or stability or mixed:
            return ("Dual Solid Tires", "Rough/debris-prone surfaces — dual solid improves stability and is puncture-resistant.")
        return ("Solid Tires", "Rough/debris-prone surfaces — solid is puncture-proof and low maintenance.")

    if mixed or (indoorish and outdoorish):
        return ("Dual Tires", "Mixed indoor/outdoor travel — dual improves stability and footprint across surfaces.")

    if outdoorish and not indoorish:
        if heavy or stability:
            return ("Dual Solid Tires", "Outdoor duty with heavier loads or ramps — dual solid adds footprint and stability.")
        return ("Solid Tires", "Outdoor pavement — solid reduces flats and maintenance.")

    if indoorish and not outdoorish:
        if heavy or stability:
            return ("Non-Marking Dual Tires", "Indoor with higher stability needs — dual non-marking reduces scuffs and adds stability.")
        return ("Non-Marking Tires", "Indoor warehouse floors — non-marking avoids scuffing on concrete/epoxy.")

    if any(k in t for k in ["electric", "battery", "lithium"]) and any(k in t for k in ["concrete", "smooth", "painted"]):
        return ("Non-Marking Tires", "Likely indoor on smooth concrete with electric — non-marking prevents scuffs.")
    return ("Dual Tires", "Mixed/unspecified environment — dual provides added stability and versatility.")

# --- model filtering & ranking ------------------------------------------
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

        # HARD FILTERS
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

# ── build final prompt chunk --------------------------------------------
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
                chosen_tire = {
                    "name": nm_row["name"],
                    "benefit": (nm_row.get("benefit") or "Non-marking compound prevents black marks on painted/epoxy floors.")
                }
            else:
                chosen_tire = {
                    "name": "Non-Marking Tires",
                    "benefit": "Non-marking compound prevents black marks on painted/epoxy floors."
                }

    lines.append("Customer Profile:")
    lines.append(f"- Environment: {env}")
    if want.get("cap_lbs"):
        lines.append(f"- Capacity Min: {int(round(want['cap_lbs'])):,} lb")
    else:
        lines.append("- Capacity Min: Not specified")

    lines.append("\nModel:")
    if hits:
        top = hits[0]
        top_name = _safe_model_name(top)
        lines.append(f"- Top Pick: {top_name}")
        if len(hits) > 1:
            alts = [ _safe_model_name(m) for m in hits[1:5] ]
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
        lines.append("- Top pick vs peers: HELI advantages typically include tight turning (102 in).")
        lines.append("- We can demo against peers on your dock to validate turning, lift, and cycle times.")
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
    lines.append("- Is it suitable for heavy-duty tasks? — Ask: What tasks will you be performing? | Reframe: Designed for robust applications. | Proof: Tested under heavy loads. | Next: Would you like to see a demonstration?")

    lines.append(user_q)

    return "\n".join(lines)

# --- expose selected list + ALLOWED block for strict grounding -----------
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

# --- promo helpers: expose top-pick code / class / power -----------------
def _class_of(row: Dict[str, Any]) -> str:
    t = _text_from_keys(row, TYPE_KEYS)
    m = re.search(r'\bclass\s*([ivx]+)\b', (t or ""), re.I)
    if m:
        roman = m.group(1).upper()
        roman = roman.replace("V","V").replace("X","X")
        return roman
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

# Explicit public surface for importers (helps Pylance)
__all__ = [
    "load_catalogs",
    "load_options",
    "load_attachments",
    "load_tires_as_options",
    "options_lookup_by_name",
    "option_benefit",
    "load_catalog_rows",
    "recommend_from_query",
    "recommend_options_from_sheet",
    "render_catalog_sections",
    "parse_catalog_intent",
    "generate_catalog_mode_response",
    "refresh_catalog_caches",
    "filter_models",
    "generate_forklift_context",
    "select_models_for_question",
    "allowed_models_block",
    "top_pick_meta",
]
