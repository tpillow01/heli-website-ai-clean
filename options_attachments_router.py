# options_attachments_router.py
# Excel-backed, environment-aware router for "Options & Attachments"
# - Pulls from ai_logic.load_options(), load_attachments(), and (optionally) load_tires_*()
# - Hybrid selection:
#     1) Deterministic relevance scoring from your Excel names/blurbs + scenario hints
#     2) If no strong matches, LLM selector that NEVER invents items
# - NO global cap; TIRES curated to top 1–2 unless user says "all"
# - Industry/environment & pedestrian-aware profiles (indoor, lumber yard, cold storage, construction)
# - Strips all markdown emphasis (** __ *) from outputs; section headers are HTML-styled
# - Deep-dive supported

import os
import re
import json
from typing import Dict, Tuple, List
from openai import OpenAI

# ----------------------------- OpenAI setup ----------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_CLASSIFIER = os.getenv("OA_CLASSIFIER_MODEL", "gpt-4o-mini")
MODEL_RESPONDER  = os.getenv("OA_RESPONDER_MODEL",  "gpt-4o-mini")
MODEL_SELECTOR   = os.getenv("OA_SELECTOR_MODEL",   "gpt-4o-mini")

# ----------------------- Excel-backed loaders (ai_logic) ---------------------
try:
    from ai_logic import load_options as _load_options_catalog  # list[dict] with name, benefit
except Exception:
    _load_options_catalog = None

try:
    from ai_logic import load_attachments as _load_attachments_catalog
except Exception:
    _load_attachments_catalog = None

# Tire loader: either load_tires_as_options() or load_tires()
try:
    from ai_logic import load_tires_as_options as _load_tires_catalog
except Exception:
    _load_tires_catalog = None
try:
    if _load_tires_catalog is None:
        from ai_logic import load_tires as _load_tires_catalog
except Exception:
    _load_tires_catalog = None

# --------------------------- Minimal emergency fallbacks ---------------------
FALLBACK_OPTIONS = {
    "3 Valve with Handle": "Adds a third hydraulic circuit to power basic attachments.",
    "4 Valve with Handle": "Two auxiliary circuits for multi-function attachments.",
    "5 Valve with Handle": "Additional hydraulic circuits for complex attachments.",
    "Non-Marking Tires":  "Prevents scuffing on indoor finished floors.",
    "Dual Tires":         "Wider footprint for stability on soft or uneven ground.",
    "Dual Solid Tires":   "Puncture-resistant and stable in debris-prone areas.",
}
FALLBACK_ATTACHMENTS = {
    "Sideshifter":            "Shift load left/right without moving the truck.",
    "Fork Positioner":        "Adjust fork spread from the seat for mixed pallet widths.",
    "Fork Extensions":        "Temporarily lengthen forks for long/oversized loads.",
    "Paper Roll Clamp":       "Securely handle paper rolls without core damage.",
    "Push/Pull (Slip-Sheet)": "Use slip-sheets instead of pallets to cut cost/weight.",
    "Carpet Pole":            "Ram/pole for carpet, coils, and tubing.",
}
FALLBACK_TIRES = {
    "Non-Marking Tires":     "Prevents scuffing on indoor finished floors.",
    "Dual Pneumatic Tires":  "Wider footprint and flotation on soft ground.",
    "Solid Pneumatic Tires": "Puncture resistance in debris-prone areas.",
}

# ------------------------------ Catalog hydrate ------------------------------
def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

def _to_dict_by_name(rows: List[dict], name_key="name", blurb_key="benefit") -> Dict[str, str]:
    out = {}
    for r in rows or []:
        name = (r.get(name_key) or "").strip()
        if not name:
            continue
        blurb = (r.get(blurb_key) or r.get("desc") or r.get("description") or "").strip()
        out[name] = blurb
    return out

def _hydrate_catalogs() -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    # OPTIONS
    try:
        if _load_options_catalog:
            opt_rows = _load_options_catalog()
            options = _to_dict_by_name(opt_rows, name_key="name", blurb_key="benefit")
            if not options:
                options = FALLBACK_OPTIONS
        else:
            options = FALLBACK_OPTIONS
    except Exception:
        options = FALLBACK_OPTIONS

    # ATTACHMENTS
    try:
        if _load_attachments_catalog:
            att_rows = _load_attachments_catalog()
            attachments = _to_dict_by_name(att_rows, name_key="name", blurb_key="benefit")
            if not attachments:
                attachments = FALLBACK_ATTACHMENTS
        else:
            attachments = FALLBACK_ATTACHMENTS
    except Exception:
        attachments = FALLBACK_ATTACHMENTS

    # TIRES (treated as options group; included when asked for options or both)
    try:
        if _load_tires_catalog:
            tire_rows = _load_tires_catalog()
            tires = _to_dict_by_name(tire_rows, name_key="name", blurb_key="benefit")
            if not tires:
                tires = FALLBACK_TIRES
        else:
            tires = FALLBACK_TIRES
    except Exception:
        tires = FALLBACK_TIRES

    return options, attachments, tires

OPTIONS, ATTACHMENTS, TIRES = _hydrate_catalogs()

def _merged_options() -> Dict[str, str]:
    merged = dict(OPTIONS)
    for k, v in TIRES.items():
        if k not in merged:
            merged[k] = v
    return merged

# ----------------------- Fuzzy lookup for deep dives -------------------------
def fuzzy_lookup(item: str) -> Tuple[str, str, str]:
    n = _normalize(item)
    for k, v in _merged_options().items():
        if n == _normalize(k) or n in _normalize(k):
            return ("option", k, v)
    for k, v in ATTACHMENTS.items():
        if n == _normalize(k) or n in _normalize(k):
            return ("attachment", k, v)
    # Hints
    if "clamp" in n:
        k = next((kk for kk in ATTACHMENTS if "clamp" in _normalize(kk)), "Paper Roll Clamp")
        return ("attachment", k, ATTACHMENTS.get(k, ""))
    if "position" in n or "positioner" in n:
        k = next((kk for kk in ATTACHMENTS if "position" in _normalize(kk)), "Fork Positioner")
        return ("attachment", k, ATTACHMENTS.get(k, ""))
    if "side shift" in n or "sideshift" in n or "side-shift" in n:
        k = next((kk for kk in ATTACHMENTS if "side" in _normalize(kk) and "shift" in _normalize(kk)), "Sideshifter")
        return ("attachment", k, ATTACHMENTS.get(k, ""))
    if "valve" in n or "aux" in n:
        k = next((kk for kk in _merged_options() if "valve" in _normalize(kk)), "3 Valve with Handle")
        return ("option", k, _merged_options().get(k, ""))
    return ("", "", "")

# --------------------------------- Prompts -----------------------------------
SYSTEM_PROMPT = (
    "You are the Options & Attachments expert for Heli forklifts.\n"
    "STRICT RULES:\n"
    "- Answer ONLY what the user asked for.\n"
    "- If they say 'options', return options only. If they say 'attachments', return attachments only.\n"
    "- If they name a specific item, give a concise deep-dive: Purpose, Benefits, When to Use, "
    "Prerequisites/Valving, Compatibility/Capacity impacts, Trade-offs.\n"
    "- NEVER output the full catalog unless the user explicitly asks for 'all', 'catalog', or 'everything'.\n"
    "- If the user is ambiguous, ask ONE clarifying question, then stop.\n"
    "- Keep responses short and skimmable. Bullets > paragraphs.\n"
    "- Call out capacity/visibility/turning/maintenance impacts when relevant."
)

CLASSIFIER_INSTRUCTION = (
    "Classify the user request for forklift options/attachments into exactly one of:\n"
    "- list_options\n"
    "- list_attachments\n"
    "- detail_item\n"
    "- list_all\n"
    "- both_lists\n"
    "- unknown\n\n"
    "Rules:\n"
    "- If the user clearly wants BOTH (e.g., 'attachments and options'), use both_lists.\n"
    "- 'all', 'everything', or 'catalog' implies list_all for the relevant type(s) mentioned.\n\n"
    "Return JSON only: {{\"intent\":\"...\", \"item\": \"<named item or ''>\"}}\n\n"
    "User: {user_text}\n"
)

# ------------------------ Scenario profiles & scoring ------------------------
# Each profile has positive keywords (boost), avoid keywords (penalize).
# We split some boosts specific to options vs attachments for finer control.
_PROFILES = {
    "lumber": {
        "pos_any": [],  # shared
        "avoid_any": ["non-marking", "indoor only", "cold storage", "freezer", "paper roll"],
        "options_pos": [
            "pneumatic", "solid pneumatic", "dual", "traction",
            "work light", "led", "beacon",
            "protection", "guard", "radiator", "screen", "belly pan",
            "load backrest", "long fork", "fork length",
        ],
        "attachments_pos": [
            "fork extension", "extension",
            "fork positioner", "positioner",
            "sideshift", "side shift", "sideshifter",
            "carpet pole", "ram", "pole",
            "load backrest",
        ],
    },
    "indoor": {
        "pos_any": ["warehouse", "aisle"],
        "avoid_any": ["pneumatic", "solid pneumatic", "dual", "belly pan", "radiator"],
        "options_pos": ["non-marking", "cushion", "visibility", "led", "blue light", "beacon"],
        "attachments_pos": ["sideshift", "positioner"],
    },
    "construction": {
        "pos_any": ["debris", "outdoor", "rough", "yard"],
        "avoid_any": ["non-marking", "cushion"],
        "options_pos": ["solid pneumatic", "pneumatic", "dual", "belly pan", "guard", "radiator", "beacon"],
        "attachments_pos": ["sideshift", "positioner", "fork extension", "load backrest"],
    },
    "cold storage": {
        "pos_any": ["cold", "freezer", "low-temp", "low temp"],
        "avoid_any": ["radiator only", "hot"],
        "options_pos": ["cold", "freezer", "low-temp", "heater", "defroster", "enclosed cab", "non-marking"],
        "attachments_pos": ["sideshift", "positioner"],  # generic helpers; sheet-driven
    },
    "pipe": {
        "pos_any": ["pipe", "tubing"],
        "avoid_any": [],
        "options_pos": ["pneumatic", "solid pneumatic", "dual", "visibility"],
        "attachments_pos": ["carpet pole", "ram", "pole", "fork extension", "positioner", "load backrest"],
    },
}

# Pedestrian-heavy modifier (applies on top of profile)
_PEDESTRIAN_HINTS = {
    "pos": ["blue light", "blue spot", "pedestrian", "beacon", "horn", "backup handle", "mirror", "camera", "led"],
    "avoid": [],
}

def _active_profiles(query: str) -> List[str]:
    q = (query or "").lower()
    act = []
    for key in _PROFILES:
        if key in q:
            act.append(key)
    # heuristics
    if "indoor" in q or "finished floor" in q or "warehouse" in q:
        if "indoor" not in act: act.append("indoor")
    if "cold storage" in q or "freezer" in q:
        if "cold storage" not in act: act.append("cold storage")
    if "construction" in q or "rough" in q or "yard" in q or "debris" in q or "outdoor" in q:
        if "construction" not in act: act.append("construction")
    if "lumber" in q or "timber" in q:
        if "lumber" not in act: act.append("lumber")
    if "pipe" in q or "tubing" in q:
        if "pipe" not in act: act.append("pipe")
    return act

def _pedestrian_mode(query: str) -> bool:
    q = (query or "").lower()
    return any(k in q for k in ["people", "pedestrian", "busy", "foot traffic", "crowded"])

def _score_item(name: str, blurb: str, query: str, kind: str) -> float:
    """Score an item for relevance given query and kind ('options' or 'attachments')."""
    s = f"{name} {blurb}".lower()
    profs = _active_profiles(query)
    score = 0.0

    # Base boosts/penalties from profiles
    for p in profs:
        pd = _PROFILES[p]
        for kw in pd.get("pos_any", []):
            if kw in s: score += 1.0
        for kw in pd.get("avoid_any", []):
            if kw in s: score -= 1.0
        for kw in pd.get(f"{kind}_pos", []):
            if kw in s: score += 2.0

    # Pedestrian-heavy site: emphasize visibility/alerts/ergonomics; de-emphasize heavy-duty guards indoors
    if _pedestrian_mode(query):
        for kw in _PEDESTRIAN_HINTS["pos"]:
            if kw in s: score += 2.0
        if "belly pan" in s or "radiator" in s:
            score -= 1.0

    # General sanity:
    # Indoor: penalize heavy rough-terrain options; Outdoor: penalize indoor-only
    ql = query.lower()
    if any(k in ql for k in ["indoor", "finished", "warehouse"]):
        if any(k in s for k in ["pneumatic", "solid pneumatic", "dual", "belly pan", "radiator"]):
            score -= 1.0
        if "non-marking" in s or "cushion" in s:
            score += 1.0
    if any(k in ql for k in ["outdoor", "rough", "yard", "debris", "construction", "lumber", "timber"]):
        if "non-marking" in s or "cushion" in s:
            score -= 1.0
        if any(k in s for k in ["pneumatic", "solid pneumatic", "dual", "belly pan", "guard", "radiator", "beacon"]):
            score += 1.0

    return score

def _rank_by_score(pool: Dict[str, str], query: str, kind: str, min_score: float = 1.5) -> List[str]:
    """Return all items with score >= min_score, sorted by score desc then name."""
    scored = []
    for n, b in pool.items():
        sc = _score_item(n, b, query, kind)
        if sc >= min_score:
            scored.append((n, sc))
    scored.sort(key=lambda x: (-x[1], _normalize(x[0])))
    return [n for n, _ in scored]

# Selector prompt (fallback only)
SELECTOR_INSTRUCTION = (
    "You are selecting relevant forklift {kind} for the user's scenario.\n"
    "You can ONLY choose items from the CANDIDATES list provided. Do NOT invent anything.\n"
    "Use general forklift knowledge and HINTS.\n"
    "Return JSON only with exact names from the candidates: {\"items\": [\"Name 1\", \"Name 2\", ...]}\n"
    "If nothing is relevant, return {\"items\": []}.\n"
    "\nUSER SCENARIO:\n{query}\n"
    "\nHINTS:\n{hints}\n"
    "\nCANDIDATES ({count}):\n{candidates}\n"
)

def _format_candidates(d: Dict[str, str]) -> str:
    return "\n".join(f"- {n}: {d.get(n,'')}" for n in d.keys())

def _llm(messages, model: str, temperature: float = 0.2) -> str:
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages,
    )
    return resp.choices[0].message.content

def _classify(user_text: str) -> Dict[str, str]:
    try:
        out = _llm(
            [
                {"role": "system", "content": "Return JSON only. No prose."},
                {"role": "user", "content": CLASSIFIER_INSTRUCTION.format(user_text=user_text)},
            ],
            model=MODEL_CLASSIFIER,
            temperature=0.1,
        )
        data = json.loads(out)
        return {"intent": str(data.get("intent", "unknown") or "unknown"),
                "item":   str(data.get("item", "") or "")}
    except Exception:
        t = _normalize(user_text)
        if "attachment" in t and "option" in t: return {"intent":"both_lists","item":""}
        if "attachment" in t:                  return {"intent":"list_attachments","item":""}
        if "option" in t:                      return {"intent":"list_options","item":""}
        if any(w in t for w in ["catalog","everything","all","full list","complete"]):
            return {"intent":"list_all","item":""}
        return {"intent":"unknown","item":""}

def _llm_select(d: Dict[str, str], kind_label: str, query: str, hints: List[str]) -> List[str]:
    if not d:
        return []
    try:
        out = _llm(
            [
                {"role":"system","content":"Return JSON only. No explanations."},
                {"role":"user","content": SELECTOR_INSTRUCTION.format(
                    kind=kind_label,
                    query=query,
                    hints=", ".join(hints) or "(none)",
                    count=len(d),
                    candidates=_format_candidates(d)
                )},
            ],
            model=MODEL_SELECTOR,
            temperature=0.1,
        )
        data = json.loads(out)
        names = [n for n in data.get("items", []) if isinstance(n, str)]
        set_keys = {k.strip() for k in d.keys()}
        return [n for n in names if n.strip() in set_keys]
    except Exception:
        return []

# -------------------------- Output formatting --------------------------------
_HEADER_MAP = {
    "Purpose:":                        '<span class="section-label">Purpose:</span>',
    "Benefits:":                       '<span class="section-label">Benefits:</span>',
    "When to use:":                    '<span class="section-label">When to use:</span>',
    "Prerequisites/Valving:":          '<span class="section-label">Prerequisites/Valving:</span>',
    "Compatibility/Capacity impacts:": '<span class="section-label">Compatibility/Capacity impacts:</span>',
    "Trade-offs:":                     '<span class="section-label">Trade-offs:</span>',
}
_HEADER_PATTERN = re.compile(
    r'(?im)^(Purpose:|Benefits:|When to use:|Prerequisites/Valving:|Compatibility/Capacity impacts:|Trade-offs:)\s*$'
)

_MD_EMPH_ALL = re.compile(r"(\*\*|__|\*)(.*?)\1")
def _strip_all_md_emphasis(text: str) -> str:
    if not text:
        return text
    s = re.sub(_MD_EMPH_ALL, r"\2", text)
    s = re.sub(r"(?<!\S)[*_]+|[*_]+(?!\S)", "", s)
    return s

_MD_BOLD_HEADERS = re.compile(
    r'(?im)^\s*[*_]{1,3}\s*(Purpose:|Benefits:|When to use:|Prerequisites/Valving:|Compatibility/Capacity impacts:|Trade-offs:)\s*[*_]{1,3}\s*$'
)
def _strip_md_bold_headers(text: str) -> str:
    if not text:
        return text
    s = text.replace("\r\n", "\n").replace("\r", "\n")
    s = _MD_BOLD_HEADERS.sub(lambda m: m.group(1), s)
    return s

def _decorate_headers(text: str, title: str = "") -> str:
    if not text:
        return text
    s = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if title:
        s = f'<span class="section-label">{title}</span>\n{s}'
    def repl(m):
        hdr = m.group(1)
        return _HEADER_MAP.get(hdr, hdr)
    s = _HEADER_PATTERN.sub(repl, s)
    return s

# ------------------------------ Tire curation --------------------------------
_TIRE_WORDS = ("tire", "tyre", "pneumatic", "cushion")
def _is_tire(name: str, blurb: str = "") -> bool:
    s = f"{name} {blurb}".lower()
    return any(w in s for w in _TIRE_WORDS) or any(k in s for k in [
        "solid pneumatic", "non-marking", "dual"
    ])

def _score_tire(name: str, blurb: str, query: str) -> float:
    q = (query or "").lower()
    s = f"{name} {blurb}".lower()
    score = 0.0
    # Outdoor / rough / debris / lumber
    if any(k in q for k in ["lumber", "outdoor", "rough", "debris", "yard", "construction"]):
        if "solid pneumatic" in s: score += 5
        if "pneumatic" in s:      score += 4
        if "dual" in s:           score += 3
        if "non-marking" in s:    score -= 3
        if "cushion" in s:        score -= 2
    # Indoor / finished floors
    if any(k in q for k in ["indoor", "finished floor", "epoxy", "polished", "warehouse"]):
        if "non-marking" in s:    score += 5
        if "cushion" in s:        score += 4
        if "pneumatic" in s:      score -= 2
        if "solid pneumatic" in s:score -= 1
        if "dual" in s:           score -= 1
    # Cold storage: prefer non-marking if present
    if "cold storage" in q or "freezer" in q:
        if "non-marking" in s:    score += 2
    return score

def _curate_tires(names: List[str], pool: Dict[str, str], query: str, list_all: bool) -> List[str]:
    if list_all:
        return names
    tires = [(n, _score_tire(n, pool.get(n, ""), query)) for n in names if _is_tire(n, pool.get(n, ""))]
    non_tires = [n for n in names if not _is_tire(n, pool.get(n, ""))]
    tires_sorted = [n for n, _ in sorted(tires, key=lambda x: (-x[1], _normalize(x[0])))]
    top_tires = tires_sorted[:2]  # show only the top 1–2
    result = [n for n in names if n in non_tires or n in top_tires]
    return result

# ------------------------------ Heuristics -----------------------------------
def _heuristic_options_for_outdoor_long(pool: Dict[str, str]) -> List[str]:
    picks = []
    for name, blurb in pool.items():
        lower = f"{name} {blurb}".lower()
        if any(k in lower for k in [
            "solid pneumatic", "pneumatic", "dual", "traction",
            "work light", "led", "beacon",
            "protection", "guard", "radiator", "screen", "belly pan",
            "long fork", "fork length", "load backrest"
        ]):
            picks.append(name)
    return sorted(set(picks), key=lambda k: _normalize(k))

def _heuristic_attachments_for_outdoor_long(pool: Dict[str, str]) -> List[str]:
    picks = []
    for name, blurb in pool.items():
        lower = f"{name} {blurb}".lower()
        if any(k in lower for k in [
            "fork extension", "extension",            # long loads
            "fork positioner", "positioner",          # mixed widths
            "sideshift", "side shift", "sideshifter", # placement
            "load backrest", "lbr",                   # tall/long support
            "carpet pole", "ram", "pole"              # long round materials
        ]):
            picks.append(name)
    return sorted(set(picks), key=lambda k: _normalize(k))

# ------------------------------ List builders --------------------------------
def _build_list(kind_name: str, names: List[str], d: Dict[str, str]) -> str:
    header_html = f'<span class="section-label">{kind_name}:</span>'
    if not names:
        typename = kind_name.lower()
        return (
            f'{header_html}\n'
            f'- No {typename} matched your scenario. '
            f'Try asking more specifically (e.g., "{typename} for long loads", "{typename} for cold storage"), '
            f'or say "all {typename}".'
        )
    lines = [f"- {n} — {d.get(n, '') or '—'}" for n in names]  # no ** styling
    out = header_html + "\n" + "\n".join(lines)
    return _strip_all_md_emphasis(out)

def _detail_prompt(name: str, seed_blurb: str) -> str:
    return (
        "Give a concise deep-dive on this SINGLE item.\n"
        "Use the exact headers below and bullet points. Do NOT use bold or italics.\n\n"
        f"Item: {name}\n\n"
        "Purpose:\n"
        "- \n\n"
        "Benefits:\n"
        "- \n- \n- \n\n"
        "When to use:\n"
        "- \n- \n\n"
        "Prerequisites/Valving:\n"
        "- \n- \n\n"
        "Compatibility/Capacity impacts:\n"
        "- \n- \n\n"
        "Trade-offs:\n"
        "- \n- \n\n"
        f"Helpful context: {seed_blurb}\n"
    )

# ------------------------------- Public entry --------------------------------
def respond_options_attachments(user_text: str) -> str:
    if not (user_text or "").strip():
        return "Ask about options, attachments, or a specific item (e.g., Fork Positioner)."

    c = _classify(user_text)
    intent = c.get("intent", "unknown")
    item   = (c.get("item") or "").strip()
    t = _normalize(user_text)
    asked_all = any(w in t for w in [" all ", "catalog", "everything", "full list", "complete"])

    # Build pools
    opt_pool = _merged_options()
    att_pool = ATTACHMENTS

    # BOTH lists
    if intent == "both_lists":
        if asked_all:
            opt_names = sorted(opt_pool.keys(), key=lambda k: _normalize(k))
            att_names = sorted(att_pool.keys(), key=lambda k: _normalize(k))
        else:
            # 1) Deterministic scoring
            opt_names = _rank_by_score(opt_pool, user_text, "options", min_score=1.5)
            att_names = _rank_by_score(att_pool, user_text, "attachments", min_score=1.5)

            # 2) If nothing strong, LLM selector with hints
            hints = []
            for p in _active_profiles(user_text):
                pd = _PROFILES[p]
                hints.extend(pd.get("pos_any", []) + pd.get("options_pos", []) + pd.get("attachments_pos", []))
            if _pedestrian_mode(user_text):
                hints.extend(_PEDESTRIAN_HINTS["pos"])

            if not opt_names:
                opt_names = _llm_select(opt_pool, "options (incl. tires)", user_text, hints)
            if not att_names:
                att_names = _llm_select(att_pool, "attachments", user_text, hints)

            # 3) Heuristic fallbacks for outdoor/long if still empty
            if not opt_names and ("lumber" in user_text.lower() or "outdoor" in user_text.lower()):
                opt_names = _heuristic_options_for_outdoor_long(opt_pool)
            if not att_names and ("lumber" in user_text.lower() or "outdoor" in user_text.lower()):
                att_names = _heuristic_attachments_for_outdoor_long(att_pool)

            # Curate tires within options unless "all"
            opt_names = _curate_tires(opt_names, opt_pool, user_text, list_all=False)

        if not asked_all and not opt_names and not att_names:
            return "Do you want both lists, or details on a specific item?"

        parts = []
        parts.append(_build_list("Options", opt_names, opt_pool))
        parts.append(_build_list("Attachments", att_names, att_pool))
        return "\n\n".join(parts)

    # ALL for type(s)
    if intent == "list_all":
        if "option" in t and "attachment" in t:
            opt_names = sorted(opt_pool.keys(), key=lambda k: _normalize(k))
            att_names = sorted(att_pool.keys(), key=lambda k: _normalize(k))
            out = _build_list("Options", opt_names, opt_pool) + "\n\n" + _build_list("Attachments", att_names, att_pool)
            return _strip_all_md_emphasis(out)
        if "option" in t:
            names = sorted(opt_pool.keys(), key=lambda k: _normalize(k))
            return _build_list("Options", names, opt_pool)
        if "attachment" in t or "catalog" in t or "everything" in t:
            names = sorted(att_pool.keys(), key=lambda k: _normalize(k))
            return _build_list("Attachments", names, att_pool)
        return "Do you want all options, all attachments, or both?"

    # Options-only
    if intent == "list_options":
        if asked_all:
            names = sorted(opt_pool.keys(), key=lambda k: _normalize(k))
        else:
            names = _rank_by_score(opt_pool, user_text, "options", min_score=1.5)
            if not names:
                names = _llm_select(opt_pool, "options (incl. tires)", user_text, [])
            names = _curate_tires(names, opt_pool, user_text, list_all=False)
        return _build_list("Options", names, opt_pool)

    # Attachments-only
    if intent == "list_attachments":
        if asked_all:
            names = sorted(att_pool.keys(), key=lambda k: _normalize(k))
        else:
            names = _rank_by_score(att_pool, user_text, "attachments", min_score=1.5)
            if not names:
                names = _llm_select(att_pool, "attachments", user_text, [])
        return _build_list("Attachments", names, att_pool)

    # Single item deep dive
    if intent == "detail_item":
        kind, key, blurb = fuzzy_lookup(item or user_text)
        if not key:
            return "Which specific item do you mean (e.g., Fork Positioner, Sideshifter, Paper Roll Clamp)?"
        try:
            content = _llm(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": _detail_prompt(key, blurb)},
                ],
                model=MODEL_RESPONDER,
                temperature=0.2,
            )
            content = _strip_md_bold_headers(content)
            content = _decorate_headers(content, title=f"{key} — Deep Dive")
            return _strip_all_md_emphasis(content)
        except Exception:
            fallback = (
                "Purpose:\n"
                f"- {blurb or 'Attachment/option for specialized handling.'}\n\n"
                "Benefits:\n"
                "- Faster handling.\n- Safer adjustments.\n- Better load fit.\n\n"
                "When to use:\n"
                "- Mixed load profiles or frequent changes.\n\n"
                "Prerequisites/Valving:\n"
                "- May require 3rd/4th hydraulic function.\n\n"
                "Compatibility/Capacity impacts:\n"
                "- Minor capacity derate; verify on data plate.\n\n"
                "Trade-offs:\n"
                "- Higher upfront cost and modest maintenance."
            )
            fallback = _decorate_headers(fallback, title=f"{key} — Deep Dive")
            return _strip_all_md_emphasis(fallback)

    # Clarifiers
    if "attachment" in t and "option" in t:
        return "Do you want both lists, or details on a specific item?"
    if "attachment" in t:
        return "Do you want a list of attachments or details on a specific attachment?"
    if "option" in t:
        return "Do you want a list of options or details on a specific option?"
    return "Do you want options, attachments, or details on a specific item (e.g., Fork Positioner)?"
