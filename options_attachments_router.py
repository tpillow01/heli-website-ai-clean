# options_attachments_router.py
# Excel-backed, AI-selected router for "Options & Attachments"
# - Pulls from ai_logic.load_options(), load_attachments(), and (optionally) load_tires_*()
# - Uses AI to SELECT relevant items (never invents names not in your Excel)
# - NO global cap on list size; TIRES are curated to the top 1–2 unless user says "all"
# - Industry/environment profiles guide selection (e.g., lumber yard, indoor warehouse, construction, cold storage)
# - Heuristic fallbacks for outdoor/long-load scenarios when selector returns nothing
# - Strips all markdown emphasis (** __ *) from responses
# - Styles headers via <span class="section-label">…</span>

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

# Scenario profiles: positive/negative hints to steer selection
_SCENARIO_PROFILES = {
    # Outdoor + long bundles (e.g., lumber yards)
    "lumber": {
        "positive": [
            "pneumatic", "solid pneumatic", "dual tire", "dual", "traction",
            "outdoor", "rough terrain", "debris", "puncture",
            "visibility", "work light", "led", "beacon",
            "protection", "guard", "radiator screen", "belly pan",
            "long fork", "fork length", "load backrest",
            "fork extension", "positioner", "sideshift", "carpet pole", "ram"
        ],
        "avoid": ["cold storage", "freezer", "non-marking", "indoor only", "paper roll"],
    },
    # Indoor warehouse with finished floors
    "indoor": {
        "positive": ["non-marking", "cushion", "visibility", "led", "blue light", "compact"],
        "avoid": ["pneumatic", "solid pneumatic", "dual"],
    },
    # Construction / rough debris
    "construction": {
        "positive": ["solid pneumatic", "pneumatic", "dual", "belly pan", "guard", "radiator", "beacon"],
        "avoid": ["non-marking", "cushion"],
    },
    # Cold storage
    "cold storage": {
        "positive": ["cold", "freezer", "low-temp", "heater", "defroster", "enclosed cab", "non-marking"],
        "avoid": ["radiator", "hot", "cooling fan only"],
    },
    # Pipe / tubing / long
    "pipe": {
        "positive": ["fork extension", "carpet pole", "ram", "positioner", "sideshift", "load backrest", "long fork"],
        "avoid": [],
    },
}

def _scenario_hints(query: str):
    q = (query or "").lower()
    pos: List[str] = []
    neg: List[str] = []
    for key, profile in _SCENARIO_PROFILES.items():
        if key in q:
            pos.extend(profile.get("positive", []))
            neg.extend(profile.get("avoid", []))
    return list(dict.fromkeys(pos)), list(dict.fromkeys(neg))  # de-dupe, keep order

# Selector prompt: pick relevant items ONLY from provided candidates
SELECTOR_INSTRUCTION = (
    "You are selecting relevant forklift {kind} for the user's scenario.\n"
    "You can ONLY choose items from the CANDIDATES list provided. Do NOT invent anything.\n"
    "Use general forklift knowledge and the provided HINTS to select items that best fit.\n"
    "Prefer items matching POSITIVE keywords; avoid items matching AVOID keywords.\n"
    "Return JSON only with exact names from the candidates: {\"items\": [\"Name 1\", \"Name 2\", ...]}\n"
    "If nothing is relevant, return {\"items\": []}.\n"
    "\nUSER SCENARIO:\n{query}\n"
    "\nHINTS:\nPOSITIVE: {pos}\nAVOID: {neg}\n"
    "\nCANDIDATES ({count}):\n{candidates}\n"
)

def _format_candidates(d: Dict[str, str]) -> str:
    lines = []
    for name, blurb in d.items():
        if blurb:
            lines.append(f"- {name}: {blurb}")
        else:
            lines.append(f"- {name}:")
    return "\n".join(lines)

# ------------------------------- LLM helpers ---------------------------------
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

def _select_relevant(d: Dict[str, str], kind: str, query: str) -> List[str]:
    """
    Ask the model to pick relevant item names from dict d for the given query.
    Returns a list of names that MUST be subset of d.keys().
    """
    if not d:
        return []
    candidates_text = _format_candidates(d)
    pos, neg = _scenario_hints(query)
    try:
        out = _llm(
            [
                {"role":"system","content":"Return JSON only. No explanations."},
                {"role":"user","content": SELECTOR_INSTRUCTION.format(
                    kind=kind,
                    query=query,
                    pos=", ".join(pos) or "(none)",
                    neg=", ".join(neg) or "(none)",
                    count=len(d),
                    candidates=candidates_text
                )},
            ],
            model=MODEL_SELECTOR,
            temperature=0.1,
        )
        data = json.loads(out)
        names = [n for n in data.get("items", []) if isinstance(n, str)]
        # Keep only exact names that exist in d
        set_keys = {k.strip() for k in d.keys()}
        safe = [n for n in names if n.strip() in set_keys]
        return safe
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

# Strip bold/italics markers (**, __, *) anywhere in the final text
_MD_EMPH_ALL = re.compile(r"(\*\*|__|\*)(.*?)\1")

def _strip_all_md_emphasis(text: str) -> str:
    if not text:
        return text
    # Remove any paired emphasis markers
    s = re.sub(_MD_EMPH_ALL, r"\2", text)
    # Also clean stray single asterisks/underscores around words
    s = re.sub(r"(?<!\S)[*_]+|[*_]+(?!\S)", "", s)
    return s

# Strip **/__ around deep-dive headers specifically (before decorating)
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
    """Higher score = more relevant for the scenario."""
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
    # Cold storage: prefer non-marking compounds if present in catalog
    if "cold storage" in q or "freezer" in q:
        if "non-marking" in s:    score += 2
    return score

def _curate_tires(names: List[str], pool: Dict[str, str], query: str, list_all: bool) -> List[str]:
    """
    Keep only the 1–2 most relevant tire choices unless user asked for ALL.
    Everything else (non-tires) is left untouched by this function.
    """
    if list_all:
        return names
    tires = [(n, _score_tire(n, pool.get(n, ""), query)) for n in names if _is_tire(n, pool.get(n, ""))]
    non_tires = [n for n in names if not _is_tire(n, pool.get(n, ""))]
    # Pick the top 2 tires if any
    tires_sorted = [n for n, _ in sorted(tires, key=lambda x: (-x[1], _normalize(x[0])))]
    top_tires = tires_sorted[:2]
    # Preserve original relative order where possible
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
            "sideshift", "side shift", "sideshifter", # placement/aisle positioning
            "load backrest", "lbr",                   # tall/long bundle support
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
    # Plain names; no markdown emphasis
    lines = [f"- {n} — {d.get(n, '') or '—'}" for n in names]
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

    # BOTH lists
    if intent == "both_lists":
        opt_pool = _merged_options()
        att_pool = ATTACHMENTS

        if asked_all:
            opt_names = sorted(opt_pool.keys(), key=lambda k: _normalize(k))
            att_names = sorted(att_pool.keys(), key=lambda k: _normalize(k))
        else:
            # AI picks relevant items
            opt_names = _select_relevant(opt_pool, "options (incl. tires)", user_text)
            att_names = _select_relevant(att_pool, "attachments", user_text)

            # Heuristic fallbacks for outdoor/long if nothing selected
            pos, _neg = _scenario_hints(user_text)
            if (("lumber" in user_text.lower()) or pos) and not opt_names:
                opt_names = _heuristic_options_for_outdoor_long(opt_pool)
            if (("lumber" in user_text.lower()) or pos) and not att_names:
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
            opt_names = sorted(_merged_options().keys(), key=lambda k: _normalize(k))
            att_names = sorted(ATTACHMENTS.keys(),       key=lambda k: _normalize(k))
            out = _build_list("Options", opt_names, _merged_options()) + "\n\n" + _build_list("Attachments", att_names, ATTACHMENTS)
            return _strip_all_md_emphasis(out)
        if "option" in t:
            names = sorted(_merged_options().keys(), key=lambda k: _normalize(k))
            return _build_list("Options", names, _merged_options())
        if "attachment" in t or "catalog" in t or "everything" in t:
            names = sorted(ATTACHMENTS.keys(), key=lambda k: _normalize(k))
            return _build_list("Attachments", names, ATTACHMENTS)
        return "Do you want all options, all attachments, or both?"

    # Options-only
    if intent == "list_options":
        pool = _merged_options()
        names = sorted(pool.keys(), key=lambda k: _normalize(k)) if asked_all else _select_relevant(pool, "options (incl. tires)", user_text)
        if not asked_all:
            names = _curate_tires(names, pool, user_text, list_all=False)
        return _build_list("Options", names, pool)

    # Attachments-only
    if intent == "list_attachments":
        pool = ATTACHMENTS
        names = sorted(pool.keys(), key=lambda k: _normalize(k)) if asked_all else _select_relevant(pool, "attachments", user_text)
        return _build_list("Attachments", names, pool)

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
            # Friendly fallback if model hiccups
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
