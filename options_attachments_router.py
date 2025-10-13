# options_attachments_router.py
# Excel-backed, AI-selected router for "Options & Attachments"
# - Pulls from ai_logic.load_options(), load_attachments(), and (optionally) load_tires_*()
# - Uses AI to SELECT relevant items by general forklift knowledge + scenario/industry hints (never invents names)
# - NO global cap on list size; TIRES are curated to the top 1–2 for the scenario (unless user says "all")
# - Heuristic fallbacks for scenario-specific attachments/options when selector returns nothing
# - Strips all markdown emphasis (** __ *) from responses
# - Styles headers via <span class="section-label">…</span>

import os
import re
import json
from typing import Dict, Tuple, List, Optional
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

# ---------------------- Scenario/industry profiles ---------------------------
# These hints help the selector choose items that fit the environment/industry.
_SCENARIO_PROFILES = {
    # Outdoor + long bundles: lumber yards / timber
    "lumber": {
        "positive": [
            "pneumatic", "solid pneumatic", "dual", "traction",
            "outdoor", "rough", "debris", "puncture",
            "visibility", "work light", "led", "beacon",
            "protection", "guard", "radiator", "screen", "belly pan",
            "long fork", "fork extension", "load backrest", "positioner", "sideshift", "carpet pole"
        ],
        "avoid": ["cold storage", "freezer", "non-marking", "paper roll", "slip-sheet"],
        "attachments": ["fork extension", "positioner", "sideshift", "carpet pole", "load backrest"],
    },
    # Indoor warehouse with finished floors
    "warehouse": {
        "positive": ["non-marking", "cushion", "visibility", "led", "blue light", "beacon", "positioner", "sideshift"],
        "avoid": ["pneumatic", "solid pneumatic", "dual"],
        "attachments": ["positioner", "sideshift", "fork extensions"],
    },
    # Cold storage / freezer
    "cold storage": {
        "positive": ["cold", "freezer", "low-temp", "heater", "defroster", "enclosed cab", "non-marking cold"],
        "avoid": ["pneumatic", "dual"],
        "attachments": ["positioner", "sideshift"],  # general-purpose ones still apply
    },
    # Construction / debris / rough ground
    "construction": {
        "positive": ["pneumatic", "solid pneumatic", "dual", "traction", "protection", "guard", "radiator", "screen", "belly pan", "led", "beacon"],
        "avoid": ["non-marking"],
        "attachments": ["fork extensions", "positioner", "sideshift", "carpet pole"],
    },
    # Food & Beverage (indoor, hygiene, visibility)
    "food": {
        "positive": ["non-marking", "enclosed cab", "stainless", "anti-corrosion", "visibility", "led", "beacon", "positioner"],
        "avoid": ["pneumatic", "dual"],
        "attachments": ["positioner", "sideshift"],
    },
    "beverage": {
        "positive": ["non-marking", "enclosed cab", "stainless", "anti-corrosion", "visibility", "led", "beacon", "positioner"],
        "avoid": ["pneumatic", "dual"],
        "attachments": ["positioner", "sideshift"],
    },
    # Paper Mill / Printing
    "paper": {
        "positive": ["roll", "clamp", "non-marking", "visibility", "led"],
        "avoid": [],
        "attachments": ["roll clamp", "sideshift", "positioner"],
    },
    # Foundry / Steel / Heavy industry
    "foundry": {
        "positive": ["solid pneumatic", "dual", "protection", "belly pan", "radiator", "heat", "led", "beacon"],
        "avoid": ["non-marking"],
        "attachments": ["fork extensions", "positioner", "sideshift"],
    },
    "steel": {
        "positive": ["solid pneumatic", "dual", "protection", "belly pan", "radiator", "heat", "led", "beacon"],
        "avoid": ["non-marking"],
        "attachments": ["fork extensions", "positioner", "sideshift", "carpet pole"],
    },
    # Port / Yard / Outdoor logistics
    "port": {
        "positive": ["pneumatic", "solid pneumatic", "dual", "protection", "visibility", "led", "beacon"],
        "avoid": ["non-marking"],
        "attachments": ["fork extensions", "positioner", "sideshift"],
    },
    # Recycling / Waste
    "recycling": {
        "positive": ["solid pneumatic", "puncture", "debris", "protection", "belly pan", "radiator", "led", "beacon"],
        "avoid": ["non-marking"],
        "attachments": ["fork extensions", "positioner", "sideshift"],
    },
}

def _match_scenario_key(query: str) -> Optional[str]:
    q = (query or "").lower()
    for key in _SCENARIO_PROFILES.keys():
        if key in q:
            return key
    # map synonyms
    if "warehouse" in q:
        return "warehouse"
    if "freezer" in q:
        return "cold storage"
    if "yard" in q and "lumber" in q:
        return "lumber"
    return None

def _scenario_hints(query: str) -> Tuple[List[str], List[str], Optional[str]]:
    key = _match_scenario_key(query)
    if not key:
        return [], [], None
    prof = _SCENARIO_PROFILES.get(key, {})
    return prof.get("positive", []), prof.get("avoid", []), key

# ------------------------------- Selector prompt -----------------------------
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
    pos, neg, _key = _scenario_hints(query)
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
        set_keys = {k.strip() for k in d.keys()}
        safe = [n for n in names if n.strip() in set_keys]
        return safe
    except Exception:
        return []

# ----------------------------- Tire curation ---------------------------------
_TIRE_WORDS = ("tire", "tyre")

def _is_tire(name: str, blurb: str = "") -> bool:
    s = f"{name} {blurb}".lower()
    return any(w in s for w in _TIRE_WORDS) or any(k in s for k in [
        "pneumatic", "cushion", "solid pneumatic", "non-marking", "dual"
    ])

def _score_tire(name: str, blurb: str, query: str) -> float:
    """Higher score = more relevant for the scenario."""
    q = (query or "").lower()
    s = f"{name} {blurb}".lower()
    score = 0.0

    # Outdoor / rough / debris / lumber → favor pneumatic / solid pneumatic / dual
    if any(k in q for k in ["lumber", "outdoor", "rough", "debris", "yard", "construction", "steel", "port", "recycling"]):
        if "solid pneumatic" in s: score += 5
        if "pneumatic" in s:       score += 4
        if "dual" in s:            score += 3
        if "non-marking" in s:     score -= 4
        if "cushion" in s:         score -= 3

    # Indoor / finished floors / food-bev → favor non-marking / cushion
    if any(k in q for k in ["indoor", "finished floor", "epoxy", "polished", "food", "beverage", "warehouse"]):
        if "non-marking" in s:     score += 5
        if "cushion" in s:         score += 4
        if "pneumatic" in s:       score -= 3
        if "solid pneumatic" in s: score -= 3
        if "dual" in s:            score -= 1

    # Cold storage: non-marking cold-rated gets a bump (if present in your sheet text)
    if any(k in q for k in ["cold", "freezer"]):
        if "cold" in s or "low-temp" in s: score += 2
        if "pneumatic" in s:               score -= 1

    return score

def _curate_tires(names: List[str], pool: Dict[str, str], query: str, asked_all: bool) -> List[str]:
    """
    Keep non-tire items as-is. For tires:
      - If asked_all=True, keep all.
      - Else, keep only the top 1–2 most relevant by _score_tire.
    """
    if asked_all:
        return names

    tire_names = []
    non_tire = []
    for n in names:
        if _is_tire(n, pool.get(n, "")):
            tire_names.append(n)
        else:
            non_tire.append(n)

    if not tire_names:
        return names

    scored = sorted(
        [(n, _score_tire(n, pool.get(n, ""), query)) for n in tire_names],
        key=lambda x: x[1],
        reverse=True
    )
    keep = [n for n, _s in scored[:2]]  # top 1–2 tires
    # Preserve original relative order for readability: non-tires first, then curated tires
    final = non_tire + [n for n in names if n in keep]
    # De-duplicate while preserving order
    seen = set()
    dedup = []
    for n in final:
        if n not in seen:
            seen.add(n)
            dedup.append(n)
    return
