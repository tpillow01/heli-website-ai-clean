# options_attachments_router.py
# Environment-aware, Excel-grounded Options & Attachments router
# Loop-proof clarifier logic:
#  - Robust, typo-tolerant classifier
#  - If user describes an environment (indoor/outdoor/etc.) but not a type, default to BOTH LISTS
#  - Short replies to clarifiers (“both”, “options”, “attachments”, “specific item”) are handled directly
# Selection pipeline:
#  1) Semantic prefilter (embeddings) over Excel names/blurbs
#  2) Auto environment cue extraction (indoor/outdoor/pedestrians/long loads/dust/cold/precision/wet)
#  3) Light scoring nudges from cues (no giant hand lists)
#  4) LLM final pick from candidates ONLY (never invents)
#  5) Tire curation (top 1–2 unless "all")
# Formatting: no markdown emphasis; section headers are HTML-styled

import os
import re
import json
from typing import Dict, Tuple, List
from math import sqrt
from openai import OpenAI

# ----------------------------- OpenAI setup ----------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Models (override via env if you want)
MODEL_CLASSIFIER = os.getenv("OA_CLASSIFIER_MODEL", "gpt-4o-mini")
MODEL_RESPONDER  = os.getenv("OA_RESPONDER_MODEL",  "gpt-4o-mini")
MODEL_SELECTOR   = os.getenv("OA_SELECTOR_MODEL",   "gpt-4o-mini")
EMBED_MODEL      = os.getenv("OA_EMBED_MODEL",      "text-embedding-3-small")

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

# ---------------------- Automatic environment cue extraction ------------------
_CUES = {
    "indoor":        ["indoor", "warehouse", "finished floor", "epoxy", "polished"],
    "outdoor":       ["outdoor", "yard", "rough", "gravel", "pothole", "construction", "lumber", "timber", "debris"],
    "pedestrians":   ["busy", "people", "pedestrian", "foot traffic", "crowded"],
    "long_loads":    ["long", "bundle", "lumber", "pipe", "tubing", "oversized"],
    "dust":          ["dust", "sawdust", "powder", "grain", "cement"],
    "cold":          ["cold storage", "freezer", "low-temp", "low temp", "freezing"],
    "precision":     ["precision", "fine", "tight aisle", "narrow aisle", "delicate"],
    "wet":           ["wet", "washdown", "rain", "condensation"],
}
def _extract_cues(q: str) -> Dict[str, bool]:
    t = (q or "").lower()
    return {k: any(w in t for w in words) for k, words in _CUES.items()}

# ------------------------------ Embeddings & sim ------------------------------
def _embed(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def _cos(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x*y for x, y in zip(a, b))
    na = sqrt(sum(x*x for x in a))
    nb = sqrt(sum(y*y for y in b))
    return dot / (na * nb + 1e-8)

def _rank_semantic(pool: Dict[str, str], query: str, extra_context: str = "", top_k: int = 30) -> List[str]:
    if not pool:
        return []
    items = list(pool.items())
    texts = [f"{n}. {pool.get(n,'')}" for n,_ in items]
    qtext = query.strip()
    if extra_context:
        qtext += f" | context: {extra_context}"
    vecs = _embed([qtext] + texts)
    qv = vecs[0]
    iv = vecs[1:]
    sims = [(items[i][0], _cos(qv, iv[i])) for i in range(len(items))]
    sims.sort(key=lambda x: (-x[1], _normalize(x[0])))
    return [n for n,_ in sims[:top_k]]

# -------------------------- Scoring nudges from cues --------------------------
def _cue_score_nudge(name: str, blurb: str, cues: Dict[str, bool], kind: str) -> float:
    s = f"{name} {blurb}".lower()
    score = 0.0

    # Indoor vs outdoor surfaces
    if cues["indoor"]:
        if "non-marking" in s or "cushion" in s: score += 2.0
        if any(k in s for k in ["pneumatic", "solid pneumatic", "dual", "belly pan", "radiator"]): score -= 0.5
    if cues["outdoor"]:
        if any(k in s for k in ["pneumatic", "solid pneumatic", "dual", "belly pan", "guard", "radiator"]): score += 1.5
        if "non-marking" in s or "cushion" in s: score -= 0.5

    # Pedestrians — visibility & operator assistance
    if cues["pedestrians"]:
        if any(k in s for k in [
            "blue light", "blue spot", "beacon", "horn", "backup handle", "mirror",
            "camera", "rear camera", "radar", "backup radar", "proximity", "collision",
            "ops", "operator presence", "speed control", "speed limiter", "finger control", "fine control"
        ]):
            score += 3.0

    # Long loads — positioning & support
    if cues["long_loads"]:
        if any(k in s for k in ["fork extension", "extension", "positioner", "sideshift", "load backrest", "long fork", "fork length", "carpet pole", "ram", "pole"]):
            score += 2.0

    # Dust — filtration/cooling
    if cues["dust"]:
        if any(k in s for k in ["dual air filter", "air filter", "radiator screen", "cooling", "fan"]):
            score += 1.0

    # Cold — freezer/cab/low-temp
    if cues["cold"]:
        if any(k in s for k in ["cold", "freezer", "low-temp", "heater", "defroster", "enclosed cab", "non-marking"]):
            score += 1.5

    # Precision — controls & visibility
    if cues["precision"]:
        if any(k in s for k in ["finger control", "fine control", "camera", "rear camera", "sideshift", "positioner"]):
            score += 1.5

    # Wet — glass/wiper visibility
    if cues["wet"]:
        if any(k in s for k in ["windshield", "wiper", "rain-proof", "defroster"]):
            score += 1.0

    # Light bias by kind
    if kind == "attachments":
        if any(k in s for k in ["sideshift", "positioner", "clamp", "extension", "load backrest", "carpet pole", "ram", "pole"]):
            score += 0.5

    return score

def _apply_cue_nudges(names: List[str], pool: Dict[str, str], cues: Dict[str, bool], kind: str, min_keep: int = 8) -> List[str]:
    scored = []
    for n in names:
        scored.append((n, _cue_score_nudge(n, pool.get(n, ""), cues, kind)))
    scored.sort(key=lambda x: (-x[1], _normalize(x[0])))
    if len(scored) <= min_keep:
        return [n for n,_ in scored]
    boosted = [n for n,s in scored if s > 0]
    if len(boosted) >= min_keep:
        return boosted[:min_keep]
    need = min_keep - len(boosted)
    tail = [n for n,_ in scored if n not in boosted][:need]
    return boosted + tail

# ------------------------------- Tire curation --------------------------------
_TIRE_WORDS = ("tire", "tyre", "pneumatic", "cushion")
def _is_tire(name: str, blurb: str = "") -> bool:
    s = f"{name} {blurb}".lower()
    return any(w in s for w in _TIRE_WORDS) or any(k in s for k in [
        "solid pneumatic", "non-marking", "dual"
    ])

def _score_tire(name: str, blurb: str, cues: Dict[str, bool]) -> float:
    s = f"{name} {blurb}".lower()
    score = 0.0
    if cues["outdoor"]:
        if "solid pneumatic" in s: score += 5
        if "pneumatic" in s:      score += 4
        if "dual" in s:           score += 3
        if "non-marking" in s:    score -= 3
        if "cushion" in s:        score -= 2
    if cues["indoor"]:
        if "non-marking" in s:    score += 5
        if "cushion" in s:        score += 4
        if "pneumatic" in s:      score -= 2
        if "solid pneumatic" in s:score -= 1
        if "dual" in s:           score -= 1
    if cues["cold"]:
        if "non-marking" in s:    score += 2
    return score

def _curate_tires(names: List[str], pool: Dict[str, str], cues: Dict[str, bool], list_all: bool) -> List[str]:
    if list_all:
        return names
    tires = [(n, _score_tire(n, pool.get(n, ""), cues)) for n in names if _is_tire(n, pool.get(n, ""))]
    non_tires = [n for n in names if not _is_tire(n, pool.get(n, ""))]
    tires_sorted = [n for n, _ in sorted(tires, key=lambda x: (-x[1], _normalize(x[0])))]
    top_tires = tires_sorted[:2]
    return [n for n in names if n in non_tires or n in top_tires]

# ------------------------------- LLM helpers ---------------------------------
def _llm(messages, model: str, temperature: float = 0.2) -> str:
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages,
    )
    return resp.choices[0].message.content

def _classify_llm(user_text: str) -> Dict[str, str]:
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

# ----------------------- LOOP-PROOF robust classifier ------------------------
def _classify(user_text: str) -> Dict[str, str]:
    """
    Typo-tolerant + environment-aware classifier that avoids loops.
    - 'both', 'both list(s)', 'bith', 'both please', 'options and attachments' => both_lists
    - 'options only', 'just options', 'show options' => list_options
    - 'attachments only', 'just attachments', 'show attachments' => list_attachments
    - 'all', 'catalog', 'everything', 'full list', 'complete list' => list_all (for mentioned type, else both)
    - If a known item name appears, => detail_item
    - If the user describes an environment (indoor/outdoor/etc.) but not a type => both_lists
    """
    t_raw = (user_text or "").strip()
    t = f" {t_raw.lower()} "
    t_norm = _normalize(t_raw)

    # 1) direct short answers to a clarifier
    if re.search(r"\bboth( lists?)?\b", t) or " bith " in t or " bothlist " in t or " bothlists " in t or " options and attachments " in t or " attachments and options " in t:
        return {"intent":"both_lists","item":""}
    if re.search(r"\boptions( only)?\b", t) or " just options " in t or " show options " in t:
        return {"intent":"list_options","item":""}
    if re.search(r"\battachments( only)?\b", t) or " just attachments " in t or " show attachments " in t:
        return {"intent":"list_attachments","item":""}
    if re.search(r"\bspecific item\b", t) or re.search(r"\bdetail\b", t):
        # Try to extract an item if present; else fallback handled by caller
        k = fuzzy_lookup(t_raw)[1]
        if k: return {"intent":"detail_item","item":k}

    # 2) explicit ALL
    if any(sig in t for sig in [" catalog", " everything", " all ", " full list", " complete list"]):
        has_opt = " option" in t
        has_att = " attachment" in t
        if has_opt and has_att:
            return {"intent":"list_all","item":""}
        if has_opt:
            return {"intent":"list_all","item":"options"}
        if has_att:
            return {"intent":"list_all","item":"attachments"}
        return {"intent":"list_all","item":""}

    # 3) specific item heuristic
    kind, key, _ = fuzzy_lookup(t_raw)
    if key:
        return {"intent":"detail_item","item":key}

    # 4) explicit mentions
    has_opt = " option" in t
    has_att = " attachment" in t
    if has_opt and has_att:
        return {"intent":"both_lists","item":""}
    if has_opt:
        return {"intent":"list_options","item":""}
    if has_att:
        return {"intent":"list_attachments","item":""}

    # 5) environment cues => default BOTH LISTS (prevents clarifier loops)
    cues = _extract_cues(t_raw)
    if any(cues.values()):
        return {"intent":"both_lists","item":""}

    # 6) fallback to LLM classifier once; if still unknown, default BOTH
    try:
        llm_guess = _classify_llm(t_raw)
        if llm_guess.get("intent") and llm_guess["intent"] != "unknown":
            return llm_guess
    except Exception:
        pass
    return {"intent":"both_lists","item":""}

# Final grounded selector (never invents names)
_SELECTOR_INSTRUCTION = (
    "Choose the most relevant forklift {kind} for the user's scenario, using general forklift knowledge. "
    "You MUST select only from the CANDIDATES provided. Do NOT invent new items.\n"
    "Return JSON only: {\"items\": [\"Exact Name 1\", \"Exact Name 2\", ...]}\n\n"
    "SCENARIO:\n{query}\n\n"
    "CANDIDATES ({count}):\n{candidates}\n"
)

def _format_candidates_list(names: List[str], pool: Dict[str, str]) -> str:
    return "\n".join(f"- {n}: {pool.get(n,'')}" for n in names)

def _llm_pick_from(names: List[str], pool: Dict[str, str], kind: str, query: str) -> List[str]:
    if not names:
        return []
    try:
        out = _llm(
            [
                {"role":"system","content":"Return JSON only. No explanations."},
                {"role":"user","content": _SELECTOR_INSTRUCTION.format(
                    kind=kind,
                    query=query,
                    count=len(names),
                    candidates=_format_candidates_list(names, pool)
                )},
            ],
            model=MODEL_SELECTOR,
            temperature=0.1,
        )
        data = json.loads(out)
        sel = [n for n in data.get("items", []) if isinstance(n, str)]
        allowed = {n.strip() for n in names}
        return [n for n in sel if n.strip() in allowed]
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
    lines = [f"- {n} — {d.get(n, '') or '—'}" for n in names]  # no markdown styling
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

# --------------------------- Public entry point ------------------------------
def respond_options_attachments(user_text: str) -> str:
    if not (user_text or "").strip():
        return "Ask about options, attachments, or a specific item (e.g., Fork Positioner)."

    # Intent (loop-proof)
    c = _classify(user_text)
    intent = c.get("intent", "both_lists")  # defaulted to BOTH_LISTS already for safety
    item   = (c.get("item") or "").strip()
    t = _normalize(user_text)
    asked_all = any(w in t for w in [" all ", "catalog", "everything", "full list", "complete"])

    opt_pool = _merged_options()
    att_pool = ATTACHMENTS

    # BOTH lists
    if intent == "both_lists":
        cues = _extract_cues(user_text)
        if asked_all:
            opt_names = sorted(opt_pool.keys(), key=lambda k: _normalize(k))
            att_names = sorted(att_pool.keys(), key=lambda k: _normalize(k))
        else:
            opt_pref = _rank_semantic(opt_pool, user_text, extra_context="Select options (including tires) for this scenario.", top_k=30)
            att_pref = _rank_semantic(att_pool, user_text, extra_context="Select attachments for this scenario.", top_k=30)

            opt_pref = _apply_cue_nudges(opt_pref, opt_pool, cues, kind="options",     min_keep=10)
            att_pref = _apply_cue_nudges(att_pref, att_pool, cues, kind="attachments", min_keep=8)

            opt_names = _llm_pick_from(opt_pref, opt_pool, "options (incl. tires)", user_text)
            att_names = _llm_pick_from(att_pref, att_pool, "attachments", user_text)

            opt_names = _curate_tires(opt_names, opt_pool, cues, list_all=False)

            if not opt_names and not att_names:
                # Single clarifier (no loop): choose one of three
                return 'Do you want options, attachments, or both lists? (e.g., "options only", "attachments only", "both lists")'

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
        if c.get("item") == "options" or "option" in t:
            names = sorted(opt_pool.keys(), key=lambda k: _normalize(k))
            return _build_list("Options", names, opt_pool)
        if c.get("item") == "attachments" or "attachment" in t:
            names = sorted(att_pool.keys(), key=lambda k: _normalize(k))
            return _build_list("Attachments", names, att_pool)
        # default both
        opt_names = sorted(opt_pool.keys(), key=lambda k: _normalize(k))
        att_names = sorted(att_pool.keys(), key=lambda k: _normalize(k))
        return _build_list("Options", opt_names, opt_pool) + "\n\n" + _build_list("Attachments", att_names, att_pool)

    # Options-only
    if intent == "list_options":
        cues = _extract_cues(user_text)
        if asked_all:
            names = sorted(opt_pool.keys(), key=lambda k: _normalize(k))
        else:
            pref = _rank_semantic(opt_pool, user_text, extra_context="Select options (including tires) for this scenario.", top_k=30)
            pref = _apply_cue_nudges(pref, opt_pool, cues, kind="options", min_keep=10)
            names = _llm_pick_from(pref, opt_pool, "options (incl. tires)", user_text)
            names = _curate_tires(names, opt_pool, cues, list_all=False)
            if not names:
                return 'Want me to include attachments too, or show all options? (say "both lists" or "all options")'
        return _build_list("Options", names, opt_pool)

    # Attachments-only
    if intent == "list_attachments":
        cues = _extract_cues(user_text)
        if asked_all:
            names = sorted(att_pool.keys(), key=lambda k: _normalize(k))
        else:
            pref = _rank_semantic(att_pool, user_text, extra_context="Select attachments for this scenario.", top_k=30)
            pref = _apply_cue_nudges(pref, att_pool, cues, kind="attachments", min_keep=8)
            names = _llm_pick_from(pref, att_pool, "attachments", user_text)
            if not names:
                return 'Want me to include options too, or show all attachments? (say "both lists" or "all attachments")'
        return _build_list("Attachments", names, att_pool)

    # Single item deep dive
    if intent == "detail_item":
        # If classifier didn't extract an item but user wrote "specific item", try fuzzy
        if not item:
            kind, key, blurb = fuzzy_lookup(user_text)
        else:
            kind, key, blurb = fuzzy_lookup(item)
        if not key:
            return "Which item do you want details on? (e.g., Fork Positioner, Sideshifter)"
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

    # Absolute fallback (never loop): assume BOTH LISTS for described scenarios; else one clarifier
    cues = _extract_cues(user_text)
    if any(cues.values()):
        # act as both_lists
        return respond_options_attachments(user_text + " (both lists)")
    return 'Do you want options, attachments, or both lists? (e.g., "options only", "attachments only", "both lists")'
