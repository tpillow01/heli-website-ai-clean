# options_attachments_router.py
# Excel-backed, AI-selected router for "Options & Attachments":
# - Lists ONLY options when asked for options
# - Lists ONLY attachments when asked for attachments
# - "attachments and options" can include tires (if loaded) when relevant
# - Deep-dive when a single item is named
# - Uses AI to SELECT relevant items from your Excel catalogs by general forklift knowledge
# - NEVER invents items not present in your catalogs
# - NO list-size cap: returns all items the AI deems relevant
# - Styles headers via <span class="section-label">…</span>, keeps other **bold** intact

import os
import re
import json
from typing import Dict, Tuple, List
from openai import OpenAI

# ----------------------------- OpenAI setup ----------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_CLASSIFIER = os.getenv("OA_CLASSIFIER_MODEL", "gpt-4o-mini")
MODEL_RESPONDER  = os.getenv("OA_RESPONDER_MODEL",  "gpt-4o-mini")
MODEL_SELECTOR   = os.getenv("OA_SELECTOR_MODEL",   "gpt-4o-mini")  # used to pick relevant items from your catalog

# ----------------------- Excel-backed loaders (ai_logic) ---------------------
# Required: load_options()
try:
    from ai_logic import load_options as _load_options_catalog  # returns list[dict] with name, benefit
except Exception:
    _load_options_catalog = None

# Optional: load_attachments()
try:
    from ai_logic import load_attachments as _load_attachments_catalog
except Exception:
    _load_attachments_catalog = None

# Optional: tire loader (either works; only one needs to exist)
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
    "Non-Marking Tires":  "Prevents scuffing on indoor finished floors.",
    "Dual Pneumatic Tires": "Wider footprint and flotation on soft ground.",
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

# Consolidated options pool (options + tires)
def _merged_options() -> Dict[str, str]:
    merged = dict(OPTIONS)
    # Avoid duplicate keys; TIRES may contain names already present
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

# Selector prompt: pick relevant items ONLY from provided candidates
SELECTOR_INSTRUCTION = (
    "You are selecting relevant forklift {kind} for the user's scenario.\n"
    "You can ONLY choose items from the CANDIDATES list provided. Do NOT invent anything.\n"
    "Pick items that would practically help, using general forklift knowledge (industry context, environment, load type).\n"
    "Return JSON only with exact names from the candidates: {{\"items\": [\"Name 1\", \"Name 2\", ...]}}\n"
    "If nothing is relevant, return {{\"items\": []}}.\n"
    "\nUSER SCENARIO:\n{query}\n"
    "\nCANDIDATES ({count}):\n{candidates}\n"
)

def _format_candidates(d: Dict[str, str]) -> str:
    # Compact, name-first lines so the model can anchor on names; blurbs help context.
    # We deliberately keep it simple to reduce tokens.
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
        return {
            "intent": str(data.get("intent", "unknown") or "unknown"),
            "item":   str(data.get("item", "") or "")
        }
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
    Returns a list of names that MUST be subset of d.keys(). If parsing fails,
    return [] (caller will ask a clarifier).
    """
    if not d:
        return []
    candidates_text = _format_candidates(d)
    try:
        out = _llm(
            [
                {"role":"system","content":"Return JSON only. No explanations."},
                {"role":"user","content": SELECTOR_INSTRUCTION.format(
                    kind=kind, query=query, count=len(d), candidates=candidates_text
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

# ------------------------------ List builders --------------------------------
def _build_list(kind_name: str, names: List[str], d: Dict[str, str]) -> str:
    header_html = f'<span class="section-label">{kind_name}:</span>'
    if not names:
        typename = kind_name.lower()
        return (
            f'{header_html}\n'
            f'- No {typename} matched your scenario. '
            f'Try asking more specifically (e.g., “{typename} for long loads”, “{typename} for cold storage”), '
            f'or say “all {typename}”.'
        )
    lines = [f"- **{n}** — {d.get(n, '') or '—'}" for n in names]
    return header_html + "\n" + "\n".join(lines)

def _detail_prompt(name: str, seed_blurb: str) -> str:
    # Ask the model for plain headers (no bold). We'll style them ourselves.
    return (
        "Give a concise deep-dive on this SINGLE item.\n"
        "Use the exact headers below and bullet points. Do NOT use bold/italics.\n\n"
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
        return "Ask about **options**, **attachments**, or a **specific item** (e.g., Fork Positioner)."

    c = _classify(user_text)
    intent = c.get("intent", "unknown")
    item   = (c.get("item") or "").strip()

    t = _normalize(user_text)
    asked_all = any(w in t for w in [" all ", "catalog", "everything", "full list", "complete"])

    # BOTH lists
    if intent == "both_lists":
        # Use AI selector to choose relevant items from each catalog
        opt_pool = _merged_options()
        att_pool = ATTACHMENTS

        opt_names = sorted(opt_pool.keys(), key=lambda k: _normalize(k)) if asked_all else _select_relevant(opt_pool, "options (incl. tires)", user_text)
        att_names = sorted(att_pool.keys(), key=lambda k: _normalize(k)) if asked_all else _select_relevant(att_pool, "attachments", user_text)

        # If the selector finds nothing for both, ask a single clarifier
        if not asked_all and not opt_names and not att_names:
            return "Do you want **all options**, **all attachments**, or details on a **specific item**?"

        parts = []
        parts.append(_build_list("Options", opt_names, opt_pool))
        parts.append(_build_list("Attachments", att_names, att_pool))
        return "\n\n".join(parts)

    # ALL for type(s)
    if intent == "list_all":
        if "option" in t and "attachment" in t:
            opt_names = sorted(_merged_options().keys(), key=lambda k: _normalize(k))
            att_names = sorted(ATTACHMENTS.keys(),       key=lambda k: _normalize(k))
            return _build_list("Options", opt_names, _merged_options()) + "\n\n" + _build_list("Attachments", att_names, ATTACHMENTS)
        if "option" in t:
            names = sorted(_merged_options().keys(), key=lambda k: _normalize(k))
            return _build_list("Options", names, _merged_options())
        if "attachment" in t or "catalog" in t or "everything" in t:
            names = sorted(ATTACHMENTS.keys(), key=lambda k: _normalize(k))
            return _build_list("Attachments", names, ATTACHMENTS)
        return "Do you want **all options**, **all attachments**, or **both**?"

    # Options-only
    if intent == "list_options":
        pool = _merged_options()
        names = sorted(pool.keys(), key=lambda k: _normalize(k)) if asked_all else _select_relevant(pool, "options (incl. tires)", user_text)
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
            content = _strip_md_bold_headers(content)  # only strips ** around section headers
            return _decorate_headers(content, title=f"{key} — Deep Dive")
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
            return _decorate_headers(fallback, title=f"{key} — Deep Dive")

    # Clarifiers
    if "attachment" in t and "option" in t:
        return "Do you want **both lists**, or details on a **specific item**?"
    if "attachment" in t:
        return "Do you want a **list of attachments** or details on a **specific attachment**?"
    if "option" in t:
        return "Do you want a **list of options** or details on a **specific option**?"
    return "Do you want **options**, **attachments**, or details on a specific item (e.g., Fork Positioner)?"
