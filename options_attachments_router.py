# options_attachments_router.py
# Router for "Options & Attachments":
# - Lists ONLY options when asked for options
# - Lists ONLY attachments when asked for attachments
# - Deep-dive when a single item is named
# - "attachments and options" prints both (short lists unless "all" is requested)
# - Loads full catalogs from ai_logic loaders (Excel-backed), with safe fallbacks
# - Filters list results to match user keywords (unless "all" is requested)
# - Styles headers with <span class="section-label">…</span>, keeps other **bold** intact

import os
import re
import json
from typing import Dict, Tuple, List
from openai import OpenAI

# ------------------------------------------------------------
# OpenAI setup (override via env if desired)
# ------------------------------------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_CLASSIFIER = os.getenv("OA_CLASSIFIER_MODEL", "gpt-4o-mini")
MODEL_RESPONDER  = os.getenv("OA_RESPONDER_MODEL",  "gpt-4o-mini")

# ------------------------------------------------------------
# Try to import your real loaders from ai_logic
# (We know load_options() exists from your /api/options endpoint.)
# If load_attachments() doesn't exist, we fall back gracefully.
# ------------------------------------------------------------
try:
    from ai_logic import load_options as _load_options_catalog  # returns list[dict]
except Exception:
    _load_options_catalog = None

try:
    from ai_logic import load_attachments as _load_attachments_catalog  # optional
except Exception:
    _load_attachments_catalog = None

# ------------------------------------------------------------
# Fallbacks (used only if loaders are missing or fail)
# ------------------------------------------------------------
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

def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

def _to_dict_by_name(rows: List[dict], name_key="name", blurb_key="benefit") -> Dict[str, str]:
    """
    Turn a list of rows from your loader into {name: blurb}.
    Keeps only rows with a non-empty name. Trims whitespace. De-dupes by last write wins.
    """
    out = {}
    for r in rows or []:
        name = (r.get(name_key) or "").strip()
        if not name:
            continue
        blurb = (r.get(blurb_key) or r.get("desc") or r.get("description") or "").strip()
        out[name] = blurb
    return out

def _hydrate_catalogs() -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Attempts to load from ai_logic loaders first; falls back to hard-coded sets.
    - OPTIONS comes from load_options()  -> expects fields: name, benefit
    - ATTACHMENTS comes from load_attachments() if present; otherwise fallback
    """
    # OPTIONS
    try:
        if _load_options_catalog:
            opt_rows = _load_options_catalog()  # expected list[dict] with 'name' and 'benefit'
            options = _to_dict_by_name(opt_rows, name_key="name", blurb_key="benefit")
            if not options:
                options = FALLBACK_OPTIONS
        else:
            options = FALLBACK_OPTIONS
    except Exception:
        options = FALLBACK_OPTIONS

    # ATTACHMENTS (only if you have a loader; otherwise fallback)
    try:
        if _load_attachments_catalog:
            att_rows = _load_attachments_catalog()  # expected list[dict] with 'name' and 'benefit'/desc
            attachments = _to_dict_by_name(att_rows, name_key="name", blurb_key="benefit")
            if not attachments:
                attachments = FALLBACK_ATTACHMENTS
        else:
            attachments = FALLBACK_ATTACHMENTS
    except Exception:
        attachments = FALLBACK_ATTACHMENTS

    return options, attachments

OPTIONS, ATTACHMENTS = _hydrate_catalogs()

# ------------------------------------------------------------
# Fuzzy lookup for single-item deep dives (works with full catalogs)
# ------------------------------------------------------------
def fuzzy_lookup(item: str) -> Tuple[str, str, str]:
    """
    Return (kind, canonical_name, blurb), where kind in {'option','attachment',''}.
    """
    n = _normalize(item)

    # Exact or contains match across loaded catalogs
    for k, v in OPTIONS.items():
        if n == _normalize(k) or n in _normalize(k):
            return ("option", k, v)
    for k, v in ATTACHMENTS.items():
        if n == _normalize(k) or n in _normalize(k):
            return ("attachment", k, v)

    # Keyword hints (kept minimal)
    if "clamp" in n:
        k = next((kk for kk in ATTACHMENTS.keys() if "clamp" in _normalize(kk)), "Paper Roll Clamp")
        return ("attachment", k, ATTACHMENTS.get(k, "Clamp attachment."))
    if "position" in n or "positioner" in n:
        # prefer exact catalog name if it exists
        k = next((kk for kk in ATTACHMENTS.keys() if "position" in _normalize(kk)), "Fork Positioner")
        return ("attachment", k, ATTACHMENTS.get(k, ""))
    if "side shift" in n or "sideshift" in n or "side-shift" in n:
        k = next((kk for kk in ATTACHMENTS.keys() if "side" in _normalize(kk) and "shift" in _normalize(kk)), "Sideshifter")
        return ("attachment", k, ATTACHMENTS.get(k, ""))
    if "valve" in n or "aux" in n:
        k = next((kk for kk in OPTIONS.keys() if "valve" in _normalize(kk)), "3 Valve with Handle")
        return ("option", k, OPTIONS.get(k, ""))

    return ("", "", "")

# ------------------------------------------------------------
# Prompts
# ------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are the Options & Attachments expert for Heli forklifts.\n"
    "STRICT RULES:\n"
    "- Answer ONLY what the user asked for.\n"
    "- If they say “options”, return options only. If they say “attachments”, return attachments only.\n"
    "- If they name a specific item, give a concise deep-dive: Purpose, Benefits, When to Use, "
    "Prerequisites/Valving, Compatibility/Capacity impacts, Trade-offs.\n"
    "- NEVER output the full catalog unless the user explicitly asks for “all”, “catalog”, or “everything”.\n"
    "- If the user is ambiguous, ask ONE clarifying question, then stop.\n"
    "- Keep responses short and skimmable. Bullets > paragraphs.\n"
    "- Call out capacity/visibility/turning/maintenance impacts when relevant."
)

# Use doubled braces to keep .format() from eating JSON keys.
CLASSIFIER_INSTRUCTION = (
    "Classify the user request for forklift options/attachments into exactly one of:\n"
    "- list_options\n"
    "- list_attachments\n"
    "- detail_item\n"
    "- list_all\n"
    "- both_lists\n"
    "- unknown\n\n"
    "Rules:\n"
    "- If the user clearly wants BOTH (e.g., 'attachments and options', 'everything I can add'), use both_lists.\n"
    "- 'all', 'everything', or 'catalog' implies list_all for the relevant type(s) mentioned.\n\n"
    "Return JSON only: {{\"intent\":\"...\", \"item\": \"<named item or ''>\"}}\n\n"
    "User: {user_text}\n"
)

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
        # Simple keyword fallback if classifier hiccups
        t = _normalize(user_text)
        if "attachment" in t and "option" in t:
            return {"intent":"both_lists","item":""}
        if "attachment" in t:
            return {"intent":"list_attachments","item":""}
        if "option" in t:
            return {"intent":"list_options","item":""}
        if any(w in t for w in ["catalog","everything","all","full list","complete"]):
            return {"intent":"list_all","item":""}
        return {"intent":"unknown","item":""}

# ------------------------------------------------------------
# Output formatting (uses your .section-label CSS class)
# ------------------------------------------------------------
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

# Strip ** or __ ONLY when used around the above headers (so your other **bold** stays intact)
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

# ------------------------------------------------------------
# Keyword filtering for list results
# ------------------------------------------------------------
_STOPWORDS = {
    "the","a","an","for","to","of","and","or","me","show","list","give","all",
    "everything","full","complete","what","do","you","have","options","option",
    "attachment","attachments","can","i","add","my","on","in"
}

def _keywords(text: str) -> List[str]:
    words = re.findall(r"[a-z0-9\-]+", (text or "").lower())
    return [w for w in words if w not in _STOPWORDS]

def _filter_catalog(d: Dict[str, str], query: str) -> List[Tuple[str, str]]:
    """
    Returns a filtered list of (name, blurb) pairs from the given catalog
    that contain at least one keyword from the query in either name or blurb.
    If no matches, returns [].
    """
    kws = _keywords(query)
    if not kws:
        return []
    hits: List[Tuple[str, str]] = []
    for name, blurb in d.items():
        hay = f"{name} {blurb}".lower()
        if any(k in hay for k in kws):
            hits.append((name, blurb))
    return hits

def _build_list_from_dict(kind_name: str, d: Dict[str, str], list_all: bool, query: str) -> str:
    header_html = f'<span class="section-label">{kind_name}:</span>'

    # If user asked for ALL, skip filtering and show everything (sorted)
    if list_all:
        items = sorted(d.items(), key=lambda kv: _normalize(kv[0]))
        lines = [f"- {name} — {blurb or '—'}" for name, blurb in items]
        return header_html + "\n" + "\n".join(lines)

    # Otherwise, try to filter by their words (e.g., "cold storage", "paper rolls")
    filtered = _filter_catalog(d, query)

    if filtered:
        items = sorted(filtered, key=lambda kv: _normalize(kv[0]))[:12]
    else:
        # No keyword match — fall back to a short, general list (still from catalog)
        items = sorted(d.items(), key=lambda kv: _normalize(kv[0]))[:12]

    lines = [f"- {name} — {blurb or '—'}" for name, blurb in items]
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

# ------------------------------------------------------------
# Public entry
# ------------------------------------------------------------
def respond_options_attachments(user_text: str) -> str:
    if not (user_text or "").strip():
        return "Ask about **options**, **attachments**, or a **specific item** (e.g., Fork Positioner)."

    c = _classify(user_text)
    intent = c.get("intent", "unknown")
    item   = (c.get("item") or "").strip()

    t = _normalize(user_text)
    asked_all = any(w in t for w in [" all ", "catalog", "everything", "full list", "complete"])

    # BOTH lists (short unless explicitly "all")
    if intent == "both_lists":
        parts = []
        parts.append(_build_list_from_dict("Options", OPTIONS, list_all=asked_all, query=user_text))
        parts.append(_build_list_from_dict("Attachments", ATTACHMENTS, list_all=asked_all, query=user_text))
        return "\n\n".join(parts)

    # Explicit "all" for one/both types
    if intent == "list_all":
        if "option" in t and "attachment" in t:
            return (
                _build_list_from_dict("Options", OPTIONS, list_all=True, query=user_text) + "\n\n" +
                _build_list_from_dict("Attachments", ATTACHMENTS, list_all=True, query=user_text)
            )
        if "option" in t:
            return _build_list_from_dict("Options", OPTIONS, list_all=True, query=user_text)
        if "attachment" in t or "catalog" in t or "everything" in t:
            return _build_list_from_dict("Attachments", ATTACHMENTS, list_all=True, query=user_text)
        # ambiguous "all"
        return "Do you want **all options**, **all attachments**, or **both**?"

    # Options-only list
    if intent == "list_options":
        return _build_list_from_dict("Options", OPTIONS, list_all=asked_all, query=user_text)

    # Attachments-only list
    if intent == "list_attachments":
        return _build_list_from_dict("Attachments", ATTACHMENTS, list_all=asked_all, query=user_text)

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
            content = _strip_md_bold_headers(content)  # only strips ** around the headers
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

    # Fallback clarifier
    if "attachment" in t and "option" in t:
        return "Do you want **both lists**, or details on a **specific item**?"
    if "attachment" in t:
        return "Do you want a **list of attachments** or details on a **specific attachment**?"
    if "option" in t:
        return "Do you want a **list of options** or details on a **specific option**?"
    return "Do you want **options**, **attachments**, or details on a specific item (e.g., Fork Positioner)?"
