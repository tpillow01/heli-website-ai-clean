# options_attachments_router.py
# Focused router for "Options & Attachments" chat behavior:
# - Lists ONLY options when asked for options
# - Lists ONLY attachments when asked for attachments
# - Deep-dive when a single item is named
# - Never dumps the full catalog unless explicitly asked
# - Formats section headers with <span class="section-label">…</span> to match site styling

import os
import re
import json
from typing import Dict, Tuple
from openai import OpenAI

# ---------------------------------------------------------------------
# OpenAI setup (models default to widely-available ones; override via env)
# ---------------------------------------------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL_CLASSIFIER = os.getenv("OA_CLASSIFIER_MODEL", "gpt-4o-mini")
MODEL_RESPONDER  = os.getenv("OA_RESPONDER_MODEL",  "gpt-4o-mini")

# ---------------------------------------------------------------------
# Catalog loading (optional external data file + safe fallbacks)
# ---------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "forklift_options_benefits.json")

FALLBACK_OPTIONS = {
    "3 valve with handle": "Adds a third hydraulic circuit to power basic attachments.",
    "4 valve with handle": "Two auxiliary circuits for multi-function attachments.",
    "5 valve with handle": "Additional hydraulic circuits for complex attachments.",
    "non-marking tires":  "Prevents scuffing on indoor finished floors.",
    "dual tires":         "Wider footprint for stability on soft or uneven ground.",
    "dual solid tires":   "Puncture-proof with extra stability in debris-prone areas.",
}

FALLBACK_ATTACHMENTS = {
    "sideshifter":            "Shift load left/right without moving the truck.",
    "fork positioner":        "Adjust fork spread from the seat for mixed pallet widths.",
    "fork extensions":        "Temporarily lengthen forks for long/oversized loads.",
    "paper roll clamp":       "Securely handle paper rolls without core damage.",
    "push/pull (slip-sheet)": "Use slip-sheets instead of pallets to cut cost/weight.",
    "carpet pole":            "Ram/pole for carpet, coils, and tubing.",
}

def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

def _load_catalog() -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Load an optional JSON data file with richer blurbs:
    {
      "options":     [{"name":"Non-Marking Tires","blurb":"..."}],
      "attachments": [{"name":"Sideshifter","blurb":"..."}]
    }
    Falls back gracefully if missing or invalid.
    """
    if not os.path.exists(DATA_FILE):
        return FALLBACK_OPTIONS, FALLBACK_ATTACHMENTS
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        opts = {i["name"]: i.get("blurb", "") for i in data.get("options", []) if i.get("name")}
        atts = {i["name"]: i.get("blurb", "") for i in data.get("attachments", []) if i.get("name")}
        if not opts: opts = FALLBACK_OPTIONS
        if not atts: atts = FALLBACK_ATTACHMENTS
        return opts, atts
    except Exception:
        return FALLBACK_OPTIONS, FALLBACK_ATTACHMENTS

OPTIONS, ATTACHMENTS = _load_catalog()

# ---------------------------------------------------------------------
# Light fuzzy lookup for single-item deep dives
# ---------------------------------------------------------------------
def fuzzy_lookup(item: str) -> Tuple[str, str, str]:
    """
    Return (kind, canonical_name, blurb), where kind in {'option','attachment',''}.
    """
    n = _normalize(item)

    # exact/contains
    for k, v in OPTIONS.items():
        if n == _normalize(k) or n in _normalize(k):
            return ("option", k, v)
    for k, v in ATTACHMENTS.items():
        if n == _normalize(k) or n in _normalize(k):
            return ("attachment", k, v)

    # keyword hints
    if "clamp" in n:
        k = next((kk for kk in ATTACHMENTS if "clamp" in _normalize(kk)), "paper roll clamp")
        return ("attachment", k, ATTACHMENTS.get(k, "Clamp attachment"))
    if "position" in n or "positioner" in n:
        return ("attachment", "fork positioner", ATTACHMENTS.get("fork positioner", ""))
    if "side shift" in n or "sideshift" in n:
        return ("attachment", "sideshifter", ATTACHMENTS.get("sideshifter", ""))
    if "valve" in n:
        return ("option", "3 valve with handle", OPTIONS.get("3 valve with handle", ""))

    return ("", "", "")

# ---------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------
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

# IMPORTANT: use doubled braces to avoid str.format eating the JSON keys
CLASSIFIER_INSTRUCTION = (
    "Classify the user request for forklift options/attachments into exactly one of:\n"
    "- list_options\n"
    "- list_attachments\n"
    "- detail_item\n"
    "- list_all\n"
    "- unknown\n\n"
    "Return JSON only: {{\"intent\":\"...\", \"item\": \"<named item or ''>\"}}\n\n"
    "User: {user_text}\n"
)

# ---------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------
def _llm(messages, model: str, temperature: float = 0.2) -> str:
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages,
    )
    return resp.choices[0].message.content

# ---------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------
def _classify(user_text: str) -> Dict[str, str]:
    """
    Returns {"intent": "...", "item": "..."}.
    If anything goes wrong, returns {"intent":"unknown","item":""}.
    """
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
        intent = str(data.get("intent", "unknown") or "unknown")
        item   = str(data.get("item", "") or "")
        return {"intent": intent, "item": item}
    except Exception:
        return {"intent": "unknown", "item": ""}

# ---------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------
# Map of plain headers -> HTML-decorated headers that your CSS styles (bold + dark red)
_HEADER_MAP = {
    "Purpose:":                       '<span class="section-label">Purpose:</span>',
    "Benefits:":                      '<span class="section-label">Benefits:</span>',
    "When to use:":                   '<span class="section-label">When to use:</span>',
    "Prerequisites/Valving:":         '<span class="section-label">Prerequisites/Valving:</span>',
    "Compatibility/Capacity impacts:":'<span class="section-label">Compatibility/Capacity impacts:</span>',
    "Trade-offs:":                    '<span class="section-label">Trade-offs:</span>',
}

_HEADER_PATTERN = re.compile(
    r'(?im)^(Purpose:|Benefits:|When to use:|Prerequisites/Valving:|Compatibility/Capacity impacts:|Trade-offs:)\s*$'
)

def _decorate_headers(text: str, title: str = "") -> str:
    """Wrap known headers with <span class="section-label"> and optionally add a title line."""
    if not text:
        return text
    # Ensure consistent line breaks
    s = text.replace('\r\n', '\n').replace('\r', '\n').strip()
    # Inject title at top if provided
    if title:
        s = f'<span class="section-label">{title}</span>\n{s}'
    # Replace headers
    def repl(m):
        hdr = m.group(1)
        return _HEADER_MAP.get(hdr, hdr)
    s = _HEADER_PATTERN.sub(repl, s)
    return s

def _build_list(kind: str, list_all: bool = False) -> str:
    if kind == "options":
        items = list(OPTIONS.items())
        header_html = '<span class="section-label">Options:</span>'
    else:
        items = list(ATTACHMENTS.items())
        header_html = '<span class="section-label">Attachments:</span>'

    if not list_all:
        items = items[:10]  # sensible default slice

    lines = [f"- {name} — {blurb or '—'}" for name, blurb in items]
    return header_html + "\n" + "\n".join(lines)

def _detail_prompt(name: str, seed_blurb: str) -> str:
    # Ask the model to use our exact plain-text headers; we’ll decorate after.
    return (
        "Give a concise deep-dive on this SINGLE item.\n"
        "Use the exact headers below and bullet points. Keep it brief.\n\n"
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

# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------
def respond_options_attachments(user_text: str) -> str:
    """
    Main router used by /api/options_attachments_chat
    """
    if not (user_text or "").strip():
        return "Ask about **options**, **attachments**, or a **specific item** (e.g., Fork Positioner)."

    c = _classify(user_text)
    intent = c.get("intent", "unknown")
    item   = (c.get("item") or "").strip()

    t = _normalize(user_text)
    asked_all = any(w in t for w in [" all ", "catalog", "everything"])

    # Explicit "all"
    if intent == "list_all":
        if "option" in t:
            return _build_list("options", list_all=True)
        if "attachment" in t or "catalog" in t or "everything" in t:
            return _build_list("attachments", list_all=True)
        return "Do you want **all options** or **all attachments**?"

    # Options-only list
    if intent == "list_options":
        return _build_list("options", list_all=asked_all)

    # Attachments-only list
    if intent == "list_attachments":
        return _build_list("attachments", list_all=asked_all)

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
            # Decorate headers and add a title line matching your style
            return _decorate_headers(content, title=f"{key} — Deep Dive")
        except Exception:
            # Friendly fallback if model hiccups
            fallback = (
                f"{_HEADER_MAP['Purpose:']}\n"
                f"- {blurb or 'Attachment/option for specialized handling.'}\n\n"
                f"{_HEADER_MAP['Benefits:']}\n"
                "- Faster handling.\n- Safer adjustments.\n- Better load fit.\n\n"
                f"{_HEADER_MAP['When to use:']}\n"
                "- Mixed load profiles or frequent changes.\n\n"
                f"{_HEADER_MAP['Prerequisites/Valving:']}\n"
                "- May require 3rd/4th hydraulic function.\n\n"
                f"{_HEADER_MAP['Compatibility/Capacity impacts:']}\n"
                "- Minor capacity derate; verify on data plate.\n\n"
                f"{_HEADER_MAP['Trade-offs:']}\n"
                "- Higher upfront cost and modest maintenance."
            )
            return _decorate_headers(fallback, title=f"{key} — Deep Dive")

    # Fallback clarifier
    if "attachment" in t:
        return "Do you want a **list of attachments** or details on a **specific attachment**?"
    if "option" in t:
        return "Do you want a **list of options** or details on a **specific option**?"
    return "Do you want **options**, **attachments**, or details on a specific item (e.g., Fork Positioner)?"
