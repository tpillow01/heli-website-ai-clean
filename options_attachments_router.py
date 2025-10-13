# options_attachments_router.py
import os, re, json
from typing import Dict, Tuple, List
from openai import OpenAI

# ---------- LLM client ----------
client = OpenAI()

# ---------- Catalog loading ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "forklift_options_benefits.json")

# Built-in fallback (short blurbs). You can expand or ignore if you have the JSON.
FALLBACK_OPTIONS = {
    "3 valve with handle": "Adds a third hydraulic circuit to power basic attachments.",
    "4 valve with handle": "Adds two auxiliary functions for multi-function attachments.",
    "5 valve with handle": "Additional hydraulic circuits for complex attachments.",
    "non-marking tires": "Prevents scuffing on finished indoor floors.",
    "dual tires": "Wider footprint for stability on soft or uneven ground.",
    "dual solid tires": "Puncture-proof, extra stability in debris-prone areas.",
}
FALLBACK_ATTACHMENTS = {
    "sideshifter": "Shift load left/right without moving the truck.",
    "fork positioner": "Adjust fork spread from seat for varied pallet widths.",
    "fork extensions": "Temporarily lengthen forks for long/oversized loads.",
    "paper roll clamp": "Securely handle paper rolls without core damage.",
    "push/pull (slip-sheet)": "Use slip-sheets instead of pallets to cut cost/weight.",
    "carpet pole": "Ram/pole for carpet, coils, and tubing.",
}

def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

def _load_catalog() -> Tuple[Dict[str,str], Dict[str,str]]:
    """
    Look for data/forklift_options_benefits.json.
    Expected shape (simple):
    {
      "options": [{"name":"Non-Marking Tires","blurb":"..."}],
      "attachments": [{"name":"Sideshifter","blurb":"..."}]
    }
    If not present or invalid, fall back to hard-coded dicts above.
    """
    if not os.path.exists(DATA_FILE):
        return FALLBACK_OPTIONS, FALLBACK_ATTACHMENTS
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        opt = {i["name"]: i.get("blurb","") for i in data.get("options", []) if i.get("name")}
        att = {i["name"]: i.get("blurb","") for i in data.get("attachments", []) if i.get("name")}
        if not opt: opt = FALLBACK_OPTIONS
        if not att: att = FALLBACK_ATTACHMENTS
        return opt, att
    except Exception:
        return FALLBACK_OPTIONS, FALLBACK_ATTACHMENTS

OPTIONS, ATTACHMENTS = _load_catalog()

# ---------- Light fuzzy lookup ----------
def fuzzy_lookup(item: str) -> Tuple[str, str, str]:
    """Return (kind, canonical_name, blurb). kind in {'option','attachment',''}"""
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
        k = next((k for k in ATTACHMENTS if "clamp" in _normalize(k)), "paper roll clamp")
        return ("attachment", k, ATTACHMENTS.get(k, "Clamp attachment"))
    if "position" in n:
        return ("attachment", "fork positioner", ATTACHMENTS.get("fork positioner",""))
    if "side shift" in n or "sideshift" in n:
        return ("attachment", "sideshifter", ATTACHMENTS.get("sideshifter",""))
    if "valve" in n:
        return ("option", "3 valve with handle", OPTIONS.get("3 valve with handle",""))
    return ("", "", "")

# ---------- Prompts ----------
SYSTEM_PROMPT = """You are the Options & Attachments expert for Heli forklifts.
STRICT RULES:
- Answer ONLY what the user asked for. 
- If they say “options”, return options only. If they say “attachments”, return attachments only.
- If they name a specific item, give a concise deep-dive: Purpose, Benefits, When to Use, Prerequisites/Valving, Compatibility/Capacity impacts, Trade-offs.
- NEVER output the full catalog unless the user explicitly asks for “all”, “catalog”, or “everything”.
- If the user is ambiguous, ask ONE clarifying question, then stop.
- Keep responses short and skimmable. Bullets > paragraphs.
- Call out capacity/visibility/turning/maintenance impacts when relevant.
"""

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

def _llm(messages, model="gpt-5.1-mini", temperature=0.2) -> str:
    return client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages
    ).choices[0].message.content

def _classify(user_text: str) -> Dict[str, str]:
    out = _llm(
        [
            {"role":"system","content":"Return JSON only. No prose."},
            {"role":"user","content": CLASSIFIER_INSTRUCTION.format(user_text=user_text)}
        ],
        model="gpt-5.1-mini",
        temperature=0.1,
    )
    try:
        return json.loads(out)
    except Exception:
        return {"intent":"unknown","item":""}

def _build_list(kind: str, list_all: bool=False) -> str:
    if kind == "options":
        items = list(OPTIONS.items())
        header = "Options you can add:"
    else:
        items = list(ATTACHMENTS.items())
        header = "Attachments you can use:"
    if not list_all:
        items = items[:10]  # sensible default slice
    lines = [f"- **{name}** — {blurb or '—'}" for name, blurb in items]
    return header + "\n" + "\n".join(lines)

def _detail_prompt(name: str, seed_blurb: str) -> str:
    return f"""Give a concise deep-dive on **{name}** ONLY.
Seed context: {seed_blurb}

Format:
- Purpose:
- Benefits:
- When to use:
- Prerequisites/Valving:
- Compatibility/Capacity impacts:
- Trade-offs:
"""

def respond_options_attachments(user_text: str) -> str:
    # 1) classify intent
    c = _classify(user_text)
    intent = c.get("intent","unknown")
    item = (c.get("item") or "").strip()

    t = _normalize(user_text)
    asked_all = any(w in t for w in [" all ", "catalog", "everything"])

    # 2) route
    if intent == "list_all":
        if "option" in t:
            return _build_list("options", list_all=True)
        if "attachment" in t or "catalog" in t or "everything" in t:
            return _build_list("attachments", list_all=True)
        return "Do you want **all options** or **all attachments**?"

    if intent == "list_options":
        return _build_list("options", list_all=asked_all)

    if intent == "list_attachments":
        return _build_list("attachments", list_all=asked_all)

    if intent == "detail_item":
        kind, key, blurb = fuzzy_lookup(item or user_text)
        if not key:
            return "Which specific item do you mean (e.g., Fork Positioner, Sideshifter, Paper Roll Clamp)?"
        content = _llm(
            [
                {"role":"system","content": SYSTEM_PROMPT},
                {"role":"user","content": _detail_prompt(key, blurb)}
            ],
            model="gpt-5.1",
            temperature=0.2,
        )
        return content

    # Fallback clarifier
    if "attachment" in t:
        return "Do you want a **list of attachments** or details on a **specific attachment**?"
    if "option" in t:
        return "Do you want a **list of options** or details on a **specific option**?"
    return "Do you want **options**, **attachments**, or details on a specific item (e.g., Fork Positioner)?"
