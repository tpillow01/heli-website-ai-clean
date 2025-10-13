# options_attachments_router.py
# Excel-grounded Options & Attachments router with robust "ALL" handling and loop-proof clarifiers.
# - "all ..." returns complete catalog from Excel (no filtering/curation).
# - Normal queries are environment-aware (indoor/outdoor/pedestrians/long loads/etc.).
# - Short replies like "both lists", "options only", "attachments only" are supported.
# - No markdown bold/italics; uses <span class="section-label">...</span> headers.

import os
import re
import json
from math import sqrt
from typing import Dict, Tuple, List
from openai import OpenAI

# ---------------- OpenAI client & models ----------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL_CLASSIFIER = os.getenv("OA_CLASSIFIER_MODEL", "gpt-4o-mini")
MODEL_RESPONDER  = os.getenv("OA_RESPONDER_MODEL",  "gpt-4o-mini")
MODEL_SELECTOR   = os.getenv("OA_SELECTOR_MODEL",   "gpt-4o-mini")
EMBED_MODEL      = os.getenv("OA_EMBED_MODEL",      "text-embedding-3-small")

# ---------------- Excel loaders from ai_logic ----------------
try:
    from ai_logic import load_options as _load_options_catalog
except Exception:
    _load_options_catalog = None

try:
    from ai_logic import load_attachments as _load_attachments_catalog
except Exception:
    _load_attachments_catalog = None

# Tires may be separate
try:
    from ai_logic import load_tires_as_options as _load_tires_catalog
except Exception:
    _load_tires_catalog = None
try:
    if _load_tires_catalog is None:
        from ai_logic import load_tires as _load_tires_catalog
except Exception:
    _load_tires_catalog = None

# ---------------- Fallbacks (safety only) ----------------
FALLBACK_OPTIONS = {
    "3 Valve with Handle": "Adds a third hydraulic circuit to power basic attachments.",
    "4 Valve with Handle": "Two auxiliary circuits for multi-function attachments.",
    "5 Valve with Handle": "Additional hydraulic circuits for complex attachments.",
}
FALLBACK_ATTACHMENTS = {
    "Sideshifter":     "Shift load left/right without moving the truck.",
    "Fork Positioner": "Adjust fork spread from the seat for mixed pallet widths.",
}
FALLBACK_TIRES = {
    "Non-Marking Tires":     "Prevents scuffing on indoor finished floors.",
    "Solid Pneumatic Tires": "Puncture resistance in debris-prone areas.",
}

# ---------------- Small utils ----------------
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

# ---------------- Catalog hydrate + hot reload ----------------
OPTIONS: Dict[str, str] = {}
ATTACHMENTS: Dict[str, str] = {}
TIRES: Dict[str, str] = {}

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

    # TIRES
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

def reload_catalogs() -> None:
    """Hot-reload catalogs from Excel (call this from an endpoint after updating your sheet)."""
    global OPTIONS, ATTACHMENTS, TIRES
    OPTIONS, ATTACHMENTS, TIRES = _hydrate_catalogs()

# initial load
reload_catalogs()

def _merged_options() -> Dict[str, str]:
    # "Options" + Tires unified for selection; in ALL mode we still list everything.
    merged = dict(OPTIONS)
    for k, v in (TIRES or {}).items():
        if k not in merged:
            merged[k] = v
    return merged

# ---------------- Fuzzy item lookup (deep dives) ----------------
def fuzzy_lookup(item: str):
    n = _normalize(item)
    merged = _merged_options()
    for k, v in merged.items():
        if n == _normalize(k) or n in _normalize(k):
            return ("option", k, v)
    for k, v in ATTACHMENTS.items():
        if n == _normalize(k) or n in _normalize(k):
            return ("attachment", k, v)
    # loose hints
    if "position" in n or "positioner" in n:
        k = next((kk for kk in ATTACHMENTS if "position" in _normalize(kk)), "Fork Positioner")
        return ("attachment", k, ATTACHMENTS.get(k, ""))
    if "side shift" in n or "sideshift" in n or "side-shift" in n:
        k = next((kk for kk in ATTACHMENTS if "side" in _normalize(kk) and "shift" in _normalize(kk)), "Sideshifter")
        return ("attachment", k, ATTACHMENTS.get(k, ""))
    if "valve" in n or "aux" in n:
        k = next((kk for kk in merged if "valve" in _normalize(kk)), "3 Valve with Handle")
        return ("option", k, merged.get(k, ""))
    return ("", "", "")

# ---------------- Prompts ----------------
SYSTEM_PROMPT = (
    "You are the Options & Attachments expert for Heli forklifts.\n"
    "Rules:\n"
    "- Answer ONLY what the user asked for.\n"
    "- If they say 'options', return options only. If they say 'attachments', return attachments only.\n"
    "- If they name a specific item, give a concise deep-dive with labeled sections.\n"
    "- Never output the full catalog unless they explicitly ask for 'all', 'catalog', or 'everything'.\n"
    "- Bullets > paragraphs.\n"
)

CLASSIFIER_INSTRUCTION = (
    "Classify the request into exactly one of:\n"
    "- list_options | list_attachments | detail_item | list_all | both_lists | unknown\n\n"
    "Return JSON only: {\"intent\":\"...\", \"item\":\"<name or ''>\"}\n\n"
    "User: {user_text}\n"
)

# ---------------- Environment cues (normal queries) ----------------
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

# ---------------- Embeddings & semantic rank (normal) ----------------
def _embed(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def _cos(a, b) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na = sum(x*x for x in a) ** 0.5
    nb = sum(y*y for y in b) ** 0.5
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
    qv, iv = vecs[0], vecs[1:]
    sims = [(items[i][0], _cos(qv, iv[i])) for i in range(len(items))]
    sims.sort(key=lambda x: (-x[1], _normalize(x[0])))
    return [n for n,_ in sims[:top_k]]

# ---------------- Tires (curation only for non-ALL) ----------------
_TIRE_WORDS = ("tire", "tyre", "pneumatic", "cushion", "non-marking", "solid", "dual")
def _is_tire(name: str, blurb: str = "") -> bool:
    s = f"{name} {blurb}".lower()
    return any(w in s for w in _TIRE_WORDS)

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

def _curate_tires(names: List[str], pool: Dict[str, str], cues: Dict[str, bool]) -> List[str]:
    tires = [(n, _score_tire(n, pool.get(n, ""), cues)) for n in names if _is_tire(n, pool.get(n, ""))]
    non_tires = [n for n in names if not _is_tire(n, pool.get(n, ""))]
    tires_sorted = [n for n,_ in sorted(tires, key=lambda x: (-x[1], _normalize(x[0])))]
    top_tires = tires_sorted[:2]
    return [n for n in names if n in non_tires or n in top_tires]

# ---------------- LLM helpers ----------------
def _llm(messages, model: str, temperature: float = 0.2) -> str:
    resp = client.chat.completions.create(model=model, temperature=temperature, messages=messages)
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

# ---- Final grounded selector (choose only from given candidates) ----
_SELECTOR_INSTRUCTION = (
    "Choose the most relevant forklift {kind} for the user's scenario, using general forklift knowledge. "
    "You MUST select only from the CANDIDATES provided. Do NOT invent new items.\n"
    "Return JSON only: {\"items\": [\"Exact Name 1\", \"Exact Name 2\", ...]}\n\n"
    "SCENARIO:\n{query}\n\n"
    "CANDIDATES ({count}):\n{candidates}\n"
)

def _format_candidates_list(names: List[str], pool: Dict[str, str]) -> str:
    return "\n".join(f"- {n}: {pool.get(n, '')}" for n in names)

def _llm_pick_from(names: List[str], pool: Dict[str, str], kind: str, query: str) -> List[str]:
    """
    Ask the model to choose from 'names' only. Returns a filtered list of selected names.
    """
    if not names:
        return []
    try:
        out = _llm(
            [
                {"role": "system", "content": "Return JSON only. No explanations."},
                {"role": "user", "content": _SELECTOR_INSTRUCTION.format(
                    kind=kind,
                    query=query,
                    count=len(names),
                    candidates=_format_candidates_list(names, pool),
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
        # On any error, return empty and let the caller use its fallback
        return []

# ---------------- Loop-proof classifier ----------------
def _is_short_reply(text: str) -> str:
    t = _normalize(text)
    shorts = {
        "both": "both_lists",
        "both list": "both_lists",
        "both lists": "both_lists",
        "options": "list_options",
        "options only": "list_options",
        "attachments": "list_attachments",
        "attachments only": "list_attachments",
        "specific item": "detail_item",
        "detail": "detail_item",
    }
    return shorts.get(t, "")

def _classify(user_text: str) -> Dict[str, str]:
    traw = (user_text or "").strip()
    t = f" {traw.lower()} "

    # short replies
    sr = _is_short_reply(traw)
    if sr:
        return {"intent": sr, "item": ""}

    # "both" style
    if re.search(r"\bboth( lists?)?\b", t) or " bith " in t or " options and attachments " in t or " attachments and options " in t:
        return {"intent":"both_lists","item":""}

    # explicit ALL (covers "all attachments and options" etc.)
    if any(sig in t for sig in [" catalog", " everything", " all ", " full list", " complete list"]):
        has_opt = " option" in t
        has_att = " attachment" in t
        if has_opt and has_att:  # “all attachments and options”
            return {"intent":"list_all","item":"both"}
        if has_opt:
            return {"intent":"list_all","item":"options"}
        if has_att:
            return {"intent":"list_all","item":"attachments"}
        return {"intent":"list_all","item":"both"}

    # specific item?
    kind, key, _ = fuzzy_lookup(traw)
    if key:
        return {"intent":"detail_item","item":key}

    # explicit mentions
    has_opt = " option" in t
    has_att = " attachment" in t
    if has_opt and has_att:
        return {"intent":"both_lists","item":""}
    if has_opt:
        return {"intent":"list_options","item":""}
    if has_att:
        return {"intent":"list_attachments","item":""}

    # scenario → both by default
    cues = _extract_cues(traw)
    if any(cues.values()):
        return {"intent":"both_lists","item":""}

    # last resort
    try:
        guess = _classify_llm(traw)
        if guess.get("intent") and guess["intent"] != "unknown":
            return guess
    except Exception:
        pass
    return {"intent":"both_lists","item":""}

# ---------------- Output formatting ----------------
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
    return _HEADER_PATTERN.sub(repl, s)

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
    lines = [f"- {n} — {d.get(n, '') or '—'}" for n in names]
    out = header_html + "\n" + "\n".join(lines)
    return _strip_all_md_emphasis(out)

def _detail_prompt(name: str, seed_blurb: str) -> str:
    return (
        "Give a concise deep-dive on this SINGLE item.\n"
        "Use the exact headers below and bullet points. Do NOT use bold or italics.\n\n"
        f"Item: {name}\n\n"
        "Purpose:\n- \n\n"
        "Benefits:\n- \n- \n- \n\n"
        "When to use:\n- \n- \n\n"
        "Prerequisites/Valving:\n- \n- \n\n"
        "Compatibility/Capacity impacts:\n- \n- \n\n"
        "Trade-offs:\n- \n- \n\n"
        f"Helpful context: {seed_blurb}\n"
    )

# ---------------- Public entry point ----------------
def respond_options_attachments(user_text: str) -> str:
    if not (user_text or "").strip():
        return "Ask about options, attachments, or a specific item (e.g., Fork Positioner)."

    # Detect "all" from RAW text
    raw = f" {(user_text or '').lower()} "
    asked_all = any(sig in raw for sig in [" all ", " catalog", " everything", " full list", " complete list"])

    c = _classify(user_text)
    intent = c.get("intent", "both_lists")
    which_all = (c.get("item") or "").lower().strip()  # "options" | "attachments" | "both" | ""

    opt_pool = _merged_options()
    att_pool = ATTACHMENTS

    # -------- LIST ALL (true full dump from Excel) --------
    if intent == "list_all" or asked_all:
        show_opts = (which_all in ("options","both","")) and (" attachment" not in raw or " option" in raw or which_all != "attachments")
        show_atts = (which_all in ("attachments","both","")) and (" option" not in raw or " attachment" in raw or which_all != "options")

        parts = []
        if show_opts:
            names = sorted(opt_pool.keys(), key=lambda k: _normalize(k))
            parts.append(_build_list("Options", names, opt_pool))
        if show_atts:
            names = sorted(att_pool.keys(), key=lambda k: _normalize(k))
            parts.append(_build_list("Attachments", names, att_pool))

        if not parts:  # ambiguous "all"
            names_o = sorted(opt_pool.keys(), key=lambda k: _normalize(k))
            names_a = sorted(att_pool.keys(), key=lambda k: _normalize(k))
            parts = [
                _build_list("Options", names_o, opt_pool),
                _build_list("Attachments", names_a, att_pool),
            ]
        return "\n\n".join(parts)

    # -------- BOTH LISTS (environment-aware) --------
    if intent == "both_lists":
        cues = _extract_cues(user_text)

        opt_pref = _rank_semantic(opt_pool, user_text, extra_context="Select options (including tires) for this scenario.", top_k=30)
        att_pref = _rank_semantic(att_pool, user_text, extra_context="Select attachments for this scenario.", top_k=30)

        # Lightweight nudges
        def _nudge(names: List[str], pool: Dict[str, str], kind: str) -> List[str]:
            if kind == "options" and (cues.get("indoor") or cues.get("pedestrians")):
                pri = ["camera", "radar", "ops", "operator presence", "speed", "finger", "blue", "beacon", "horn", "mirror", "non-marking", "cushion", "led"]
            elif kind == "attachments" and (cues.get("indoor") or cues.get("precision")):
                pri = ["sideshift", "positioner", "load backrest"]
            else:
                pri = []
            def score(n):
                s = f"{n} {pool.get(n,'')}".lower()
                hit = any(p in s for p in pri)
                return (1 if hit else 0, _normalize(n))
            return [n for n,_ in sorted([(n, score(n)) for n in names], key=lambda x: (-x[1][0], x[1][1]))]

        opt_pref = _nudge(opt_pref, opt_pool, "options")
        att_pref = _nudge(att_pref, att_pool, "attachments")

        # Final LLM choose from candidates
        opt_names = _llm_pick_from(opt_pref, opt_pool, "options (incl. tires)", user_text)
        att_names = _llm_pick_from(att_pref, att_pool, "attachments", user_text)

        # Curate tire count for normal answers
        if opt_names:
            opt_names = _curate_tires(opt_names, opt_pool, _extract_cues(user_text))

        # Hard fallback if empty (no loops)
        if not opt_names and not att_names:
            opt_keywords = ["camera", "radar", "ops", "operator presence", "speed", "finger", "blue", "beacon", "horn", "mirror", "non-marking", "cushion", "led"]
            att_keywords = ["sideshift", "positioner", "load backrest"]

            def pick(pool: Dict[str,str], keys: List[str], limit: int) -> List[str]:
                out = []
                for n,b in pool.items():
                    s = f"{n} {b}".lower()
                    if any(k in s for k in keys): out.append(n)
                seen, uniq = set(), []
                for n in out:
                    if n not in seen:
                        seen.add(n); uniq.append(n)
                return uniq[:limit]

            opt_names = pick(opt_pool, opt_keywords, 8)
            att_names = pick(att_pool, att_keywords, 5)

        parts = [
            _build_list("Options", opt_names, opt_pool),
            _build_list("Attachments", att_names, att_pool),
        ]
        return "\n\n".join(parts)

    # -------- OPTIONS ONLY (normal) --------
    if intent == "list_options":
        if asked_all:
            names = sorted(opt_pool.keys(), key=lambda k: _normalize(k))
            return _build_list("Options", names, opt_pool)
        cues = _extract_cues(user_text)
        pref = _rank_semantic(opt_pool, user_text, extra_context="Select options (including tires) for this scenario.", top_k=30)
        names = _llm_pick_from(pref, opt_pool, "options (incl. tires)", user_text)
        if names:
            names = _curate_tires(names, opt_pool, cues)
        if not names:
            return 'Want me to include attachments too, or show all options? (say "both lists" or "all options")'
        return _build_list("Options", names, opt_pool)

    # -------- ATTACHMENTS ONLY (normal) --------
    if intent == "list_attachments":
        if asked_all:
            names = sorted(att_pool.keys(), key=lambda k: _normalize(k))
            return _build_list("Attachments", names, att_pool)
        pref = _rank_semantic(att_pool, user_text, extra_context="Select attachments for this scenario.", top_k=30)
        names = _llm_pick_from(pref, att_pool, "attachments", user_text)
        if not names:
            return 'Want me to include options too, or show all attachments? (say "both lists" or "all attachments")'
        return _build_list("Attachments", names, att_pool)

    # -------- DETAIL ITEM --------
    if intent == "detail_item":
        item = (c.get("item") or "").strip()
        if not item:
            _, key, blurb = fuzzy_lookup(user_text)
        else:
            _, key, blurb = fuzzy_lookup(item)
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
                "Purpose:\n- Attachment/option for specialized handling.\n\n"
                "Benefits:\n- Faster handling.\n- Safer adjustments.\n- Better load fit.\n\n"
                "When to use:\n- Mixed load profiles or frequent changes.\n\n"
                "Prerequisites/Valving:\n- May require 3rd/4th hydraulic function.\n\n"
                "Compatibility/Capacity impacts:\n- Minor capacity derate; verify on data plate.\n\n"
                "Trade-offs:\n- Higher upfront cost and modest maintenance."
            )
            fallback = _decorate_headers(fallback, title=f"{key} — Deep Dive")
            return _strip_all_md_emphasis(fallback)

    # final fallback: both lists (no loop)
    return respond_options_attachments(user_text + " (both lists)")
