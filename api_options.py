# api_options.py
from __future__ import annotations

from flask import Blueprint, jsonify, request
from typing import Dict, Any, List

# ---- Pull Excel-driven helpers from your codebase ----
from ai_logic import load_options, recommend_options_from_sheet

# ---- Try to pull the router bits; fall back gracefully if absent ----
try:
    from options_attachments_router import respond_options_attachments as _respond_oa
    _router_err: Exception | None = None
except Exception as e:
    _respond_oa = None
    _router_err = e  # keep for /api/health

bp_options = Blueprint("bp_options", __name__)  # no url_prefix so paths match your frontend

# ─────────────────────────────────────────────────────────────────────────
# Small helper: infer a coarse category for /api/options listing
# ─────────────────────────────────────────────────────────────────────────
def _infer_category(name: str) -> str:
    n = (name or "").lower()
    if "tire" in n:
        return "Tires"
    if "valve" in n or "finger control" in n:
        return "Hydraulics / Controls"
    if any(k in n for k in ["light", "beacon", "radar", "ops", "blue spot", "red side"]):
        return "Lighting / Safety"
    if any(k in n for k in ["seat", "cab", "windshield", "wiper", "heater", "air conditioner", "rain-proof"]):
        return "Cab / Comfort"
    if any(k in n for k in ["radiator", "screen", "belly pan", "protection bar", "fan"]):
        return "Protection / Cooling"
    if "brake" in n:
        return "Braking"
    if "fics" in n or "fleet" in n:
        return "Telematics"
    if "cold storage" in n:
        return "Environment"
    if "lpg" in n or "fuel" in n:
        return "Fuel / LPG"
    if any(k in n for k in ["overhead guard", "lifting eyes"]):
        return "Chassis / Structure"
    return "Other"


# ─────────────────────────────────────────────────────────────────────────
# GET /api/options  → flat list for your UI (code, name, benefit, category)
# ─────────────────────────────────────────────────────────────────────────
@bp_options.get("/api/options")
def list_options():
    try:
        items = load_options()  # [{code, name, benefit}]
    except Exception as e:
        return jsonify({"error": f"Failed to load options: {e}"}), 500

    out: List[Dict[str, Any]] = []
    for o in items or []:
        nm = o.get("name", "")
        out.append({
            "code": o.get("code", ""),
            "name": nm,
            "benefit": o.get("benefit", "") or "",
            "category": _infer_category(nm),
        })
    return jsonify(out)


# ─────────────────────────────────────────────────────────────────────────
# POST /api/recommend  → Excel-driven picks (tire, attachments, options)
# Body: { "query": "<free text>" }
# ─────────────────────────────────────────────────────────────────────────
@bp_options.post("/api/recommend")
def recommend():
    data: Dict[str, Any] = request.get_json(silent=True) or {}
    q = (data.get("query") or "").strip()
    if not q:
        q = "mixed indoor/outdoor, 10000 lb, pallets, busy aisles"

    try:
        rec = recommend_options_from_sheet(q)
        # {"tire": {...}|None, "attachments": [...], "options": [...]}
        return jsonify(rec)
    except Exception as e:
        return jsonify({"error": f"Failed to recommend: {e}"}), 500


# ─────────────────────────────────────────────────────────────────────────
# POST /api/options_attachments_chat  → Focused catalog chat
# Body: { "message": "<user text>" }
# Returns: { "answer": "<plain text or HTML>" }
# Fallback: if router func is missing, synthesize a minimal, useful answer.
# ─────────────────────────────────────────────────────────────────────────
@bp_options.post("/api/options_attachments_chat")
def options_attachments_chat():
    data: Dict[str, Any] = request.get_json(silent=True) or {}
    user_text = (data.get("message") or "").strip()

    if _respond_oa is None:
        # Fallback: run the recommender and format a readable answer.
        try:
            rec = recommend_options_from_sheet(user_text or "indoor, pallets, busy aisles")
            parts = []
            if rec.get("tire"):
                t = rec["tire"]
                parts.append(f"**Tire (recommended):** {t.get('name','')} — {t.get('benefit','')}")
            if rec.get("attachments"):
                atts = "\n".join(f"- {a.get('name','')} — {a.get('benefit','')}" for a in rec["attachments"])
                parts.append(f"**Attachments (relevant):**\n{atts}" if atts else "")
            if rec.get("options"):
                opts = "\n".join(f"- {o.get('name','')} — {o.get('benefit','')}" for o in rec["options"])
                parts.append(f"**Options (relevant):**\n{opts}" if opts else "")
            msg = (
                "_Router fallback active._\n\n" + "\n\n".join(p for p in parts if p)
                if parts else
                "_Router fallback active. No specific matches found._"
            )
            return jsonify({"answer": msg})
        except Exception as e:
            note = f"_Router unavailable; and fallback failed: {e}_"
            return jsonify({"answer": note}), 503

    # Happy path: call the real router’s responder
    if not user_text:
        return jsonify({
            "answer": (
                "Ask about options, attachments, both lists, or describe your use-case "
                "(e.g., “indoor polished floors, tight aisles, heavy pedestrian traffic”)."
            )
        })
    try:
        answer = _respond_oa(user_text)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"answer": f"Error generating response: {e}"}), 500


# ─────────────────────────────────────────────────────────────────────────
# POST /api/reload_catalogs  → hot-reload Excel catalogs without restart
# Tries the router’s reload; falls back to ai_logic’s cache refresh.
# ─────────────────────────────────────────────────────────────────────────
@bp_options.post("/api/reload_catalogs")
def api_reload_catalogs():
    # Try the router’s reload if present
    try:
        from options_attachments_router import reload_catalogs as _reload
        _reload()
        return jsonify({"ok": True, "message": "Catalogs reloaded via router."})
    except Exception:
        # Fallback to ai_logic’s cache refresher
        try:
            from ai_logic import refresh_catalog_caches
            refresh_catalog_caches()
            return jsonify({"ok": True, "message": "Catalogs reloaded via ai_logic."})
        except Exception as e2:
            return jsonify({"ok": False, "error": f"Failed to reload catalogs: {e2}"}), 500


# ─────────────────────────────────────────────────────────────────────────
# Tiny health probe (shows whether router import succeeded)
# ─────────────────────────────────────────────────────────────────────────
@bp_options.get("/api/health")
def health():
    return jsonify({
        "ok": True,
        "router_available": _respond_oa is not None,
        "router_error": (str(_router_err) if _router_err else None)
    })
