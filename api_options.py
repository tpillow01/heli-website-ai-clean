# api_options.py
from __future__ import annotations

from flask import Blueprint, jsonify, request
from typing import Dict, Any, List, Tuple

# ── Imports from your codebase (NO generate_catalog_mode_response here) ──
from ai_logic import load_options, recommend_options_from_sheet
from options_attachments_router import respond_options_attachments, reload_catalogs

bp_options = Blueprint("bp_options", __name__)  # no url_prefix so paths match your frontend

# ─────────────────────────────────────────────────────────────────────────
# Small helper: infer a coarse category for /api/options listing
# (kept in sync with your existing logic so the UI grouping stays stable)
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
        # Provide a safe default so the endpoint always returns shape the UI expects
        q = "mixed indoor/outdoor, 10000 lb, pallets, busy aisles"

    try:
        rec = recommend_options_from_sheet(q)
        # Shape: {"tire": {...}, "attachments": [...], "options": [...]}
        return jsonify(rec)
    except Exception as e:
        return jsonify({"error": f"Failed to recommend: {e}"}), 500


# ─────────────────────────────────────────────────────────────────────────
# POST /api/options_attachments_chat  → Focused chat (no follow-up loops)
# Body: { "message": "<user text>" }
# Returns: { "answer": "<plain text or HTML>" }
# ─────────────────────────────────────────────────────────────────────────
@bp_options.post("/api/options_attachments_chat")
def options_attachments_chat():
    data: Dict[str, Any] = request.get_json(silent=True) or {}
    user_text = (data.get("message") or "").strip()
    if not user_text:
        return jsonify({
            "answer": (
                "Ask about: options, attachments, 'both lists', 'all options', "
                "or a specific item (e.g., Fork Positioner)."
            )
        })

    try:
        # respond_options_attachments handles:
        #  - 'options only' / 'attachments only' / 'both lists' / 'list all'
        #  - environment-aware suggestions when user_text is a scenario
        answer = respond_options_attachments(user_text)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"answer": f"Error generating response: {e}"}), 500


# ─────────────────────────────────────────────────────────────────────────
# POST /api/reload_catalogs  → hot-reload Excel catalogs without restart
# ─────────────────────────────────────────────────────────────────────────
@bp_options.post("/api/reload_catalogs")
def api_reload_catalogs():
    try:
        reload_catalogs()
        return jsonify({"ok": True, "message": "Catalogs reloaded from Excel."})
    except Exception as e:
        return jsonify({"ok": False, "error": f"Failed to reload catalogs: {e}"}), 500


# ─────────────────────────────────────────────────────────────────────────
# Optional tiny health probe so Render can show green checks quickly
# ─────────────────────────────────────────────────────────────────────────
@bp_options.get("/api/health")
def health():
    return jsonify({"ok": True})
