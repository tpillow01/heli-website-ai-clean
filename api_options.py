# api_options.py
from __future__ import annotations

from flask import Blueprint, jsonify, request
from typing import Dict, Any, List

# ⬇️ Your logic (new selector + catalogs)
from ai_logic import load_catalogs, recommend_from_query

# ⬇️ Keep your focused chat & its hot-reload hook
from options_attachments_router import respond_options_attachments, reload_catalogs as router_reload_catalogs

bp_options = Blueprint("bp_options", __name__)  # no url_prefix so paths match your frontend


# ─────────────────────────────────────────────────────────────────────────
# Helper: infer a coarse UI category for /api/options listing
# (intentionally conservative so your existing UI groupings remain stable)
# ─────────────────────────────────────────────────────────────────────────
def _infer_category(name: str) -> str:
    n = (name or "").lower()
    if "tire" in n or "pneumatic" in n or "cushion" in n:
        return "Tires"
    if "valve" in n or "finger control" in n:
        return "Hydraulics / Controls"
    if any(k in n for k in ["light", "beacon", "radar", "ops", "blue spot", "red side"]):
        return "Lighting / Safety"
    if any(k in n for k in ["seat", "cab", "windshield", "wiper", "heater", "air conditioner", "rain-proof"]):
        return "Cab / Comfort"
    if any(k in n for k in ["radiator", "screen", "belly pan", "protection bar", "fan", "filter"]):
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
# GET /api/options → flat list for your UI
# Returns a unified list (options + tires) with {code?, name, benefit, category}
# Note: We build this from load_catalogs() so you don’t depend on load_options().
# ─────────────────────────────────────────────────────────────────────────
@bp_options.get("/api/options")
def list_options() -> Any:
    try:
        options, attachments, tires = load_catalogs()
    except Exception as e:
        return jsonify({"error": f"Failed to load catalogs: {e}"}), 500

    out: List[Dict[str, Any]] = []

    # Options
    for name, benefit in (options or {}).items():
        out.append({
            "code": "",  # kept for backward compatibility (not all rows have codes)
            "name": name,
            "benefit": benefit or "",
            "category": _infer_category(name),
            "type": "option",
        })

    # Tires (many UIs previously showed tires within options; keep that behavior)
    for name, benefit in (tires or {}).items():
        out.append({
            "code": "",
            "name": name,
            "benefit": benefit or "",
            "category": "Tires",
            "type": "tire",
        })

    # Attachments are not included here on purpose (your UI likely lists them elsewhere).
    return jsonify(out)


# ─────────────────────────────────────────────────────────────────────────
# POST /api/recommend → scenario-driven picks (1 tire + short, relevant lists)
# Body: { "query": "<free text>" }
# ─────────────────────────────────────────────────────────────────────────
@bp_options.post("/api/recommend")
def api_recommend() -> Any:
    data: Dict[str, Any] = request.get_json(silent=True) or {}
    q = (data.get("query") or "").strip()
    if not q:
        # Safe default so shape is always consistent
        q = "mixed indoor/outdoor, pallets, busy aisles"

    try:
        sel = recommend_from_query(q, top_attachments=5, top_options=5)
        return jsonify({
            "query": q,
            "context_tags": sel.debug.get("tags", []),
            "tire": [{"name": n, "benefit": b} for (n, b) in sel.tire_primary],  # usually length 1
            "attachments": [{"name": n, "benefit": b} for (n, b) in sel.attachments_top],
            "options": [{"name": n, "benefit": b} for (n, b) in sel.options_top],
        })
    except Exception as e:
        return jsonify({"error": f"Failed to recommend: {e}"}), 500


# ─────────────────────────────────────────────────────────────────────────
# POST /api/options_attachments_chat → Focused chat answer (HTML/plain)
# Body: { "message": "<user text>" }
# Keeps your existing curated router behavior for “list all”, “just options”, etc.
# ─────────────────────────────────────────────────────────────────────────
@bp_options.post("/api/options_attachments_chat")
def options_attachments_chat() -> Any:
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
        answer = respond_options_attachments(user_text)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"answer": f"Error generating response: {e}"}), 500


# ─────────────────────────────────────────────────────────────────────────
# POST /api/reload_catalogs → hot-reload Excel without restart
# Clears ai_logic.load_catalogs() cache and calls your router’s reload for parity.
# ─────────────────────────────────────────────────────────────────────────
@bp_options.post("/api/reload_catalogs")
def api_reload_catalogs() -> Any:
    try:
        # Clear ai_logic cache
        try:
            load_catalogs.cache_clear()  # type: ignore[attr-defined]
        except Exception:
            # If Python version/type checker complains, just ignore
            pass

        # Keep router in sync (it loads the same Excel via your other path)
        try:
            router_reload_catalogs()
        except Exception:
            # Router may be optional — do not fail the endpoint if absent
            pass

        # Touch catalogs once to repopulate cache (and to surface early errors)
        load_catalogs()
        return jsonify({"ok": True, "message": "Catalogs reloaded from Excel."})
    except Exception as e:
        return jsonify({"ok": False, "error": f"Failed to reload catalogs: {e}"}), 500


# ─────────────────────────────────────────────────────────────────────────
# Tiny health probe
# ─────────────────────────────────────────────────────────────────────────
@bp_options.get("/api/health")
def health() -> Any:
    return jsonify({"ok": True})
