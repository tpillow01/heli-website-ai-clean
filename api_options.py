# api_options.py
from flask import Blueprint, jsonify, request
from typing import Dict, Any

# Pull everything from ai_logic only (no external router)
from ai_logic import (
    load_options,
    recommend_options_from_sheet,
    render_catalog_sections,  # no-followup, plain-text catalog renderer
    parse_catalog_intent,     # optional: debug/inspect intent
    # for hot-reload, we'll clear LRU caches directly:
    load_catalogs,
    load_attachments,
    load_tires_as_options,
)

bp_options = Blueprint("bp_options", __name__)

@bp_options.get("/api/options")
def list_options():
    """
    Returns a flat list of options from the Excel:
    [
      { "code": "...", "name": "...", "benefit": "...", "category": "..." },
      ...
    ]
    """
    def infer_category(name: str) -> str:
        n = (name or "").lower()
        if "tire" in n: return "Tires"
        if "valve" in n or "finger control" in n: return "Hydraulics / Controls"
        if any(k in n for k in ["light", "beacon", "radar", "ops", "blue spot", "red side"]): return "Lighting / Safety"
        if any(k in n for k in ["seat", "cab", "windshield", "wiper", "heater", "air conditioner", "rain-proof"]): return "Cab / Comfort"
        if any(k in n for k in ["radiator", "screen", "belly pan", "protection bar", "fan"]): return "Protection / Cooling"
        if "brake" in n: return "Braking"
        if "fics" in n or "fleet" in n: return "Telematics"
        if "cold storage" in n: return "Environment"
        if "lpg" in n or "fuel" in n: return "Fuel / LPG"
        if any(k in n for k in ["overhead guard", "lifting eyes"]): return "Chassis / Structure"
        return "Other"

    try:
        items = load_options()
    except Exception as e:
        return jsonify({"error": f"Failed to load options: {e}"}), 500

    out = []
    for o in items:
        out.append({
            "code": o.get("code", ""),
            "name": o.get("name", ""),
            "benefit": o.get("benefit", ""),
            "category": infer_category(o.get("name", "")),
        })
    return jsonify(out)

@bp_options.post("/api/recommend")
def recommend():
    """
    Body: { "query": "<free text about customer needs>" }
    Returns: { "tire": {...} | null, "attachments": [...], "options": [...] }
    """
    data: Dict[str, Any] = request.get_json(silent=True) or {}
    q = (data.get("query") or "").strip()
    try:
        rec = recommend_options_from_sheet(q or "mixed indoor/outdoor, 10000 lb")
        return jsonify(rec)
    except Exception as e:
        return jsonify({"error": f"Failed to recommend: {e}"}), 500

@bp_options.post("/api/catalog_text")
def catalog_text():
    """
    Focused "catalog mode" text response WITHOUT follow-up prompts.

    Body: { "message": "<user text>" }

    Behavior:
      - If user asks for "options only", returns only the Options section
      - If "attachments only", returns only Attachments
      - If "both" or "attachments and options", returns those two sections
      - If "list all", includes Tires as well (full catalog)
      - Otherwise, returns scenario-aware curated (Tires, Attachments, Options)
    """
    data: Dict[str, Any] = request.get_json(silent=True) or {}
    user_text = (data.get("message") or "").strip()
    if not user_text:
        return jsonify({"answer": "Ask about options, attachments, both lists, or say 'list all'."})

    try:
        answer = render_catalog_sections(user_text, max_per_section=6)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"answer": f"Error generating response: {e}"}), 500

@bp_options.post("/api/reload_catalogs")
def api_reload_catalogs():
    """
    Hot-reload the Excel-backed catalogs (options, attachments, tires).
    Clears the LRU caches inside ai_logic so fresh spreadsheet edits show up.
    """
    try:
        # Clear all relevant caches
        try:
            load_catalogs.cache_clear()
        except Exception:
            pass
        try:
            load_options.cache_clear()
        except Exception:
            pass
        try:
            load_attachments.cache_clear()
        except Exception:
            pass
        try:
            load_tires_as_options.cache_clear()
        except Exception:
            pass

        # Touch once to log/prime after clear (optional)
        _ = load_options()
        return jsonify({"ok": True, "message": "Catalogs reloaded from Excel."})
    except Exception as e:
        return jsonify({"ok": False, "error": f"Failed to reload catalogs: {e}"}), 500
