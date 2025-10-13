# api_options.py
from flask import Blueprint, jsonify, request
from typing import Dict, Any

# Your existing helpers
from ai_logic import load_options, recommend_options_from_sheet

# Router + hot-reload for catalogs
from options_attachments_router import respond_options_attachments, reload_catalogs

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
    Returns: { "tire": {...} | null, "others": [{...}, ...] }
    """
    data: Dict[str, Any] = request.get_json(silent=True) or {}
    q = (data.get("query") or "").strip()
    try:
        rec = recommend_options_from_sheet(q or "mixed indoor/outdoor, 10000 lb")
        return jsonify(rec)
    except Exception as e:
        return jsonify({"error": f"Failed to recommend: {e}"}), 500

@bp_options.post("/api/options_attachments_chat")
def options_attachments_chat():
    """
    Focused chat for Options & Attachments mode.

    Body: { "message": "<user text>" }

    Behavior:
      - If user asks for options -> returns options only.
      - If user asks for attachments -> returns attachments only.
      - If user asks for "all" -> returns the full catalog (no filtering).
      - If user names a specific item -> returns a concise deep-dive with labeled sections.
      - Otherwise -> environment-aware curated lists.

    Returns: { "answer": "<html/text>" }
    """
    data: Dict[str, Any] = request.get_json(silent=True) or {}
    user_text = (data.get("message") or "").strip()
    if not user_text:
        return jsonify({"answer": "Ask about options, attachments, all items, or a specific item (e.g., Fork Positioner)."})
    try:
        answer = respond_options_attachments(user_text)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"answer": f"Error generating response: {e}"}), 500

@bp_options.post("/api/reload_catalogs")
def api_reload_catalogs():
    """
    Hot-reload the Excel-backed catalogs (options, attachments, tires).
    Call this after you edit the spreadsheet, so the API reflects updates without a restart.
    """
    try:
        reload_catalogs()
        return jsonify({"ok": True, "message": "Catalogs reloaded from Excel."})
    except Exception as e:
        return jsonify({"ok": False, "error": f"Failed to reload catalogs: {e}"}), 500
