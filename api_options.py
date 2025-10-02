# api_options.py
from flask import Blueprint, jsonify, request
from typing import Dict, Any
from ai_logic import load_options, recommend_options_from_sheet

bp_options = Blueprint("bp_options", __name__)

@bp_options.get("/api/options")
def list_options():
    # Returns [{code,name,benefit,category}]
    def infer_category(name: str) -> str:
        n = (name or "").lower()
        if "tire" in n: return "Tires"
        if "valve" in n or "finger control" in n: return "Hydraulics / Controls"
        if any(k in n for k in ["light","beacon","radar","ops","blue spot","red side"]): return "Lighting / Safety"
        if any(k in n for k in ["seat","cab","windshield","wiper","heater","air conditioner","rain-proof"]): return "Cab / Comfort"
        if any(k in n for k in ["radiator","screen","belly pan","protection bar","fan"]): return "Protection / Cooling"
        if "brake" in n: return "Braking"
        if "fics" in n or "fleet" in n: return "Telematics"
        if "cold storage" in n: return "Environment"
        if "lpg" in n or "fuel" in n: return "Fuel / LPG"
        if any(k in n for k in ["overhead guard","lifting eyes"]): return "Chassis / Structure"
        return "Other"

    items = load_options()
    out = [
        {
            "code": o["code"],
            "name": o["name"],
            "benefit": o.get("benefit",""),
            "category": infer_category(o["name"])
        }
        for o in items
    ]
    return jsonify(out)

@bp_options.post("/api/recommend")
def recommend():
    """
    Body: { "query": "<free text about customer needs>" }
    Returns: { "tire": {...} | null, "others": [{...}, ...] }
    """
    data: Dict[str, Any] = request.get_json(silent=True) or {}
    q = (data.get("query") or "").strip()
    rec = recommend_options_from_sheet(q or "mixed indoor/outdoor, 10000 lb")
    return jsonify(rec)
