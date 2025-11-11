# options_attachments_router.py
# Minimal, Excel-grounded router that delegates ALL logic to ai_logic
# (intent parsing, list-all, and scenario-aware recommendations).
# Endpoints:
#   POST /api/options         { "q": "<user question>" } → {"ok": true, "text": "..."}
#   POST /api/options/reload  → clears caches after you update the Excel

from __future__ import annotations
from typing import Optional

try:
    from flask import Blueprint, request, jsonify
    options_bp: Optional["Blueprint"] = Blueprint("options_attachments", __name__)
except Exception:
    options_bp = None  # type: ignore

from ai_logic import render_catalog_sections, refresh_catalog_caches

if options_bp is not None:

    @options_bp.route("/api/options", methods=["GET", "POST"])
    def api_options():
        # Optional hot-reload via query param (?refresh=1)
        if request.args.get("refresh") == "1":
            try:
                refresh_catalog_caches()
            except Exception:
                pass

        if request.method == "POST":
            data = request.get_json(silent=True) or {}
            q = (data.get("q") or data.get("query") or "").strip()
        else:
            q = (request.args.get("q") or request.args.get("query") or "").strip()

        if not q:
            return jsonify({
                "ok": True,
                "text": "Ask about options, attachments, list all, or a specific item (e.g., Fork Positioner)."
            })

        text = render_catalog_sections(q)  # single source of truth in ai_logic.py
        return jsonify({"ok": True, "text": text})

    @options_bp.route("/api/options/reload", methods=["POST"])
    def api_options_reload():
        try:
            refresh_catalog_caches()
            return jsonify({"ok": True, "msg": "Catalog caches reloaded."})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
