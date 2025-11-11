# options_attachments_router.py
# Minimal, Excel-grounded router that delegates ALL logic to ai_logic
# (intent parsing, list-all, and scenario-aware recommendations).
# Endpoints:
#   POST /api/options                   { "q": "<user question>" } → {"ok": true, "text": "...", "html": "..."}
#   POST /api/options/reload            → clears caches after you update the Excel
#   POST /api/options_attachments_chat  { "q": "<user question>" } → legacy alias, same response shape
#
# NOTE: Be sure to register this blueprint in your app factory / main file:
#   from options_attachments_router import options_bp
#   app.register_blueprint(options_bp)

from __future__ import annotations
from typing import Optional

try:
    from flask import Blueprint, request, jsonify
    options_bp: Optional["Blueprint"] = Blueprint("options_attachments", __name__)
except Exception:
    options_bp = None  # type: ignore

# Single source of truth lives in ai_logic.py
from ai_logic import render_catalog_sections, refresh_catalog_caches


def _make_payload(q: str) -> dict:
    """
    Renders via ai_logic.render_catalog_sections and returns a payload
    that supports both old and new front-ends.
    - "text": plain text output (safe to drop into <pre> or <div>)
    - "html": same as text (kept for clients that expect an 'html' key)
    """
    text = render_catalog_sections(q)
    return {"ok": True, "text": text, "html": text}


if options_bp is not None:

    # ---- Primary endpoint (new) -----------------------------------------
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
                "text": "Ask about options, attachments, list all, or a specific item (e.g., Fork Positioner).",
                "html": "Ask about options, attachments, list all, or a specific item (e.g., Fork Positioner).",
            })

        return jsonify(_make_payload(q))

    # ---- Legacy alias to avoid 404s from existing UI --------------------
    # Your front-end was calling /api/options_attachments_chat — restore it.
    @options_bp.route("/api/options_attachments_chat", methods=["GET", "POST"])
    def api_options_attachments_chat():
        # Allow same optional refresh flag on legacy path
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
                "text": "Ask about options, attachments, list all, or a specific item (e.g., Fork Positioner).",
                "html": "Ask about options, attachments, list all, or a specific item (e.g., Fork Positioner).",
            })

        return jsonify(_make_payload(q))

    # ---- Hot reload after Excel changes --------------------------------
    @options_bp.route("/api/options/reload", methods=["POST"])
    def api_options_reload():
        try:
            refresh_catalog_caches()
            return jsonify({"ok": True, "msg": "Catalog caches reloaded."})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
