# options_attachments_router.py
# Excel-grounded, *reactive* router that delegates catalog logic to ai_logic.
# Endpoints:
#   GET/POST /api/options                    { "q": "<user question>" }
#   GET/POST /api/options_attachments_chat   { "q": "<user question>" }  (legacy alias)
#   POST     /api/options/reload             (reload Excel-driven caches)
#
# Returns (all endpoints):
#   { "ok": true, "text": "...", "html": "..." }

from __future__ import annotations
from typing import Optional

# Flask (graceful import so the module doesn't crash outside app context)
try:
    from flask import Blueprint, request, jsonify
    options_bp: Optional["Blueprint"] = Blueprint("options_attachments", __name__)
except Exception:
    options_bp = None  # type: ignore

# Single source of truth lives in ai_logic.py
# - render_catalog_sections: smart “reactive” output or list-all when requested
# - recommend_options_from_sheet: (available if you want to extend the router later)
# - refresh_catalog_caches: clears pandas/Excel caches after you update the sheet
from ai_logic import (
    render_catalog_sections,
    recommend_options_from_sheet,  # noqa: F401 (imported for future extensibility)
    refresh_catalog_caches,
)

def _answer_catalog_reactive(user_text: str) -> str:
    """
    Returns a context-aware, *filtered* catalog answer grounded to the Excel sheet.
    - If the user asks to "list/show all" (attachments/options/tires), it lists from Excel.
    - Otherwise, it returns a scenario-aware pick set (Tires + Attachments + Options).
    """
    try:
        # max_per_section keeps answers compact unless user explicitly asks to list all
        return render_catalog_sections(user_text, max_per_section=6)
    except Exception as e:
        # Defensive fallback — never crash your API
        return f"Options/Attachments (fallback): {str(e)}"

def _help_text() -> str:
    return (
        "Ask about options, attachments, or tires — e.g.:\n"
        "- 'Best options for cold warehouse with electric trucks'\n"
        "- 'Attachments for varied pallet widths in tight aisles'\n"
        "- 'List all attachments' or 'show all options'\n"
        "- 'What tire types are available?'"
    )

def _make_payload(q: str) -> dict:
    """
    Render via the reactive helper and return a payload that supports
    both old and new front-ends.
    """
    text = _answer_catalog_reactive(q)
    # Keep 'html' key for clients that expect it; plain text is safest
    return {"ok": True, "text": text, "html": text}

# ─────────────────────────────────────────────────────────────────────────
# PUBLIC EXPORTS (for api_options.py or other callers)
# ─────────────────────────────────────────────────────────────────────────
def respond_options_attachments(user_text: str) -> str:
    """Programmatic entrypoint: return the same text your endpoints would return."""
    return _answer_catalog_reactive(user_text or "")

def reload_catalogs() -> None:
    """Programmatic hot-reload of Excel-backed caches."""
    refresh_catalog_caches()

__all__ = ["options_bp", "respond_options_attachments", "reload_catalogs"]

# ─────────────────────────────────────────────────────────────────────────
# Flask endpoints
# ─────────────────────────────────────────────────────────────────────────
if options_bp is not None:

    # ---- Primary endpoint ------------------------------------------------
    @options_bp.route("/api/options", methods=["GET", "POST"])
    def api_options():
        # Optional hot-reload via ?refresh=1 or body {"refresh": true}
        if request.args.get("refresh") == "1":
            try:
                refresh_catalog_caches()
            except Exception:
                pass

        if request.method == "POST":
            data = request.get_json(silent=True) or {}
            if data.get("refresh") is True:
                try:
                    refresh_catalog_caches()
                except Exception:
                    pass
            q = (data.get("q") or data.get("query") or "").strip()
        else:
            q = (request.args.get("q") or request.args.get("query") or "").strip()

        if not q:
            msg = _help_text()
            return jsonify({"ok": True, "text": msg, "html": msg})

        return jsonify(_make_payload(q))

    # ---- Legacy alias (keeps existing front-ends working) ----------------
    @options_bp.route("/api/options_attachments_chat", methods=["GET", "POST"])
    def api_options_attachments_chat():
        # Same optional refresh flag on legacy path
        if request.args.get("refresh") == "1":
            try:
                refresh_catalog_caches()
            except Exception:
                pass

        if request.method == "POST":
            data = request.get_json(silent=True) or {}
            if data.get("refresh") is True:
                try:
                    refresh_catalog_caches()
                except Exception:
                    pass
            q = (data.get("q") or data.get("query") or "").strip()
        else:
            q = (request.args.get("q") or request.args.get("query") or "").strip()

        if not q:
            msg = _help_text()
            return jsonify({"ok": True, "text": msg, "html": msg})

        return jsonify(_make_payload(q))

    # ---- Hot reload after Excel changes ---------------------------------
    @options_bp.route("/api/options/reload", methods=["POST"])
    def api_options_reload():
        try:
            refresh_catalog_caches()
            return jsonify({"ok": True, "msg": "Catalog caches reloaded."})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
