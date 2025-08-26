# admin_usage.py
import os
import json
import sqlite3
import time
from functools import wraps
from datetime import datetime, timedelta
from flask import Blueprint, request, session, render_template, redirect, url_for, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.getenv("USAGE_DB_PATH", "usage.db")

admin_bp = Blueprint("admin_usage", __name__, template_folder="templates", static_folder="static")

# ---------- Storage ----------
def _conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def _ensure_schema():
    with _conn() as cx:
        cx.execute("""
        CREATE TABLE IF NOT EXISTS usage_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            when_utc TEXT NOT NULL,
            user_id TEXT,
            username TEXT,
            endpoint TEXT NOT NULL,
            action TEXT,
            tokens_prompt INTEGER DEFAULT 0,
            tokens_completion INTEGER DEFAULT 0,
            duration_ms INTEGER DEFAULT 0,
            ip TEXT,
            user_agent TEXT,
            extra_json TEXT
        );
        """)
        cx.execute("CREATE INDEX IF NOT EXISTS idx_usage_when ON usage_events(when_utc);")
        cx.execute("CREATE INDEX IF NOT EXISTS idx_usage_user ON usage_events(username);")
        cx.execute("CREATE INDEX IF NOT EXISTS idx_usage_endpoint ON usage_events(endpoint);")

# ---------- Admin Auth (env-driven) ----------
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
if os.getenv("ADMIN_PASSWORD_HASH"):
    ADMIN_PWHASH = os.getenv("ADMIN_PASSWORD_HASH")
else:
    ADMIN_PWHASH = generate_password_hash(os.getenv("ADMIN_PASSWORD", "admin-usage"))  # one-time hash

def _is_admin():
    return session.get("is_admin") is True

def admin_login_required(fn):
    @wraps(fn)
    def wrap(*args, **kwargs):
        if not _is_admin():
            return redirect(url_for("admin_usage.login"))
        return fn(*args, **kwargs)
    return wrap

# ---------- Public helpers you can call from app.py ----------
def init_admin_usage(app):
    """Call from your app.py to auto-ensure schema and add request timing hooks."""
    _ensure_schema()

    @app.before_request
    def _start_timer():
        # Store start time on request so we can log request duration
        request._usage_t0 = time.perf_counter()

    @app.after_request
    def _auto_request_log(resp):
        # Coarse request logging for every route (avoid static, service-worker, manifest, etc.)
        path = request.path or ""
        if not path.startswith(("/static/", "/service-worker.js", "/manifest.json", "/favicon")):
            try:
                duration_ms = int((time.perf_counter() - getattr(request, "_usage_t0", time.perf_counter())) * 1000)
                record_event(
                    endpoint=path,
                    action=request.method,
                    tokens_prompt=0,
                    tokens_completion=0,
                    duration_ms=duration_ms,
                    extra={"status": resp.status_code}
                )
            except Exception:
                pass
        return resp

def record_event(endpoint: str, action: str = "", tokens_prompt: int = 0, tokens_completion: int = 0,
                 duration_ms: int = 0, extra: dict | None = None):
    """Fine-grained logging (call this inside chat/map endpoints after OpenAI calls)."""
    _ensure_schema()
    user_id = str(session.get("user_id") or "")
    username = str(session.get("username") or "")
    ip = request.headers.get("X-Forwarded-For", request.remote_addr or "")
    ua = request.headers.get("User-Agent", "")
    when_utc = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    extra_json = json.dumps(extra or {}, ensure_ascii=False)

    with _conn() as cx:
        cx.execute("""
        INSERT INTO usage_events
            (when_utc, user_id, username, endpoint, action, tokens_prompt, tokens_completion, duration_ms, ip, user_agent, extra_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (when_utc, user_id, username, endpoint, action, int(tokens_prompt), int(tokens_completion),
              int(duration_ms), ip, ua, extra_json))

def log_model_usage(openai_response, *, endpoint: str, action: str = "chat", duration_ms: int = 0, extra: dict | None = None):
    """Pull token counts from OpenAI responses and record."""
    try:
        usage = getattr(openai_response, "usage", None) or {}
        tokens_prompt = int(getattr(usage, "prompt_tokens", 0) or usage.get("prompt_tokens", 0) or 0)
        tokens_completion = int(getattr(usage, "completion_tokens", 0) or usage.get("completion_tokens", 0) or 0)
    except Exception:
        tokens_prompt = tokens_completion = 0
    record_event(endpoint=endpoint, action=action, tokens_prompt=tokens_prompt,
                 tokens_completion=tokens_completion, duration_ms=duration_ms, extra=extra)

# ---------- Admin routes ----------
@admin_bp.route("/admin/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = (request.form.get("username") or "").strip()
        p = request.form.get("password") or ""
        if u == ADMIN_USERNAME and check_password_hash(ADMIN_PWHASH, p):
            session["is_admin"] = True
            return redirect(url_for("admin_usage.dashboard"))
        flash("Invalid admin credentials", "error")
    return render_template("admin_login.html")

@admin_bp.route("/admin/logout")
def logout():
    session.pop("is_admin", None)
    return redirect(url_for("admin_usage.login"))

@admin_bp.route("/admin")
@admin_login_required
def dashboard():
    # Default range = last 30 days
    end = datetime.utcnow()
    start = end - timedelta(days=30)
    return render_template("admin_dashboard.html",
                           default_start=start.strftime("%Y-%m-%d"),
                           default_end=end.strftime("%Y-%m-%d"))

@admin_bp.route("/admin/usage.json")
@admin_login_required
def usage_api():
    """Return aggregated stats for charts/tables with optional filters."""
    start = request.args.get("start")
    end = request.args.get("end")
    username = (request.args.get("username") or "").strip()
    endpoint = (request.args.get("endpoint") or "").strip()

    q = "SELECT when_utc, username, endpoint, action, tokens_prompt, tokens_completion, duration_ms FROM usage_events WHERE 1=1"
    params = []

    if start:
        q += " AND when_utc >= ?"
        params.append(f"{start}T00:00:00Z")
    if end:
        q += " AND when_utc <= ?"
        params.append(f"{end}T23:59:59Z")
    if username:
        q += " AND username = ?"
        params.append(username)
    if endpoint:
        q += " AND endpoint = ?"
        params.append(endpoint)

    q += " ORDER BY when_utc ASC"

    rows = []
    with _conn() as cx:
        for r in cx.execute(q, params):
            rows.append({
                "when_utc": r[0],
                "username": r[1] or "",
                "endpoint": r[2],
                "action": r[3] or "",
                "tokens_prompt": int(r[4] or 0),
                "tokens_completion": int(r[5] or 0),
                "duration_ms": int(r[6] or 0),
            })

    # Quick aggregates for header tiles
    total_events = len(rows)
    total_prompt = sum(r["tokens_prompt"] for r in rows)
    total_completion = sum(r["tokens_completion"] for r in rows)
    users = sorted(set(r["username"] for r in rows if r["username"]))
    endpoints = sorted(set(r["endpoint"] for r in rows))

    # Daily grouped series
    daily = {}
    for r in rows:
        day = r["when_utc"][:10]  # YYYY-MM-DD
        d = daily.setdefault(day, {"events": 0, "prompt": 0, "completion": 0})
        d["events"] += 1
        d["prompt"] += r["tokens_prompt"]
        d["completion"] += r["tokens_completion"]

    return jsonify({
        "summary": {
            "total_events": total_events,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "unique_users": len(users),
            "unique_endpoints": len(endpoints),
        },
        "users": users,
        "endpoints": endpoints,
        "daily": [{"date": k, **v} for k, v in sorted(daily.items())],
        "rows": rows,
    })
