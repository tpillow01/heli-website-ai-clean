# heli_backup_ai.py  — main web service (with signup/login)
import os
import json
from data_sources import make_inquiry_targets, find_inquiry_rows
import difflib
import sqlite3
from datetime import timedelta
import tiktoken

from flask import (
    Flask, render_template, request, jsonify, Response,
    redirect, url_for, session
)
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from openai import OpenAI

from ai_logic import generate_forklift_context   # ← your helper file

# ─── Flask & OpenAI client ───────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-insecure")  # set in env for prod
app.permanent_session_lifetime = timedelta(days=7)
# Optional: tighten cookies a bit
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=bool(os.getenv("SESSION_COOKIE_SECURE", "1") == "1"),
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ─── USERS DB (simple SQLite) ────────────────────────────────────────────
USERS_DB_PATH = os.getenv("USERS_DB_PATH", "heli_users.db")

def get_user_db():
    conn = sqlite3.connect(USERS_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_user_db():
    conn = get_user_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_user_db()

def find_user_by_email(email: str):
    conn = get_user_db()
    row = conn.execute("SELECT * FROM users WHERE email = ?", (email.lower(),)).fetchone()
    conn.close()
    return row

def create_user(email: str, password: str):
    conn = get_user_db()
    conn.execute(
        "INSERT INTO users (email, password_hash) VALUES (?, ?)",
        (email.lower().strip(), generate_password_hash(password))
    )
    conn.commit()
    conn.close()

# ─── Auth decorator (session-based) ──────────────────────────────────────
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login", next=request.path))
        return f(*args, **kwargs)
    return wrapper

# ─── JSON DATA LOAD (once at startup) ────────────────────────────────────
with open("accounts.json", "r", encoding="utf-8") as f:
    account_data = json.load(f)
print(f"✅ Loaded {len(account_data)} accounts from JSON")

with open("models.json", "r", encoding="utf-8") as f:
    model_data = json.load(f)
print(f"✅ Loaded {len(model_data)} models from JSON")

# ─── Helper: substring first, then fuzzy company match  ──────────────────
def find_account_by_name(text: str):
    low = text.lower()
    # 1) direct substring
    for acct in account_data:
        name = str(acct.get("Account Name", "")).lower()
        if name and name in low:
            return acct
    # 2) fuzzy fallback
    names = [a.get("Account Name", "") for a in account_data if a.get("Account Name")]
    match = difflib.get_close_matches(text, names, n=1, cutoff=0.7)
    if match:
        return next(a for a in account_data if a.get("Account Name") == match[0])
    return None

# ─── Web routes (login/signup + app) ─────────────────────────────────────
@app.route("/")
@login_required
def home():
    # your chat UI
    return render_template("chat.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        user = find_user_by_email(email)
        if not user or not check_password_hash(user["password_hash"], password):
            return render_template("login.html", error="Invalid email or password.", email=email), 401
        session.permanent = True
        session["user_id"] = user["id"]
        session["email"] = user["email"]
        return redirect(request.args.get("next") or url_for("home"))
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        confirm = request.form.get("confirm") or ""
        if not email or not password:
            return render_template("signup.html", error="Email and password are required.", email=email), 400
        if password != confirm:
            return render_template("signup.html", error="Passwords do not match.", email=email), 400
        if find_user_by_email(email):
            return render_template("signup.html", error="This email is already registered.", email=email), 409
        try:
            create_user(email, password)
        except Exception as e:
            return render_template("signup.html", error=f"Could not create user: {e}", email=email), 500
        user = find_user_by_email(email)
        session["user_id"] = user["id"]
        session["email"] = user["email"]
        return redirect(url_for("home"))
    return render_template("signup.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ─── Chat API (now accepts mode/target_id; behavior unchanged) ───────────
@app.route("/api/chat", methods=["POST"])
@login_required
def chat():
    data = request.get_json() or {}
    user_q   = (data.get("question")  or "").strip()
    mode     = (data.get("mode")      or "recommendation").lower()

    if not user_q:
        return jsonify({"response": "Please enter a description of the customer’s needs."}), 400

    app.logger.info(f"/api/chat mode={mode} qlen={len(user_q)}")

    # ───────────── INQUIRY MODE ─────────────
    if mode == "inquiry":
        from data_sources import build_inquiry_brief, find_customer_id_by_name
        # Try to extract a customer name/id from the question
        probe = find_customer_id_by_name(user_q) or user_q
        brief = build_inquiry_brief(probe)

        if not brief:
            return jsonify({"response": "I couldn’t locate that customer in the report/billing data. Please mention the company name as it appears in your system (e.g., 'What’s the best strategy for Acme Steel?')."})

        # Strong, structured system prompt for strategy brief
        system_prompt = {
            "role": "system",
            "content": (
                "You are a sales strategist for a forklift dealership. Use the INQUIRY context verbatim; do not invent numbers. "
                "Output a concise, organized brief for a sales rep arriving on-site. Use these sections and keep each to a few bullet points:\n\n"
                "1) Segmentation — Show Account Size (A/B/C/D) and Relationship (1/2/3/P) and what each means.\n"
                "2) Current Pattern — When they spend (top months), what they buy (top offerings), and how frequently (avg days between invoices).\n"
                "3) Visit Plan — What to lead with (Service / Parts / Rental / New/Used) and why, grounded in recent billing.\n"
                "4) Move from X→Y — If they are, for example, B3, give concrete steps to reach B1: which additional offerings to add to reach 4+; quick wins this quarter.\n"
                "5) Next Actions — 3 short, specific tasks to do today.\n\n"
                "Be practical. Prefer checklists. If any data is missing, state that plainly and proceed with best-guess actions."
            )
        }

        messages = [
            system_prompt,
            {"role": "system", "content": brief["context_block"]},
            {"role": "user",   "content": user_q}
        ]

        # Call OpenAI
        try:
            resp = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=700,
                temperature=0.4
            )
            ai_reply = resp.choices[0].message.content.strip()
        except Exception as e:
            ai_reply = f"❌ Internal error: {e}"

        # Tag response with current tier (handy for the rep)
        tag = f"[Segmentation: {brief['size_letter']}{brief['relationship_code']}]"
        return jsonify({"response": f"{tag}\n\n{ai_reply}"})

    # ───────── RECOMMENDATION MODE (existing flow) ─────────
    # your existing logic below remains as-is
    acct = find_account_by_name(user_q)
    context_input = user_q
    if acct:
        profile_ctx = (
            "<span class=\"section-label\">Customer Profile:</span>\n"
            f"- Company: {acct.get('Account Name')}\n"
            f"- Industry: {acct.get('Industry', 'N/A')}\n"
            f"- SIC Code: {acct.get('SIC Code', 'N/A')}\n"
            f"- Fleet Size: {acct.get('Total Company Fleet Size', 'N/A')}\n"
            f"- Truck Types: {acct.get('Truck Types at Location', 'N/A')}\n\n"
        )
        context_input = profile_ctx + user_q

    prompt_ctx = generate_forklift_context(context_input, acct)

    system_prompt = {
        "role": "system",
        "content": (
            "You are a helpful, detailed Heli Forklift sales assistant.\n"
            "When providing customer-specific data, wrap it in a "
            "<span class=\"section-label\">Customer Profile:</span> section.\n"
            "When recommending models, wrap section headers in a "
            "<span class=\"section-label\">...</span> tag.\n"
            "Use these sections in order if present:\n"
            "Customer Profile:, Model:, Power:, Capacity:, Tire Type:, "
            "Attachments:, Comparison:, Sales Pitch Techniques:, Common Objections:.\n"
            "List details underneath using hyphens and indent sub-points for clarity.\n\n"
            "Only cite forklift **Model** codes exactly as they appear in the data "
            "(e.g. CPD25, CQD16). Never use only the Series name.\n\n"
            "At the end, include:\n"
            "- <span class=\"section-label\">Sales Pitch Techniques:</span> 1–2 persuasive points.\n"
            "- <span class=\"section-label\">Common Objections:</span> 1–2 common concerns and how to address them.\n"
        )
    }

    messages = [system_prompt, {"role": "user", "content": prompt_ctx}]

    # (keep your tiktoken guard if you want)
    try:
        resp = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=600,
            temperature=0.7
        )
        ai_reply = resp.choices[0].message.content.strip()
    except Exception as e:
        ai_reply = f"❌ Internal error: {e}"

    return jsonify({"response": ai_reply})

# ─── Data for dropdowns ─────────────────────────────────────────────────
@app.route("/api/modes")
def api_modes():
    return jsonify([
        {"id": "recommendation", "label": "Forklift Recommendation"},
        {"id": "inquiry",        "label": "Customer Inquiry"}
    ])

@app.route("/api/targets")
def api_targets():
    """
    Returns dropdown choices based on mode:
      - ?mode=inquiry          -> customers built from the two CSVs
      - ?mode=recommendation   -> accounts from accounts.json
      - default (missing mode) -> recommendation
    """
    mode = (request.args.get("mode") or "recommendation").lower()

    if mode == "inquiry":
        return jsonify(make_inquiry_targets())

    # recommendation: use already-loaded account_data, label from "Account Name"
    items = []
    for i, a in enumerate(account_data):
        label = a.get("Account Name") or f"Account {i+1}"
        _id   = str(a.get("Account Name", label))  # use name as id if no explicit id
        items.append({"id": _id, "label": label})

    return jsonify(items)

# ─── Service worker at site root (so scope is '/') ───────────────────────
@app.route('/service-worker.js')
def service_worker():
    return app.send_static_file('service-worker.js')

# ─── Run locally (Render sets PORT env on deploy) ────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=int(os.getenv("PORT", 5000)))
