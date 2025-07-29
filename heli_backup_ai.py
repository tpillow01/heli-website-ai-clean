# heli_backup_ai.py  — main web service (with signup/login)
import os
import json
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

# ─── Auth decorator (session‑based) ──────────────────────────────────────
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
        if acct["Account Name"].lower() in low:
            return acct
    # 2) fuzzy fallback
    names = [a["Account Name"] for a in account_data]
    match = difflib.get_close_matches(text, names, n=1, cutoff=0.7)
    if match:
        return next(a for a in account_data if a["Account Name"] == match[0])
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
            # Re-render with a simple error; reuse your template styling
            return render_template("login.html", error="Invalid email or password.", email=email), 401
        # success
        session.permanent = True
        session["user_id"] = user["id"]
        session["email"] = user["email"]
        return redirect(request.args.get("next") or url_for("home"))
    # GET
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
        # auto-login after signup
        user = find_user_by_email(email)
        session["user_id"] = user["id"]
        session["email"] = user["email"]
        return redirect(url_for("home"))
    # GET
    return render_template("signup.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ─── Chat API (protect this too) ────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
@login_required
def chat():
    data = request.get_json() or {}
    user_q = data.get("question", "").strip()
    if not user_q:
        return jsonify({"response": "Please enter a description of the customer’s needs."}), 400

    # Look for a customer in the question
    acct = find_account_by_name(user_q)

    # Build Customer‑profile markup if we matched one
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

    # Let ai_logic format models + profile into a single prompt block
    prompt_ctx = generate_forklift_context(context_input, acct)

    # ── SYSTEM PROMPT ───────────────────────────────────────────────────
    system_prompt = {
        "role": "system",
        "content": (
            "You are a helpful, detailed Heli Forklift sales assistant.\n"
            "When providing customer‑specific data, wrap it in a "
            "<span class=\"section-label\">Customer Profile:</span> section.\n"
            "When recommending models, wrap section headers in a "
            "<span class=\"section-label\">...</span> tag.\n"
            "Use these sections in order if present:\n"
            "Customer Profile:, Model:, Power:, Capacity:, Tire Type:, "
            "Attachments:, Comparison:, Sales Pitch Techniques:, Common Objections:.\n"
            "List details underneath using hyphens and indent sub‑points for clarity.\n\n"
            "Only cite forklift **Model** codes exactly as they appear in the data "
            "(e.g. CPD25, CQD16). Never use only the Series name.\n\n"
            "At the end, include:\n"
            "- <span class=\"section-label\">Sales Pitch Techniques:</span> 1–2 persuasive points.\n"
            "- <span class=\"section-label\">Common Objections:</span> 1–2 common concerns and how to address them.\n"
        )
    }

    messages = [system_prompt, {"role": "user", "content": prompt_ctx}]

    # ── Token‑limit guard (7000 ≈ safe for gpt‑4‑8k) ─────────────────────
    enc = tiktoken.encoding_for_model("gpt-4")
    while sum(len(enc.encode(m["content"])) for m in messages) > 7000 and len(messages) > 2:
        messages.pop(1)

    # ── Call OpenAI ───────────────────────────────────────────────────────
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

# ─── Run locally (Render sets PORT env on deploy) ────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=int(os.getenv("PORT", 5000)))
