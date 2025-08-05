# heli_backup_ai.py — main web service (with signup/login)
import os
import json
import difflib
import sqlite3
from datetime import timedelta

from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from openai import OpenAI

from ai_logic import generate_forklift_context
from data_sources import make_inquiry_targets  # used by /api/targets

# ─── Flask & OpenAI client ───────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-insecure")
app.permanent_session_lifetime = timedelta(days=7)
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

# ─── Auth decorator ──────────────────────────────────────────────────────
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

def find_account_by_name(text: str):
    low = text.lower()
    for acct in account_data:
        name = str(acct.get("Account Name", "")).lower()
        if name and name in low:
            return acct
    names = [a.get("Account Name", "") for a in account_data if a.get("Account Name")]
    match = difflib.get_close_matches(text, names, n=1, cutoff=0.7)
    if match:
        return next(a for a in account_data if a.get("Account Name") == match[0])
    return None

# ─── Web routes ──────────────────────────────────────────────────────────
@app.route("/")
@login_required
def home():
    return render_template("chat.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = (request.form.get("password") or "")
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
        password = (request.form.get("password") or "")
        confirm  = (request.form.get("confirm") or "")
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

# ─── Chat API ────────────────────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
@login_required
def chat():
    data   = request.get_json(force=True) or {}
    user_q = (data.get("question") or "").strip()
    mode   = (data.get("mode") or "recommendation").lower()

    if not user_q:
        return jsonify({"response": "Please enter a description of the customer’s needs."}), 400

    app.logger.info(f"/api/chat mode={mode} qlen={len(user_q)}")

    # ───────── Inquiry mode ─────────
    if mode == "inquiry":
        from data_sources import build_inquiry_brief, make_inquiry_targets, _norm_name

        qnorm = _norm_name(user_q)

        # try to lock onto a known label present in the question
        chosen_name = None
        try:
            for it in make_inquiry_targets():
                lbl_norm = _norm_name(it.get("label", ""))
                if lbl_norm and lbl_norm in qnorm:
                    chosen_name = it["label"]
                    break
        except Exception as e:
            app.logger.warning(f"scan targets failed: {e}")

        probe = chosen_name or user_q
        brief = build_inquiry_brief(probe)
        if not brief:
            return jsonify({
                "response": (
                    "I couldn’t locate that customer in the report/billing data. "
                    "Please include the company name as it appears in your system."
                )
            })

        # Always include up to 5 recent invoices in a separate CONTEXT block
        recent5 = (brief.get("recent_invoices") or [])[:5]
        recent_block = ""
        if recent5:
            lines = ["<CONTEXT:RECENT_INVOICES>"]
            for inv in recent5:
                desc = inv.get("Description") or ""
                lines.append(
                    f"- {inv['Date']} | {inv.get('Type','')} | ${inv.get('REVENUE',0):,.2f}"
                    + (f" | {desc}" if desc else "")
                )
            lines.append("</CONTEXT:RECENT_INVOICES>")
            recent_block = "\n".join(lines)

        # Clean, detailed, no-bold prompt with strict formatting rules
        system_prompt = {
            "role": "system",
            "content": (
                "You are a sales strategist for a forklift dealership.\n"
                "Use only the numeric and factual data provided in the INQUIRY context blocks; do not invent numbers.\n"
                f"Customer name is: {brief['inferred_name']}. Do not rename it or refer to any other customer.\n\n"

                "OUTPUT FORMAT (no asterisks/bold; clean spacing):\n"
                "- Each section header on its own line (no numbering).\n"
                "- Under each header, use bullets prefixed by exactly '  - ' (two spaces + hyphen).\n"
                "- Keep one blank line between sections. Avoid extra blank lines.\n\n"

                "SECTIONS TO PRODUCE (in this order):\n"
                "Segmentation\n"
                "  - Show Account Size (A/B/C/D) and Relationship (P/3/2/1) as provided.\n"
                "  - Include a one-line meaning for each (e.g., 'D — small account size', '3 — limited breadth of offerings').\n\n"

                "Current Pattern\n"
                "  - Top spending months (up to 3) with month-year and amount, from context.\n"
                "  - Top offerings (2–3) with amounts from context.\n"
                "  - Frequency as average days between invoices (or 'N/A' if unavailable).\n\n"

                "Visit Plan\n"
                "  - What to lead with (Service / Parts / Rental / New / Used) and why (tie to gaps or recent history).\n"
                "  - Keep it short, actionable, and grounded in the data.\n\n"

                "Next Level (from current → next better only)\n"
                "  - Move exactly one step (e.g., D3 → D2, C2 → C1). Do NOT skip tiers.\n"
                "  - State how many additional distinct offerings are needed to reach the next relationship tier.\n"
                "  - List the specific missing offerings that would count toward that.\n"
                "  - If applicable, note the revenue target to reach the next size letter.\n\n"

                "Next Actions\n"
                "  - Provide three short, specific tasks the rep should do today.\n\n"

                "Recent Invoices\n"
                "  - If a <CONTEXT:RECENT_INVOICES> block is present, show up to five bullets summarizing the most recent invoices.\n"
                "  - Use the same bullet style (two spaces + hyphen)."
            )
        }

        messages = [
            system_prompt,
            {"role": "system", "content": brief["context_block"]},
        ]
        if recent_block:
            messages.append({"role": "system", "content": recent_block})
        messages.append({"role": "user", "content": user_q})

        try:
            resp = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=900,
                temperature=0.4
            )
            ai_reply = resp.choices[0].message.content.strip()
        except Exception as e:
            ai_reply = f"❌ Internal error: {e}"

        tag = f"[Segmentation: {brief['size_letter']}{brief['relationship_code']}]"
        return jsonify({"response": f"{tag}\n\n{ai_reply}"})

    # ───────── Recommendation mode (existing flow) ─────────
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
            "List details using hyphens and indent sub-points.\n"
            "Only cite forklift Model codes exactly as in the data (e.g., CPD25, CQD16).\n"
            "End with Sales Pitch Techniques and Common Objections."
        )
    }

    messages = [system_prompt, {"role": "user", "content": prompt_ctx}]

    # Optional token guard — skip if tiktoken missing
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4")
        while sum(len(enc.encode(m["content"])) for m in messages) > 7000 and len(messages) > 2:
            messages.pop(1)
    except Exception:
        pass

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

# ─── Optional data endpoints ─────────────────────────────────────────────
@app.route("/api/modes")
def api_modes():
    return jsonify([
        {"id": "recommendation", "label": "Forklift Recommendation"},
        {"id": "inquiry",        "label": "Customer Inquiry"}
    ])

@app.route("/api/targets")
def api_targets():
    mode = (request.args.get("mode") or "recommendation").lower()
    if mode == "inquiry":
        return jsonify(make_inquiry_targets())
    items = []
    for i, a in enumerate(account_data):
        label = a.get("Account Name") or f"Account {i+1}"
        _id   = str(a.get("Account Name", label))
        items.append({"id": _id, "label": label})
    return jsonify(items)

# Debug endpoint to inspect server-side aggregates (optional)
@app.route("/api/inquiry_preview")
@login_required
def inquiry_preview():
    from data_sources import (
        build_inquiry_brief, find_inquiry_rows_flexible,
        _aggregate_billing, _aggregate_report_r12,
        get_report_headers, get_billing_headers, _pick_name
    )
    q = (request.args.get("q") or "").strip()
    if not q:
        return jsonify({"error": "pass ?q=Customer Name"}), 400

    rows = find_inquiry_rows_flexible(customer_name=q)
    rep_rows = rows["report"]
    bil_rows = rows["billing"]
    if not rep_rows and not bil_rows:
        return jsonify({"error": "not found"}), 404

    brief = build_inquiry_brief(q) or {}
    return jsonify({
        "name": brief.get("inferred_name"),
        "counts": {"report": len(rep_rows), "billing": len(bil_rows)},
        "report_headers": get_report_headers(),
        "billing_headers": get_billing_headers(),
        "matched_names": list({ _pick_name(r) for r in (rep_rows + bil_rows) })[:5],
        "billing_agg": _aggregate_billing(bil_rows),
        "report_agg": _aggregate_report_r12(rep_rows),
        "recent_invoices": brief.get("recent_invoices", []),
        "context_block": brief.get("context_block", "")
    })

# ─── Map endpoints ───────────────────────────────────────────────────────
@app.route("/map")
@login_required
def map_page():
    return render_template("map.html")

@app.route("/api/locations")
@login_required
def api_locations():
    # This simply returns the rich location dicts assembled in data_sources.get_locations_with_geo()
    from data_sources import get_locations_with_geo
    items = get_locations_with_geo()
    return jsonify(items)

# ─── Service worker at site root ─────────────────────────────────────────
@app.route('/service-worker.js')
def service_worker():
    return app.send_static_file('service-worker.js')

if __name__ == "__main__":
    app.run(debug=True, port=int(os.getenv("PORT", 5000)))
