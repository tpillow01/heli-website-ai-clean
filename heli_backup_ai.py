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

        # If the rep asked for recent invoices, attach a compact listing
        want_recent = any(k in qnorm for k in (
            "recent invoice", "recent invoices", "last invoices", "latest invoices", "most recent invoices"
        ))
        recent_block = ""
        if want_recent and brief.get("recent_invoices"):
            lines = ["<CONTEXT:RECENT_INVOICES>"]
            for inv in brief["recent_invoices"]:
                lines.append(f"- {inv['Date']} | {inv['Type']} | ${inv['REVENUE']:,.2f} | {inv.get('Description','')}")
            lines.append("</CONTEXT:RECENT_INVOICES>")
            recent_block = "\n".join(lines)

            system_prompt = {
                "role": "system",
                "content": (
                    "You are a sales strategist for a forklift dealership.\n"
                    "Use ONLY the provided INQUIRY context; do not invent numbers or customers.\n"
                    "Customer name is fixed. Do not rename it.\n\n"

                    "FORMAT RULES (strict):\n"
                    "- Use exactly the sections below in this order.\n"
                    "- Put ONE blank line between sections (no more, no less).\n"
                    "- Use hyphen bullets starting with '- ' for list items.\n"
                    "- Use two spaces for sub-bullets (e.g., '  - ').\n"
                    "- Keep lines short and scannable.\n\n"

                    "RESPONSE TEMPLATE:\n"
                    "1) Segmentation\n"
                    "   - Account Size: <A/B/C/D> — short meaning.\n"
                    "   - Relationship: <P/3/2/1> — short meaning.\n\n"
                    "2) Current Pattern\n"
                    "   - Top Spending Months: <Month YYYY ($#), ...>.\n"
                    "   - Top Offerings: <Parts/Service/Rental/...>.\n"
                    "   - Frequency: <~N days between invoices>.\n\n"
                    "3) Visit Plan\n"
                    "   - Lead with: <Service/Parts/Rental/New/Used>.\n"
                    "   - Why: <1 short sentence using billing data>.\n"
                    "   - Backup: <optional 1 item>.\n\n"
                    "4) Next Level (from current → next better only)\n"
                    "   - Add Offerings: <which are missing to reach next tier>.\n"
                    "   - Revenue Path: <if relevant, R12 target>.\n"
                    "   - Quick Wins: <1–2 concrete ideas>.\n\n"
                    "5) Next Actions\n"
                    "   - <Action 1>\n"
                    "   - <Action 2>\n"
                    "   - <Action 3>\n\n"
                    "If a <CONTEXT:RECENT_INVOICES> block is present, add this section at the end:\n"
                    "Recent Invoices\n"
                    "  - <YYYY-MM-DD | Dept | $ | (desc)> up to 5 items.\n"
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
            "Only cite forklift **Model** codes exactly as in the data (e.g., CPD25, CQD16).\n"
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

# Debug endpoint to inspect server-side aggregates
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

# ─── Map routes (added) ─────────────────────────────────────────────────
@app.route("/map")
@login_required
def map_page():
    return render_template("map.html")

@app.route("/api/locations")
@login_required
def api_locations():
    """
    Returns deduped geo points from customer_location.csv, enriched with
    territory (Sales Rep Name) and segment when available.
    Optional filters: ?rep=...&segment=A&state=IN&county=Marion&zip=46214
    """
    from data_sources import get_locations_with_geo

    items = get_locations_with_geo()

    # simple filtering on server side (optional)
    rep     = (request.args.get("rep") or "").strip().lower()
    segment = (request.args.get("segment") or "").strip().upper()  # "A","B","C","D" or "A1" etc
    state   = (request.args.get("state") or "").strip().upper()
    county  = (request.args.get("county") or "").strip().lower()
    zc      = (request.args.get("zip") or "").strip()

    def keep(it):
        if rep     and rep     not in (it.get("sales_rep","").lower()): return False
        if state   and state   != (it.get("state","").upper()):         return False
        if county  and county  not in (it.get("county","").lower()):    return False
        if zc      and zc      != (it.get("zip","")):                   return False
        if segment:
            seg = (it.get("segment") or "").upper()
            if len(segment) == 1:  # match A/B/C/D
                if seg[:1] != segment[:1]:
                    return False
            else:  # match full like "C2"
                if seg != segment:
                    return False
        return True

    items = [it for it in items if keep(it)]
    return jsonify(items)

@app.route('/service-worker.js')
def service_worker():
    return app.send_static_file('service-worker.js')

if __name__ == "__main__":
    app.run(debug=True, port=int(os.getenv("PORT", 5000)))
