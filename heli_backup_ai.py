import os
import json
import pandas as pd
import difflib
import sqlite3
from datetime import timedelta

from flask import Flask, render_template, request, jsonify, redirect, url_for, session, Response
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

        recent_block = ""
        if brief.get("recent_invoices"):
            five = brief["recent_invoices"][:5]
            if five:
                lines = ["Recent Invoices"]
                for inv in five:
                    desc = inv.get("Description", "")
                    line = f"- {inv['Date']} | {inv['Type']} | ${inv['REVENUE']:,.2f}"
                    if desc:
                        line += f" | {desc}"
                    lines.append(line)
                recent_block = "\n".join(lines)

        system_prompt = {
             "role": "system",
             "content": (
                 "You are a sales strategist for a forklift dealership. Use the INQUIRY context verbatim; do not invent numbers.\n"
                 f"Customer name is: {brief['inferred_name']}. Do not rename it or refer to any other customer.\n"
                 "\n"
                 "Write the answer with these exact section headers and spacing. Use short, one-sentence bullets with hyphens. No bold or markdown syntax.\n"
                 "\n"
                 "Segmentation: <LETTER><NUMBER>\n"
                 "- Account Size: <LETTER>\n"
                 "- Relationship: <NUMBER>\n"
                 "- <LETTER> — meaning (e.g., D — small account size)\n"
                 "- <NUMBER> — meaning (e.g., 3 — limited breadth of offerings)\n"
                 "\n"
                 "Current Pattern\n"
                 "- Top spending months: Month YYYY ($#,###), Month YYYY ($#,###), Month YYYY ($#,###)\n"
                 "- Top offerings: e.g., Parts ($#,###), Service ($#,###)\n"
                 "- Frequency: Average of N days between invoices\n"
                 "\n"
                 "Visit Plan\n"
                 "- Lead with: <one offering> (Service, Parts, Rental or New Equipment) — choose the category with the highest billing total in the context block, and briefly why that gap or trend suggests this focus.\n"
                 "- Optional backup: <one secondary area> tied to the next-highest billing category.\n"
                 "\n"
                 "Next Level (from <LETTER><NUM> → next better only)\n"
                 "- Relationship requirement: specify exactly how many new distinct offerings are needed to reach the next tier (do not skip tiers).\n"
                 "- Best candidates to add: list 1–3 offerings based on gaps and what they already buy.\n"
                 "- Size path (only if applicable): if moving up a size is the next improvement, state the R12 target (e.g., 'Grow to ≥ $10,000 to move D→C'). Never skip sizes.\n"
                 "\n"
                 "Next Actions\n"
                 "- Three concrete, do-today tasks that align with the Visit Plan and Next Level steps.\n"
                 "\n"
                 "Recent Invoices\n"
                 "- List up to 5 most recent invoices as: YYYY-MM-DD | Type | $Amount | Description (omit description if blank)\n"
                 "\n"
                 "Rules:\n"
                 "- Use only numbers and facts present in the context blocks. No estimates beyond what is provided.\n"
                 "- Keep each bullet on one line. No bold, no asterisks, no extra prose.\n"
                 "- Never propose jumping from D3 to C3 (or similar); only move to the next better tier.\n"
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
                temperature=0.35
            )
            ai_reply = resp.choices[0].message.content.strip()
        except Exception as e:
            ai_reply = f"❌ Internal error: {e}"

        tag = f"Segmentation: {brief['size_letter']}{brief['relationship_code']}"
        return jsonify({"response": f"{tag}\n{ai_reply}"})

    # ───────── Recommendation mode (existing flow) ─────────
    acct = find_account_by_name(user_q)
    context_input = user_q
    if acct:
        profile_ctx = (
            "Customer Profile:\n"
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
            "You are a friendly, expert Heli Forklift sales assistant.\n"
            "When a user asks for a recommendation, always respond in the following structured sections, in order:\n\n"
            "Customer Profile:\n"
            "- Company: <Account Name>\n"
            "- Industry: <Industry>\n"
            "- SIC Code: <SIC Code>\n"
            "- Fleet Size: <Total Company Fleet Size>\n"
            "- Truck Types: <Truck Types at Location>\n\n"
            "Model:\n"
            "- List one or more forklift model codes (e.g., CPD25, CQD16) that best fit the customer's needs.\n\n"
            "Power:\n"
            "- Specify power options (electric, LPG, diesel) with brief pros/cons tailored to their environment.\n\n"
            "Capacity:\n"
            "- Recommend capacity range with rationale (e.g., \"2.5–3.5 ton for medium-duty warehouse use\").\n\n"
            "Tire Type:\n"
            "- Suggest cushion or pneumatic tires, based on indoor/outdoor use and floor conditions.\n\n"
            "Attachments:\n"
            "- Propose any useful attachments (e.g., side shifters, fork positioners) tied to their operation.\n\n"
            "Comparison:\n"
            "- If offering multiple models, include a brief bullet-list comparison of key specs.\n\n"
            "Sales Pitch Techniques:\n"
            "- Two concise, persuasive talking points (e.g., cost-savings, uptime benefits).\n\n"
            "Common Objections:\n"
            "- Two likely customer concerns plus how to overcome each.\n\n"
            "Guidelines:\n"
            "- Cite model codes exactly as they appear in models.json.\n"
            "- Use hyphens for bullets; indent sub-points by two spaces.\n"
            "- Keep each bullet to one clear sentence.\n"
            "- Write in a confident, consultative tone; no unnecessary jargon.\n\n"
            "Now, based on the user’s question and their Customer Profile, fill in each section with tailored, data-driven recommendations."
        )
    }

    messages = [system_prompt, {"role": "user", "content": prompt_ctx}]

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

# Map routes
@app.route("/map")
@login_required
def map_page():
    return render_template("map.html")

@app.route("/api/locations")
@login_required
def api_locations():
    from data_sources import get_locations_with_geo
    items = get_locations_with_geo()
    return Response(json.dumps(items, allow_nan=False), mimetype="application/json")

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

@app.route('/service-worker.js')
def service_worker():
    return app.send_static_file('service-worker.js')

# ─── AI Map Analysis Endpoint ─────────────────────────────────────────────
@app.route('/api/ai_map_analysis', methods=['POST'])
def ai_map_analysis():
    """
    Look up the exact row for a customer in customer_report.csv.
    If multiple rows share the same Sold to Name, further narrow by city/zip/address.
    Falls back to the highest 'Revenue Rolling 12 Months - Aftermarket' if still ambiguous.
    """
    try:
        payload = request.get_json(force=True) or {}
        customer = (payload.get('customer') or '').strip()
        city     = (payload.get('city') or '').strip()
        zip_code = (payload.get('zip') or '').strip()
        address  = (payload.get('address') or '').strip()

        if not customer:
            return jsonify({"error": "Customer name required"}), 400

        import pandas as pd
        # Read as strings to avoid dtype surprises; we’ll coerce as needed
        df = pd.read_csv("customer_report.csv", dtype=str, keep_default_na=False)

        # Normalize helper
        def norm(s):
            return str(s).strip().casefold()

        # Filter by Sold to Name first
        mask = df.get('Sold to Name', '').astype(str).map(norm) == norm(customer)
        subset = df[mask].copy()

        # Narrow by city if given
        if city and not subset.empty:
            m2 = subset.get('City', '').astype(str).map(norm) == norm(city)
            sub2 = subset[m2]
            if not sub2.empty:
                subset = sub2

        # Narrow by ZIP if given
        if zip_code and not subset.empty:
            z = str(zip_code).strip()
            m2 = subset.get('Zip Code', '').astype(str).str.strip() == z
            sub2 = subset[m2]
            if not sub2.empty:
                subset = sub2

        # Narrow by address (contains) if given
        if address and not subset.empty:
            adr = norm(address)
            m2 = subset.get('Address', '').astype(str).map(norm).str.contains(adr, na=False)
            sub2 = subset[m2]
            if not sub2.empty:
                subset = sub2

        if subset.empty:
            return jsonify({"error": f"No data found for {customer}"}), 404

        # If still multiple, pick the row with the largest R12 Aftermarket
        def money_to_float(x):
            s = str(x).replace("$", "").replace(",", "").strip()
            try:
                return float(s)
            except:
                return 0.0

        if len(subset) > 1:
            subset = subset.copy()
            subset["__r12__"] = subset.get(
                "Revenue Rolling 12 Months - Aftermarket", ""
            ).map(money_to_float)
            subset = subset.loc[[subset["__r12__"].idxmax()]]

        r = subset.iloc[0]

        fields = {
            "New Equip R36 Revenue": r.get("New Equip R36 Revenue", 0),
            "Used Equip R36 Revenue": r.get("Used Equip R36 Revenue", 0),
            "Parts Revenue R12": r.get("Parts Revenue R12", 0),
            "Service Revenue R12 (Includes GM)": r.get("Service Revenue R12 (Includes GM)", 0),
            "Parts & Service Revenue R12": r.get("Parts & Service Revenue R12", 0),
            "Rental Revenue R12": r.get("Rental Revenue R12", 0),
            "Revenue Rolling 12 Months - Aftermarket": r.get("Revenue Rolling 12 Months - Aftermarket", 0),
            "Revenue Rolling 13 - 24 Months - Aftermarket": r.get("Revenue Rolling 13 - 24 Months - Aftermarket", 0),
        }

        # Build prompt with normalized currency
        def fmt_money(x):
            v = money_to_float(x)
            return f"${v:,.2f}"

        metrics_block = "\n".join([f"{k}: {fmt_money(v)}" for k, v in fields.items()])

        prompt = f"""Customer: {customer}
Here are the latest financial metrics for this customer:

{metrics_block}

Based on these revenue metrics, give a short and clear insight:
- What kind of customer is this?
- What stands out?
- Where is there potential to upsell forklifts, service, rentals, or parts?
Keep it brief and analytic.
"""

        print("🔍 Selected row for:", customer, "| City:", city, "| ZIP:", zip_code, "| Address:", address)
        print("📤 Sending to OpenAI:\n", prompt)

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You're a forklift sales strategist."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        analysis = response.choices[0].message.content.strip()
        print("✅ Response:\n", analysis)

        return jsonify({"response": analysis})

    except Exception as e:
        print("❌ Error during AI map analysis:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=int(os.getenv("PORT", 5000)))