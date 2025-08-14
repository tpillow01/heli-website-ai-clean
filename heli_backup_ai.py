# app.py  — no "Rules:" section in AI responses
import os, json, difflib, sqlite3, re
from datetime import timedelta
from functools import wraps

from flask import (
    Flask, render_template, request, jsonify, redirect, url_for, session, Response
)
from werkzeug.security import generate_password_hash, check_password_hash
from openai import OpenAI

# Grounding helpers from your ai_logic.py
from ai_logic import (
    generate_forklift_context,
    select_models_for_question,   # must exist in ai_logic.py
    allowed_models_block          # must exist in ai_logic.py
)

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

# ─── Pages ───────────────────────────────────────────────────────────────
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
        session["email"]   = user["email"]
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
        session["email"]   = user["email"]
        return redirect(url_for("home"))
    return render_template("signup.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ─── Prompt leak cleaner & model enforcement ─────────────────────────────
def _strip_prompt_leak(text: str) -> str:
    if not isinstance(text, str):
        return text
    # Remove echoed “Guidelines:”, “Rules:”, or “ALLOWED MODELS” blocks if the model leaks them
    text = re.sub(r'(?is)\nGuidelines:\n(?:.*\n?)*?(?=\n[A-Z][^\n]*:|\Z)', '\n', text).strip()
    text = re.sub(r'(?is)\nRules:\n(?:.*\n?)*?(?=\n[A-Z][^\n]*:|\Z)', '\n', text).strip()
    text = re.sub(r'(?is)\nALLOWED MODELS:\n(?:- .*\n?)*', '\n', text).strip()
    return text

def _enforce_allowed_models(text: str, allowed: set[str]) -> str:
    """
    Force the 'Model:' section to contain only the allowed model codes.
    If none are allowed, show a single 'No exact match...' line.
    """
    if not isinstance(text, str):
        return text

    if not allowed:
        forced = "Model:\n- No exact match from our lineup.\n"
        if "Model:" in text:
            text = re.sub(
                r'(?:^|\n)Model:\n(?:.*\n)*?(?=\n[A-Z][^\n]*:|\Z)',
                f"\n{forced}\n",
                text,
                flags=re.MULTILINE,
            )
        else:
            text = forced + "\n" + text
        return _strip_prompt_leak(text)

    forced = "Model:\n" + "\n".join(f"- {m}" for m in allowed) + "\n"
    if "Model:" in text:
        text = re.sub(
            r'(?:^|\n)Model:\n(?:.*\n)*?(?=\n[A-Z][^\n]*:|\Z)',
            f"\n{forced}\n",
            text,
            flags=re.MULTILINE,
        )
    else:
        text = forced + "\n" + text

    # Remove any model-like tokens not in the allowed set
    tokens = set(re.findall(r'\b[A-Z]{2,}[A-Z0-9\-]{1,}\b', text))
    for tok in tokens:
        if tok not in allowed and tok not in {"N/A"}:
            text = re.sub(rf'\b{re.escape(tok)}\b', '', text)
    text = re.sub(r'[ ]{2,}', ' ', text)
    return _strip_prompt_leak(text)

# ─── Chat API (Recommendation + Inquiry) ─────────────────────────────────
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
                f"Customer name is: {brief['inferred_name']}. Do not rename it or refer to any other customer.\n\n"
                "Write the answer with these exact section headers and spacing. Use short, one-sentence bullets with hyphens. No bold or markdown syntax.\n\n"
                "Segmentation: <LETTER><NUMBER>\n"
                "- Account Size: <LETTER>\n"
                "- Relationship: <NUMBER>\n"
                "- <LETTER> — meaning (e.g., D — small account size)\n"
                "- <NUMBER> — meaning (e.g., 3 — limited breadth of offerings)\n\n"
                "Current Pattern\n"
                "- Top spending months: Month YYYY ($#,###), Month YYYY ($#,###), Month YYYY ($#,###)\n"
                "- Top offerings: e.g., Parts ($#,###), Service ($#,###)\n"
                "- Frequency: Average of N days between invoices\n\n"
                "Visit Plan\n"
                "- Lead with: <one offering> (Service, Parts, Rental or New Equipment) — choose the category with the highest billing total in the context block, and briefly why that gap or trend suggests this focus.\n"
                "- Optional backup: <one secondary area> tied to the next-highest billing category.\n\n"
                "Next Level (from <LETTER><NUM> → next better only)\n"
                "- Relationship requirement: specify exactly how many new distinct offerings are needed to reach the next tier (do not skip tiers).\n"
                "- Best candidates to add: list 1–3 offerings based on gaps and what they already buy.\n"
                "- Size path (only if applicable): if moving up a size is the next improvement, state the R12 target (e.g., 'Grow to ≥ $10,000 to move D→C'). Never skip sizes.\n\n"
                "Next Actions\n"
                "- Three concrete, do-today tasks that align with the Visit Plan and Next Level steps.\n\n"
                "Recent Invoices\n"
                "- List up to 5 most recent invoices as: YYYY-MM-DD | Type | $Amount | Description (omit description if blank)\n"
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
        ai_reply = _strip_prompt_leak(ai_reply)
        return jsonify({"response": f"{tag}\n{ai_reply}"})

    # ───────── Recommendation mode with strict grounding ─────────
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

    # Select 3–5 best-fit models from models.json for THIS question
    hits, allowed = select_models_for_question(user_q, k=5)
    allowed_block = allowed_models_block(allowed)

    system_prompt = {
        "role": "system",
        "content": (
            "You are a friendly, expert Heli Forklift sales assistant.\n"
            "Output ONLY these sections in this order and nothing else:\n"
            "Customer Profile:\n"
            "Model:\n"
            "Power:\n"
            "Capacity:\n"
            "Tire Type:\n"
            "Attachments:\n"
            "Comparison:\n"
            "Sales Pitch Techniques:\n"
            "Common Objections:\n"
            "\n"
            "Guidance (do not echo this):\n"
            "- You may only reference model codes that appear under the ALLOWED MODELS block in the context. Do not invent other codes.\n"
            "- If there are no allowed models, say: \"No exact match from our lineup.\" and discuss categories without naming model codes.\n"
            "- Do NOT repeat these instructions, the guidance, or the ALLOWED MODELS block in your answer.\n"
        )
    }

    messages = [
        system_prompt,
        {"role": "system", "content": allowed_block},  # strict grounding list
        {"role": "user",   "content": prompt_ctx}
    ]

    try:
        resp = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=600,
            temperature=0.6
        )
        ai_reply = resp.choices[0].message.content.strip()
    except Exception as e:
        ai_reply = f"❌ Internal error: {e}"

    # Enforce allowed models & remove any leaked instruction sections
    ai_reply = _enforce_allowed_models(ai_reply, set(allowed))
    return jsonify({"response": ai_reply})

# ─── Modes list ──────────────────────────────────────────────────────────
@app.route("/api/modes")
def api_modes():
    return jsonify([
        {"id": "recommendation", "label": "Forklift Recommendation"},
        {"id": "inquiry",        "label": "Customer Inquiry"}
    ])

# ─── Map routes ───────────────────────────────────────────────────────────
@app.route("/map")
@login_required
def map_page():
    return render_template("map.html")

@app.route("/api/locations")
@login_required
def api_locations():
    """
    Build map points from customer_location.csv (your exact headers) and
    attach Sales Rep from customer_report.csv for territory coloring.
    """
    import csv, json as _json, re as _re
    import pandas as _pd
    from difflib import get_close_matches
    from flask import Response as _Response

    # helpers
    def strip_suffixes(s: str) -> str:
        return _re.sub(r"\b(inc|inc\.|llc|l\.l\.c\.|co|co\.|corp|corporation|company|ltd|ltd\.|lp|plc)\b",
                      "", s or "", flags=_re.IGNORECASE)
    def norm_name(s: str) -> str:
        s = strip_suffixes(s or "")
        s = s.lower()
        s = _re.sub(r"[^a-z0-9]+", " ", s)
        return _re.sub(r"\s+", " ", s).strip()
    def norm_city(s: str) -> str:
        s = (s or "").lower()
        s = _re.sub(r"[^a-z0-9]+", " ", s)
        return _re.sub(r"\s+", " ", s).strip()
    def state_from_county_state(v: str) -> str:
        parts = _re.sub(r"\s+", " ", (v or "").strip()).split(" ")
        return parts[-1].upper() if len(parts) >= 2 else ""
    def county_from_county_state(v: str) -> str:
        parts = _re.sub(r"\s+", " ", (v or "").strip()).split(" ")
        return " ".join(parts[:-1]) if len(parts) >= 2 else ""
    def zip5(z: str) -> str:
        m = _re.search(r"\d{5}", str(z or ""))
        return m.group(0) if m else ""
    def to_float(x):
        try:
            v = float(str(x).strip())
            if v != v:  # NaN
                return None
            return v
        except Exception:
            return None

    # rep index
    rep_idx_exact = {}
    rep_idx_norm = {}
    rep_idx_norm_zip = {}
    rep_idx_norm_city_state = {}

    try:
        rep_df = _pd.read_csv("customer_report.csv", dtype=str).fillna("")
        REP_COL = "Sales Rep Name" if "Sales Rep Name" in rep_df.columns else None
        if REP_COL:
            SOLD_COL = "Sold to Name"
            SHIP_COL = "Ship to Name"
            CITY_COL = "City"
            ZIP_COL  = "Zip Code"
            CST_COL  = "County State"

            rep_df["_sold_norm"] = rep_df.get(SOLD_COL, "").apply(norm_name)
            rep_df["_ship_norm"] = rep_df.get(SHIP_COL, "").apply(norm_name)
            rep_df["_zip5"]      = rep_df.get(ZIP_COL, "").apply(zip5)
            rep_df["_city_norm"] = rep_df.get(CITY_COL, "").str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True)\
                                                     .str.replace(r"\s+", " ", regex=True).str.strip()
            rep_df["_state"]     = rep_df.get(CST_COL, "").apply(state_from_county_state)

            for _, r in rep_df.iterrows():
                rep = (r.get(REP_COL, "") or "").strip()
                if not rep:
                    continue
                for col in (SOLD_COL, SHIP_COL):
                    nm = (r.get(col, "") or "").strip()
                    if nm:
                        rep_idx_exact.setdefault(nm, rep)
                for nval in (r.get("_sold_norm", ""), r.get("_ship_norm", "")):
                    nval = (nval or "").strip()
                    if not nval:
                        continue
                    rep_idx_norm.setdefault(nval, rep)
                    z = r.get("_zip5", "")
                    if z:
                        rep_idx_norm_zip.setdefault(f"{nval}|{z}", rep)
                    cn = r.get("_city_norm", "")
                    st = r.get("_state", "")
                    if cn and st:
                        rep_idx_norm_city_state.setdefault(f"{nval}|{cn}|{st}", rep)

            rep_norm_keys = list(rep_idx_norm.keys())
        else:
            rep_norm_keys = []
    except Exception as e:
        print("⚠️ customer_report.csv not available for rep coloring:", e)
        rep_norm_keys = []

    def lookup_rep(name: str, city: str, state: str, zipc: str) -> str:
        if not name:
            return None
        if name in rep_idx_exact:
            return rep_idx_exact[name]
        n = norm_name(name)
        if not n:
            return None
        z5 = zip5(zipc)
        if z5 and f"{n}|{z5}" in rep_idx_norm_zip:
            return rep_idx_norm_zip[f"{n}|{z5}"]
        cn = norm_city(city)
        st = (state or "").upper()
        key_cs = f"{n}|{cn}|{st}" if cn and st else None
        if key_cs and key_cs in rep_idx_norm_city_state:
            return rep_idx_norm_city_state[key_cs]
        if n in rep_idx_norm:
            return rep_idx_norm[n]
        if rep_norm_keys:
            guess = difflib.get_close_matches(n, rep_norm_keys, n=1, cutoff=0.88)
            if guess:
                return rep_idx_norm.get(guess[0])
        return None

    # build points
    items = []
    try:
        with open("customer_location.csv", "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name    = (row.get("Account Name", "") or "").strip()
                city    = (row.get("City", "") or "").strip()
                address = (row.get("Address", "") or "").strip()
                cs_raw  = (row.get("County State", "") or "").strip()
                county  = county_from_county_state(cs_raw)
                state   = state_from_county_state(cs_raw)
                zipc    = zip5(row.get("Zip Code", ""))

                lat = to_float(row.get("Min of Latitude", ""))
                lon = to_float(row.get("Min of Longitude", ""))

                if lat is None or lon is None:
                    continue
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    continue

                rep = lookup_rep(name, city, state, zipc) or "Unassigned"

                items.append({
                    "label": name or "Unknown",
                    "address": address,
                    "city": city,
                    "state": state,
                    "county": county,
                    "zip": zipc,
                    "sales_rep": rep,
                    "lat": lat,
                    "lon": lon,
                    "County State": cs_raw
                })
    except FileNotFoundError:
        return _Response(_json.dumps({"error": "customer_location.csv not found"}), status=500, mimetype="application/json")
    except Exception as e:
        print("❌ /api/locations error:", e)
        return _Response(_json.dumps({"error": "Failed to read customer_location.csv"}), status=500, mimetype="application/json")

    return _Response(_json.dumps(items, allow_nan=False), mimetype="application/json")

@app.route('/service-worker.js')
def service_worker():
    return app.send_static_file('service-worker.js')

# ─── AI Map Analysis Endpoint ────────────────────────────────────────────
@app.route('/api/ai_map_analysis', methods=['POST'])
def ai_map_analysis():
    import pandas as pd, re
    from difflib import get_close_matches

    try:
        payload = request.get_json(force=True) or {}
        customer_raw = (payload.get('customer') or '').strip()
        zip_hint     = (payload.get('zip') or '').strip()
        city_hint    = (payload.get('city') or '').strip()
        state_hint   = (payload.get('state') or '').strip().upper()

        if not customer_raw:
            return jsonify({"error": "Customer name required"}), 400

        try:
            df = pd.read_csv("customer_report.csv", dtype=str).fillna("")
        except Exception as e:
            print("❌ ai_map_analysis read error:", e)
            return jsonify({"error": "Could not read customer_report.csv"}), 500

        SOLD, SHIP = "Sold to Name", "Ship to Name"
        SOLD_ID    = "Sold to ID"
        CITY, ZIP  = "City", "Zip Code"
        CST, SEG   = "County State", "R12 Segment (Sold to ID)"

        REV_COLS = [
            "New Equip R36 Revenue",
            "Used Equip R36 Revenue",
            "Parts Revenue R12",
            "Service Revenue R12 (Includes GM)",
            "Parts & Service Revenue R12",
            "Rental Revenue R12",
            "Revenue Rolling 12 Months - Aftermarket",
            "Revenue Rolling 13 - 24 Months - Aftermarket",
        ]

        def strip_suffixes(s: str) -> str:
            return re.sub(r"\b(inc|inc\.|llc|l\.l\.c\.|co|co\.|corp|corporation|company|ltd|ltd\.|lp|plc)\b", "", s or "", flags=re.IGNORECASE)
        def norm_name(s: str) -> str:
            s = strip_suffixes(s); s = s.lower()
            s = re.sub(r"[^a-z0-9]+", " ", s)
            return re.sub(r"\s+", " ", s).strip()
        def state_from_cst(v: str) -> str:
            parts = re.sub(r"\s+", " ", (v or "").strip()).split(" ")
            return parts[-1].upper() if len(parts) >= 2 else ""
        def zip5(z: str) -> str:
            m = re.search(r"\d{5}", (z or ""))
            return m.group(0) if m else ""
        def money_to_float(v) -> float:
            s = str(v or "").strip().replace("$","").replace(",","")
            if not s: return 0.0
            try: return float(s)
            except Exception:
                m = re.search(r"-?\d+(\.\d+)?", s)
                return float(m.group(0)) if m else 0.0

        df["_sold_norm"] = df[SOLD].apply(norm_name)
        df["_ship_norm"] = df[SHIP].apply(norm_name)
        df["_zip5"]      = df[ZIP].apply(zip5)
        df["_city_norm"] = df[CITY].str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True)\
                                    .str.replace(r"\s+", " ", regex=True).str.strip()
        df["_state"]     = df[CST].apply(state_from_cst)

        cust_norm  = norm_name(customer_raw)
        zip_norm   = zip5(zip_hint)
        city_norm  = re.sub(r"[^a-z0-9]+", " ", city_hint.lower()).strip() if city_hint else ""
        state_abbr = state_hint

        m_exact = (df[SOLD].str.lower() == customer_raw.lower()) | (df[SHIP].str.lower() == customer_raw.lower())
        m_norm  = (df["_sold_norm"] == cust_norm) | (df["_ship_norm"] == cust_norm)

        def refine(mask):
            out = mask.copy()
            if zip_norm:   out = out & (df["_zip5"] == zip_norm)
            if city_norm:  out = out & (df["_city_norm"] == city_norm)
            if state_abbr: out = out & (df["_state"] == state_abbr)
            return out

        masks = [refine(m_exact), refine(m_norm), m_exact, m_norm]
        hit = None
        for mk in masks:
            cand = df[mk]
            if not cand.empty:
                hit = cand
                break

        if hit is None:
            all_norms = list(set(df["_sold_norm"].tolist() + df["_ship_norm"].tolist()))
            guess = get_close_matches(cust_norm, all_norms, n=1, cutoff=0.92)
            if guess:
                g = guess[0]
                hit = df[(df["_sold_norm"] == g) | (df["_ship_norm"] == g)]

        if hit is None or hit.empty:
            print("❌ No match found for:", customer_raw)
            return jsonify({"error": f"No data found for {customer_raw}"}), 200

        has_any_sold_id = (hit[SOLD_ID].astype(str).str.strip() != "").any()
        if has_any_sold_id:
            chosen_group_id = hit[SOLD_ID].astype(str).str.strip().iloc[0]
            group_df = df[df[SOLD_ID].astype(str).str.strip() == chosen_group_id]
        else:
            base_norm = norm_name(hit[SOLD].iloc[0])
            group_df = df[df[SOLD].apply(norm_name) == base_norm]

        totals = {}
        for col in REV_COLS:
            totals[col] = group_df[col].map(money_to_float).sum() if col in group_df.columns else 0.0

        seg_val = ""
        if SEG in group_df.columns:
            mode_series = group_df[SEG].astype(str).str.strip().replace("", pd.NA).dropna().mode()
            if not mode_series.empty:
                seg_val = str(mode_series.iat[0])

        display_name = customer_raw or (hit[SOLD].iloc[0] if SOLD in hit.columns else "")

        lines = [f"Customer: {display_name}"]
        if seg_val:
            lines.append(f"Segment: {seg_val}")
        lines.append("Key financial metrics:")
        for k in REV_COLS:
            lines.append(f"- {k}: ${totals[k]:,.2f}")
        metrics_block = "\n".join(lines)

        prompt = f"""{metrics_block}

Write a concise analysis that USES the dollar figures above in your sentences.
Requirements:
- Reference at least the three largest figures by name and amount.
- 2–4 bullet points on what's driving results (with numbers inline).
- 1–3 bullets for next actions (upsell forklifts, service, rentals, parts), referencing numbers where relevant.
Keep it crisp and sales-focused.
"""

        try:
            oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            resp = oai.chat.completions.create(
                model="gpt-4",
                temperature=0.3,
                messages=[
                    {"role": "system", "content": "You are a forklift sales strategist. Be concise and analytical."},
                    {"role": "user", "content": prompt}
                ]
            )
            analysis = resp.choices[0].message.content.strip()
        except Exception as e:
            print("❌ OpenAI error:", e)
            analysis = None

        aggregated_flag = True if has_any_sold_id else False

        return jsonify({
            "result": analysis or metrics_block,
            "metrics": totals,
            "segment": seg_val,
            "matched_rows": int(len(hit)),
            "aggregated": aggregated_flag,
        })

    except Exception as e:
        print("❌ Error during AI map analysis:", e)
        return jsonify({"error": str(e)}), 500

# --- Segment lookup for map popups ---------------------------------------
@app.route("/api/segments")
@login_required
def api_segments():
    import pandas as pd, re
    from flask import jsonify

    SEG_COL   = "R12 Segment (Sold to ID)"
    SOLD_COL  = "Sold to Name"
    SHIP_COL  = "Ship to Name"
    CITY_COL  = "City"
    ZIP_COL   = "Zip Code"
    CST_COL   = "County State"

    try:
        df = pd.read_csv("customer_report.csv", dtype=str).fillna("")
    except Exception as e:
        print("❌ /api/segments read error:", e)
        return jsonify({"by_exact": {}, "by_norm": {}, "by_norm_zip": {}, "by_norm_city_state": {}}), 200

    def strip_suffixes(s: str) -> str:
        return re.sub(r"\b(inc|inc\.|llc|l\.l\.c\.|co|co\.|corp|corporation|company|ltd|ltd\.|lp|plc)\b", "", s, flags=re.IGNORECASE)
    def norm_name(s: str) -> str:
        s = strip_suffixes(s or ""); s = s.lower()
        s = re.sub(r"[^a-z0-9]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s
    def norm_city(s: str) -> str:
        s = (s or "").lower()
        s = re.sub(r"[^a-z0-9]+", " ", s)
        return re.sub(r"\s+", " ", s).strip()
    def state_from_county_state(v: str) -> str:
        parts = re.sub(r"\s+", " ", (v or "").strip()).split(" ")
        return parts[-1] if len(parts) >= 2 else ""
    def zip5(z: str) -> str:
        m = re.search(r"\d{5}", (z or ""))
        return m.group(0) if m else ""

    by_exact = {}
    by_norm = {}
    by_norm_zip = {}
    by_norm_city_state = {}

    for _, row in df.iterrows():
        seg = (row.get(SEG_COL, "") or "").strip()
        if not seg:
            continue

        city  = (row.get(CITY_COL, "") or "").strip()
        zipc  = zip5(row.get(ZIP_COL, ""))
        state = (state_from_county_state(row.get(CST_COL, "")) or "").strip().upper()

        for col in (SOLD_COL, SHIP_COL):
            name = (row.get(col, "") or "").strip()
            if not name:
                continue

            by_exact.setdefault(name, seg)

            n = norm_name(name)
            if n:
                by_norm.setdefault(n, seg)

                if zipc:
                    by_norm_zip.setdefault(f"{n}|{zipc}", seg)

                if city and state:
                    by_norm_city_state.setdefault(f"{n}|{norm_city(city)}|{state}", seg)

    return jsonify({
        "by_exact": by_exact,
        "by_norm": by_norm,
        "by_norm_zip": by_norm_zip,
        "by_norm_city_state": by_norm_city_state
    })

# ─── Entrypoint ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=int(os.getenv("PORT", 5000)))
