# app.py  — no "Rules:" section in AI responses
import os, json, difflib, sqlite3, re, time
from datetime import timedelta
from functools import wraps

from flask import (
    Flask, render_template, request, jsonify, redirect, url_for, session, Response
)

# add near the top with other imports
from csv_locations import load_csv_locations, to_geojson

from werkzeug.security import generate_password_hash, check_password_hash
from openai import OpenAI

# Grounding helpers from your ai_logic.py
from ai_logic import (
    generate_forklift_context,
    select_models_for_question,
    allowed_models_block,
    debug_parse_and_rank,   # <<< keep this
    top_pick_meta           # <<< NEW: promotions helper
)

# Admin usage tracking
from admin_usage import admin_bp, init_admin_usage, record_event, log_model_usage

# Promotions
from promotions import promos_for_context, render_promo_lines

# -------------------------------------------------------------------------
# Data boot
# (If you load accounts/models earlier, keep that; then load locations.)
locations_index = load_csv_locations()
print(f"✅ Loaded {len(locations_index)} locations from customer_location.csv")

# -------------------------------------------------------------------------
# Flask & OpenAI client
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-insecure")
app.permanent_session_lifetime = timedelta(days=7)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=bool(os.getenv("SESSION_COOKIE_SECURE", "1") == "1"),
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------------------------------------------------
# Admin usage tracking bootstrap (creates DB + coarse per-request logging)
init_admin_usage(app)
app.register_blueprint(admin_bp)  # /admin/login, /admin, /admin/usage.json

# (Your routes continue below…)


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
    Enforce a single, grounded 'Model:' section placed right AFTER 'Customer Profile:'.
    Also normalizes spacing and bullets later via _tidy_formatting().
    """
    if not isinstance(text, str):
        return text

    # Build grounded Model block
    al = [m for m in allowed if isinstance(m, str) and m.strip()]
    if not al:
        forced_block = "Model:\n- No exact match from our lineup.\n"
    else:
        top = al[0]
        alts = [x for x in al[1:5]]
        forced_block = "Model:\n- Top Pick: " + top + "\n"
        if alts:
            forced_block += "- Alternates: " + ", ".join(alts) + "\n"

    # 1) Remove EVERY existing Model: section (keep none; we re-insert once)
    model_sec = r'(?:^|\n)Model:\n(?:.*?\n)*?(?=\n[A-Z][^\n]*:|\Z)'
    text = re.sub(model_sec, "\n", text, flags=re.MULTILINE)

    # 2) Insert our Model block right AFTER Customer Profile: section if present
    cust_pat = r'(?s)(^|\n)Customer Profile:\n(?:.*?\n)*?(?=\n[A-Z][^\n]*:|\Z)'
    m = re.search(cust_pat, text)
    if m:
        end = m.end()
        text = text[:end] + ("\n" if not text[end-1] == "\n" else "") + forced_block + text[end:]
    else:
        # If somehow Customer Profile isn't present, prepend
        text = forced_block + ("\n" if text and not text.startswith("\n") else "") + text

    # 3) Final tidy pass (bullets, spacing, leaked blocks)
    text = _tidy_formatting(text)
    return _strip_prompt_leak(text)

# --- Helpers to clean the LLM output ------------------------------------

def _unify_model_mentions(text: str, allowed: list[str]) -> str:
    """
    Make sure every model name outside the Model: section refers to the Top Pick.
    We only rewrite codes that appear in the allowed list (to avoid overreach).
    """
    if not isinstance(text, str) or not text.strip() or not allowed:
        return text

    # Grab the Model: section
    sec_pat = r'(?s)(?:^|\n)Model:\n(?:.*?\n)*?(?=\n[A-Z][^\n]*:|\Z)'
    msec = re.search(sec_pat, text)
    if not msec:
        return text

    model_sec = msec.group(0)
    body = text[:msec.start()] + '<<MODEL_SECTION>>' + text[msec.end():]

    # Find Top Pick inside the section
    mtop = re.search(r'-\s*Top Pick:\s*([A-Za-z0-9().\- ]+)', model_sec)
    if not mtop:
        return text  # nothing to unify if we don't have a top pick

    top = mtop.group(1).strip()

    # Replace other allowed codes with Top Pick everywhere OUTSIDE the Model: section
    for code in allowed:
        code = code.strip()
        if not code or code == top:
            continue
        body = re.sub(rf'\b{re.escape(code)}\b', top, body)

    return body.replace('<<MODEL_SECTION>>', model_sec)

def _fix_labels_and_breaks(text: str) -> str:
    """
    Clean odd line wraps like 'Minimum\\nCapacity:' and 'Suggested\\nAttachments:'.
    Also normalize those labels to 'Capacity:', 'Attachments:', 'Tire:'.
    """
    if not isinstance(text, str) or not text.strip():
        return text

    # merge split labels across a newline
    text = re.sub(r'(?mi)^-\s*Minimum\s*\n\s*Capacity:', '- Capacity:', text)
    text = re.sub(r'(?mi)^-\s*Suggested\s*\n\s*Attachments:', '- Attachments:', text)
    text = re.sub(r'(?mi)^-\s*Suggested\s*\n\s*Tire:', '- Tire:', text)

    # direct label normalization (in case they weren’t split)
    text = re.sub(r'(?mi)^-\s*Minimum\s+Capacity:', '- Capacity:', text)
    text = re.sub(r'(?mi)^-\s*Suggested\s+Attachments:', '- Attachments:', text)
    text = re.sub(r'(?mi)^-\s*Suggested\s+Tire:', '- Tire:', text)

    # fix any stray "Next; Next:" typos the model sometimes emits
    text = text.replace('Next; Next:', 'Next:')

    return text

def _fix_common_objections(text: str) -> str:
    """
    Reformat Common Objections into concise, scan-friendly single-line bullets:
    - <Objection> — Ask: … | Reframe: … | Proof: … | Next: …
    Preserves content, just tidies punctuation/spacing and ensures 'Next:' exists.
    """
    if not isinstance(text, str) or not text.strip():
        return text

    block_pat = r'(?s)(?:^|\n)Common Objections:\n(.*?)(?=\n[A-Z][^\n]*:|\Z)'
    m = re.search(block_pat, text)
    if not m:
        return text

    block = m.group(1)
    lines_in = [ln.strip() for ln in block.splitlines() if ln.strip()]
    out = []

    for ln in lines_in:
        # strip leading dash/quotes and normalize punctuation
        s = ln.lstrip('-• ').strip()
        s = s.replace('“', '"').replace('”', '"').replace("’", "'")
        s = s.strip('"').strip("'")

        # collapse multiple spaces and normalize separators
        s = re.sub(r'\s{2,}', ' ', s)
        s = s.replace(' — ', ' — ').replace('–', '—')  # unify dash
        s = s.replace('; Next:', ' | Next:')           # make Next separator consistent
        s = s.replace('; Reframe:', ' | Reframe:')
        s = s.replace('; Proof:', ' | Proof:')
        s = s.replace('; Ask:', ' | Ask:')
        s = s.replace('Next; ', 'Next: ')
        s = s.replace('Next; Next:', 'Next:')          # safety

        # ensure labeled segments are present in preferred order
        if 'Ask:' not in s and '—' in s:
            # try to split on '—' to get the objection text
            parts = [p.strip() for p in s.split('—', 1)]
            objection = parts[0]
            rest = parts[1] if len(parts) > 1 else ''
            s = f"{objection} — {rest}"
        if 'Next:' not in s:
            s += ' | Next: Schedule a brief site walk to confirm spec.'

        # keep to one tidy bullet
        s = re.sub(r'\s*\|\s*$', '', s).rstrip('.')
        out.append(f"- {s}.")

        if len(out) >= 6:  # keep it brief but useful
            break

    new_block = "Common Objections:\n" + "\n".join(out) + "\n"
    return re.sub(block_pat, "\n" + new_block, text)

def _tidy_formatting(text: str) -> str:
    if not isinstance(text, str):
        return text

    # Normalize bullets: turn • / · into "- "
    text = re.sub(r'(?m)^\s*[•·]\s+', "- ", text)

    # Remove extra blank lines after headers like "Section:\n\n" -> "Section:\n"
    text = re.sub(r'(?m)^([A-Z][A-Za-z \/&()-]*:)\s*\n+\s*', r'\1\n', text)

    # Collapse 3+ consecutive newlines -> 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Trim trailing spaces
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)

    return text

# --- Helpers to ground "Comparison:" on competitor JSON -------------------
import re
from difflib import get_close_matches

def _find_heli_model_by_code(code: str):
    """
    Locate the HELI model dict (from _load_models()) by its exact model code.
    Falls back to a fuzzy match if exact not found.
    """
    code_norm = (code or "").strip().lower()
    models = _load_models()
    for m in models:
        if str(m.get("model", "")).strip().lower() == code_norm:
            return m
    # fuzzy fallback
    names = [m.get("model", "") for m in models]
    guess = get_close_matches(code, names, n=1, cutoff=0.92)
    if guess:
        g = guess[0]
        for m in models:
            if m.get("model") == g:
                return m
    return None

def _format_peer_row(p: dict) -> str:
    """One competitor line: Brand Model — Capacity; turn; width; fuel."""
    brand = (p.get("brand") or "").strip()
    model = (p.get("model") or "").strip()
    cap   = _as_lb(p.get("capacity_lb"))
    turn  = _as_in(p.get("turning_in"))
    width = _as_in(p.get("width_in"))
    fuel  = (p.get("fuel") or "—").title()
    parts = [f"{brand} {model}".strip()]
    extras = []
    if cap and cap != "—":   extras.append(cap)
    if turn and turn != "—": extras.append(f"turn {turn}")
    if width and width != "—":extras.append(f"width {width}")
    if fuel and fuel != "—": extras.append(fuel)
    if extras:
        return f"{parts[0]} — " + "; ".join(extras)
    return parts[0]

def _build_peer_comparison_lines(top_model_code: str, K: int = 4) -> list[str]:
    """
    Use the chosen HELI top pick to fetch K closest peers and format stable comparison bullets.
    If we can't resolve peers, we return a safe generic line.
    """
    heli_model = _find_heli_model_by_code(top_model_code)
    if not heli_model:
        return ["Similar capacity trucks from other brands are available; lithium cushion 5k class typically compares well on TCO."]

    peers = find_best_competitors(heli_model, K=K) or []
    if not peers:
        return ["Compared with common 5k electric cushion trucks from other brands, lithium uptime/PM savings often lower TCO vs IC."]

    lines = []
    # 1) Lead line anchoring why HELI wins (tie to power/geometry if present)
    power_txt = (heli_model.get("power") or "").lower()
    drive_txt = (heli_model.get("drive_type") or "").lower()
    why_bits = []
    if "li-ion" in power_txt or "electric" in power_txt:
        why_bits.append("lithium uptime & lower routine PM")
    if "cushion" in drive_txt:
        why_bits.append("indoor traction and floor protection")
    if heli_model.get("_turning_in"):
        why_bits.append(f"tight turning ({_as_in(heli_model['_turning_in'])})")
    if heli_model.get("_overall_width_in"):
        why_bits.append(f"compact width ({_as_in(heli_model['_overall_width_in'])})")
    why = "; ".join(why_bits) if why_bits else "balanced maneuverability and total cost of ownership"
    lines.append(f"Top pick vs peers: HELI advantages typically include {why}.")

    # 2) One line per peer
    for p in peers:
        lines.append(_format_peer_row(p))

    # 3) Close with a practical compare action
    lines.append("We can demo against these peers on your dock to validate turning, lift, and cycle times.")
    return lines

def _inject_section(text: str, header: str, bullets: list[str]) -> str:
    """
    Replace the section 'header:' with our own bullet list.
    Fits your app's formatting: each bullet starts '- ' and sections are 'Header:' on their own line.
    """
    if not isinstance(text, str):
        return text
    block = header + ":\n" + "\n".join(f"- {b}" for b in bullets) + "\n"

    # Match the existing section: Header:\n- ... (until next Title: or end)
    pattern = r'(?:^|\n)' + re.escape(header) + r':\n(?:- .*\n?)*'
    if re.search(pattern, text, flags=re.MULTILINE):
        return re.sub(pattern, "\n" + block, text, flags=re.MULTILINE)
    else:
        # Append at the end if not present
        return text + ("\n" if not text.endswith("\n") else "") + block

# ─── Recommendation flow helper ──────────────────────────────────────────
def run_recommendation_flow(user_q: str) -> str:
    # Build context WITHOUT prepending account text into the parser input
    acct = find_account_by_name(user_q)
    prompt_ctx = generate_forklift_context(user_q, acct)

    # strict grounding list (helpmate picks)
    hits, allowed = select_models_for_question(user_q, k=5)
    allowed_block = allowed_models_block(allowed)
    print(f"[recommendation] allowed models: {allowed}")

    # Pick the top HELI model code (if any)
    top_pick_code = allowed[0] if allowed else None

    system_prompt = {
        "role": "system",
        "content": (
            "You are a friendly, expert Heli Forklift sales assistant.\n"
            "Output ONLY these sections in this order:\n"
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
            "Formatting rules (do not echo):\n"
            "- Each section header exactly as above, followed by lines that start with '- '. No other bullet symbols.\n"
            "- Keep spacing tight; no blank lines between a header and its bullets.\n"
            "- Use ONLY model codes from the ALLOWED MODELS block. Do not invent codes.\n"
            "- Under Model: ONE line '- Top Pick: <code> — brief why'; ONE line '- Alternates: <codes...>' (up to 4). If none allowed, output exactly '- No exact match from our lineup.'\n"
            "- Capacity/Tires/Attachments: summarize needs; if missing, say 'Not specified'.\n"
            "- Sales Pitch Techniques: concise but specific as instructed in earlier rules.\n"
            "- Common Objections: 6–8 items, one line each in the pattern: '- <Objection> — Ask: <diagnostic>; Reframe: <benefit>; Proof: <fact>; Next: <action>'.\n"
            "- Never invent pricing, availability, or specs not present in the context.\n"
        )
    }

    messages = [
        system_prompt,
        {"role": "system", "content": allowed_block},  # strict grounding list
        {"role": "user",   "content": prompt_ctx}
    ]

    try:
        # Start timer for fine-grained usage logging
        t0 = time.perf_counter()

        resp = client.chat.completions.create(
            model=os.getenv("OAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            max_tokens=650,
            temperature=0.4
        )

        # Compute duration and log token usage
        duration_ms = int((time.perf_counter() - t0) * 1000)
        log_model_usage(
            resp,
            endpoint="/chat",
            action="chat_reply",
            duration_ms=duration_ms,
            extra={"who": session.get("username")}
        )

        ai_reply = resp.choices[0].message.content.strip()

    except Exception as e:
        ai_reply = f"❌ Internal error: {e}"

    # Enforce model list & tidy
    ai_reply = _enforce_allowed_models(ai_reply, set(allowed))
    ai_reply = _unify_model_mentions(ai_reply, allowed) if '_unify_model_mentions' in globals() else ai_reply
    ai_reply = _fix_labels_and_breaks(ai_reply) if '_fix_labels_and_breaks' in globals() else ai_reply
    ai_reply = _fix_common_objections(ai_reply) if '_fix_common_objections' in globals() else ai_reply
    ai_reply = _tidy_formatting(ai_reply) if '_tidy_formatting' in globals() else ai_reply

    # >>> NEW: compute grounded peer comparison and inject it
    if top_pick_code:
        peer_lines = _build_peer_comparison_lines(top_pick_code, K=4)
        ai_reply = _inject_section(ai_reply, "Comparison", peer_lines)

    return ai_reply

# ─── Chat API (Recommendation + Inquiry + Coach) ─────────────────────────
@app.route("/api/chat", methods=["POST"])
@login_required
def chat():
    data   = request.get_json(force=True) or {}
    user_q = (data.get("question") or "").strip()
    mode   = (data.get("mode") or "recommendation").lower()

    if not user_q:
        return jsonify({"response": "Please enter a description of the customer’s needs."}), 400

    app.logger.info(f"/api/chat mode={mode} qlen={len(user_q)}")

    # ───────── Sales Coach mode ─────────
    if mode == "coach":
        ai_reply = run_sales_coach(user_q)   # helper can be defined later in this file
        return jsonify({"response": ai_reply})

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

    # ───────── Recommendation mode (default) ─────────
    user_q = user_q if 'user_q' in locals() else request.form.get('message') or request.args.get('q') or ''
    ai_reply = run_recommendation_flow(user_q)

    # >>> Inject Current Promotions (no top_pick_code needed)
    from ai_logic import top_pick_meta
    from promotions import promos_for_context, render_promo_lines

    meta = top_pick_meta(user_q)  # (model_code, class, power) inferred from your models
    if meta:
        top_code, top_class, top_power = meta
        # if power preference was in the user's text, it will already be reflected in meta; this is just a gentle override
        if re.search(r'\b(lpg|propane|lp gas)\b', user_q, re.I): top_power = "lpg"
        elif re.search(r'\bdiesel\b', user_q, re.I):             top_power = "diesel"
        elif re.search(r'\b(lithium|li[-\s]?ion|electric|battery)\b', user_q, re.I): top_power = "lithium"

        promo_list = promos_for_context(top_code, top_class, top_power or "")
        promo_lines = render_promo_lines(promo_list)
        if promo_lines:
            ai_reply = _inject_section(ai_reply, "Current Promotions", promo_lines)

    return jsonify({"response": ai_reply})

@app.post("/api/debug_recommend")
def api_debug_recommend():
    data = request.get_json(silent=True) or {}
    user_q = (
        data.get("q") or data.get("question") or
        request.form.get("q") or request.form.get("question") or ""
    ).strip()
    if not user_q:
        return jsonify({"error": "Missing 'q'"}), 400
    return jsonify(debug_parse_and_rank(user_q, limit=10))

# ─── Modes list ──────────────────────────────────────────────────────────
@app.route("/api/modes")
def api_modes():
    return jsonify([
        {"id": "recommendation", "label": "Forklift Recommendation"},
        {"id": "inquiry",        "label": "Customer Inquiry"},
        {"id": "coach",          "label": "Sales Coach"},
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
    Build map points from customer_location.csv (your exact headers),
    attach Sales Rep from customer_report.csv for territory coloring,
    and (NEW) attach Segment directly so popups never miss it.
    Also (NEW) include contact fields if present.
    """
    import csv, json as _json, re as _re
    import pandas as _pd
    from difflib import get_close_matches
    from flask import Response as _Response

    # -------- helpers --------
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

    # -------- build rep + segment indexes from customer_report.csv --------
    rep_idx_exact = {}
    rep_idx_norm = {}
    rep_idx_norm_zip = {}
    rep_idx_norm_city_state = {}

    seg_idx_exact = {}
    seg_idx_norm = {}
    seg_idx_norm_zip = {}
    seg_idx_norm_city_state = {}

    try:
        rep_df = _pd.read_csv("customer_report.csv", dtype=str).fillna("")
        HAVE_REP = "Sales Rep Name" in rep_df.columns
        SOLD_COL = "Sold to Name"
        SHIP_COL = "Ship to Name"
        CITY_COL = "City"
        ZIP_COL  = "Zip Code"
        CST_COL  = "County State"
        SEG_COL  = "R12 Segment (Sold to ID)"

        rep_df["_sold_norm"] = rep_df.get(SOLD_COL, "").apply(norm_name)
        rep_df["_ship_norm"] = rep_df.get(SHIP_COL, "").apply(norm_name)
        rep_df["_zip5"]      = rep_df.get(ZIP_COL, "").apply(zip5)
        rep_df["_city_norm"] = rep_df.get(CITY_COL, "").str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True)\
                                                     .str.replace(r"\s+", " ", regex=True).str.strip()
        rep_df["_state"]     = rep_df.get(CST_COL, "").apply(state_from_county_state)

        for _, r in rep_df.iterrows():
            seg = (r.get(SEG_COL, "") or "").strip()
            rep = (r.get("Sales Rep Name", "") or "").strip() if HAVE_REP else ""

            # exact
            for col in (SOLD_COL, SHIP_COL):
                nm = (r.get(col, "") or "").strip()
                if not nm:
                    continue
                if HAVE_REP and rep:
                    rep_idx_exact.setdefault(nm, rep)
                if seg:
                    seg_idx_exact.setdefault(nm, seg)

            # normalized keys
            for nval in (r.get("_sold_norm", ""), r.get("_ship_norm", "")):
                nval = (nval or "").strip()
                if not nval:
                    continue
                if HAVE_REP and rep:
                    rep_idx_norm.setdefault(nval, rep)
                if seg:
                    seg_idx_norm.setdefault(nval, seg)

                z = r.get("_zip5", "")
                if z:
                    if HAVE_REP and rep:
                        rep_idx_norm_zip.setdefault(f"{nval}|{z}", rep)
                    if seg:
                        seg_idx_norm_zip.setdefault(f"{nval}|{z}", seg)

                cn = r.get("_city_norm", "")
                st = r.get("_state", "")
                if cn and st:
                    key = f"{nval}|{cn}|{st}"
                    if HAVE_REP and rep:
                        rep_idx_norm_city_state.setdefault(key, rep)
                    if seg:
                        seg_idx_norm_city_state.setdefault(key, seg)

        rep_norm_keys = list(rep_idx_norm.keys())
        seg_norm_keys = list(seg_idx_norm.keys())
    except Exception as e:
        print("⚠️ customer_report.csv not available for rep/segment enrichment:", e)
        rep_norm_keys = []
        seg_norm_keys = []

    def lookup_rep(name: str, city: str, state: str, zipc: str) -> str | None:
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
            guess = get_close_matches(n, rep_norm_keys, n=1, cutoff=0.88)
            if guess:
                return rep_idx_norm.get(guess[0])
        return None

    def lookup_seg(name: str, city: str, state: str, zipc: str) -> str | None:
        if not name:
            return None
        if name in seg_idx_exact:
            return seg_idx_exact[name]
        n = norm_name(name)
        if not n:
            return None
        z5 = zip5(zipc)
        if z5 and f"{n}|{z5}" in seg_idx_norm_zip:
            return seg_idx_norm_zip[f"{n}|{z5}"]
        cn = norm_city(city)
        st = (state or "").upper()
        key_cs = f"{n}|{cn}|{st}" if cn and st else None
        if key_cs and key_cs in seg_idx_norm_city_state:
            return seg_idx_norm_city_state[key_cs]
        if n in seg_idx_norm:
            return seg_idx_norm[n]
        if seg_norm_keys:
            guess = get_close_matches(n, seg_norm_keys, n=1, cutoff=0.88)
            if guess:
                return seg_idx_norm.get(guess[0])
        return None

    # -------- build points from customer_location.csv --------
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

                # NEW: contact fields (may be blank)
                first = (row.get("First Name", "") or "").strip()
                last  = (row.get("Last Name", "") or "").strip()
                title = (row.get("Job Title", "") or "").strip()
                phone = (row.get("Phone", "") or "").strip()
                mobile= (row.get("Mobile", "") or "").strip()
                email = (row.get("Email", "") or "").strip()

                if lat is None or lon is None:
                    continue
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    continue

                rep = lookup_rep(name, city, state, zipc) or "Unassigned"
                seg = lookup_seg(name, city, state, zipc) or ""

                items.append({
                    "label": name or "Unknown",
                    "address": address,
                    "city": city,
                    "state": state,
                    "county": county,
                    "zip": zipc,
                    "sales_rep": rep,                   # for coloring
                    "Sales Rep Name": rep,              # compatibility – some UIs read this
                    "segment": seg,                     # NEW: server-provided segment
                    "lat": lat,
                    "lon": lon,
                    "County State": cs_raw,

                    # NEW: contacts
                    "first_name": first,
                    "last_name": last,
                    "job_title": title,
                    "phone": phone,
                    "mobile": mobile,
                    "email": email,
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

# --- Battle Cards (reads your JSON schema + routes + AI Fit) -----------------
import os, json, re
from functools import lru_cache
from flask import render_template, jsonify, request, abort

# Reuse existing OpenAI client if one is already created above
try:
    client  # noqa: F821
except NameError:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(BASE_DIR, "models.json")

# ------------------------------- helpers -------------------------------------
_num_re = re.compile(r"-?\d+(\.\d+)?")

def _canon(s: str) -> str:
    """Normalize header keys: lowercase -> trim -> collapse ws -> drop punctuation/spaces."""
    if s is None:
        return ""
    t = re.sub(r"\s+", " ", str(s)).strip().lower()
    t = re.sub(r"[^a-z0-9 ]+", "", t)
    return t.replace(" ", "")

NA_VALUES = {"", "na", "n/a", "n.a", "null", "none", "not applicable", "—", "-", "not specified"}
_NA_CANON = {re.sub(r"[^a-z0-9]+", "", v) for v in NA_VALUES}

def _is_na_value(v) -> bool:
    if v is None:
        return True
    t = re.sub(r"[^a-z0-9]+", "", str(v).strip().lower())
    return t in _NA_CANON

def _clean_or(v, fallback):
    return fallback if _is_na_value(v) else v

def _num_from_text(s):
    """Return the FIRST number in a string; '49-57' -> 49."""
    if s is None:
        return None
    m = _num_re.search(str(s))
    return float(m.group(0)) if m else None

def _fmt_int(n):
    try:
        if n is None:
            return None
        return f"{int(round(float(n))):,}"
    except (TypeError, ValueError):
        return None

def _fmt_in(n):  v = _fmt_int(n); return f"{v} in" if v is not None else None
def _fmt_lb(n):  v = _fmt_int(n); return f"{v} lb" if v is not None else None
def _fmt_v(n):   v = _fmt_int(n); return f"{v} V"  if v is not None else None

def _slugify(s):
    return re.sub(r"[^a-z0-9\-]+", "-", str(s).lower()).strip("-")

def _norm_power(p):
    if not p: return None
    t = str(p).strip().lower()
    if "lith" in t: return "Electric (Li-ion)"
    if t in {"electric", "lead acid", "lead-acid", "la", "acid"}: return "Electric (Lead-acid)"
    if "lpg" in t: return "LPG"
    if "diesel" in t: return "Diesel"
    return p

def _row_lookup(row: dict):
    """Build normalized-header → value map for a single row."""
    return {_canon(k): v for k, v in row.items()}

def _get(lut: dict, *keys):
    """Return first non-NA value for any normalized key in *keys."""
    for k in keys:
        v = lut.get(k)
        if not _is_na_value(v):
            return v
    return None

# Canonical keys we will look for after _canon()
K = {
    # identity
    "model": {"model", "modelname", "modelnumber"},
    # series
    "series": {"series", "family", "productseries"},
    # power
    "power": {"power", "powertype", "powertrain"},
    # drive type (your JSON uses 'Type')
    "drive_type": {"drivetype", "drive", "drivesystem", "type"},
    # controller
    "controller": {"controller", "controllerbrand", "control"},
    # capacity (your JSON uses 'Capacity_lbs')
    "capacity_lbs": {"capacitylbs", "loadcapacitylbs", "ratedcapacity", "capacity", "ratedload"},
    # dimensions (your JSON uses *_in)
    "height_in": {"heightin", "overallheightin", "overallheight"},
    "width_in": {"widthin", "overallwidthin", "overallwidth"},
    "length_in": {"lengthin", "overalllengthin", "overalllength"},
    "liftheight_in": {"liftheightin", "maxliftingheightin", "maxlifthtin", "mastmaxheightin"},
    # optional extras
    "battery_v": {"batteryvoltage", "batteryv", "battvoltage", "battv", "voltage", "voltagev"},
    "wheel_base_in": {"wheelbase", "wheelbasein"},
    "turning_in": {"minoutsideturningradiusin", "outsideturningradiusin", "turningradiusin", "turningin"},
    "load_center_in": {"loadcenterin", "loadcenter", "lc"},
    "workplace": {"workplace", "environment", "application"},
}

def _normalize_record(rec):
    """One row -> clean dict for templates + AI (using your JSON keys)."""
    lut = _row_lookup(rec)

    # pull values with our tolerant keys
    raw_model = _get(lut, *K["model"])
    series     = _get(lut, *K["series"])
    power_raw  = _get(lut, *K["power"])
    drive      = _get(lut, *K["drive_type"])
    controller = _get(lut, *K["controller"])

    cap  = _num_from_text(_get(lut, *K["capacity_lbs"]))
    oh   = _num_from_text(_get(lut, *K["height_in"]))
    ow   = _num_from_text(_get(lut, *K["width_in"]))
    ol   = _num_from_text(_get(lut, *K["length_in"]))
    mlh  = _num_from_text(_get(lut, *K["liftheight_in"]))

    batt = _num_from_text(_get(lut, *K["battery_v"]))
    wbase= _num_from_text(_get(lut, *K["wheel_base_in"]))
    trn  = _num_from_text(_get(lut, *K["turning_in"]))
    lctr = _num_from_text(_get(lut, *K["load_center_in"]))
    workplace = _get(lut, *K["workplace"])

    power_norm = _norm_power(power_raw)

    # raw strings (fallbacks if numbers missing)
    cap_raw  = _get(lut, *K["capacity_lbs"])
    oh_raw   = _get(lut, *K["height_in"])
    ow_raw   = _get(lut, *K["width_in"])
    ol_raw   = _get(lut, *K["length_in"])
    mlh_raw  = _get(lut, *K["liftheight_in"])
    batt_raw = _get(lut, *K["battery_v"])
    wbase_raw= _get(lut, *K["wheel_base_in"])
    trn_raw  = _get(lut, *K["turning_in"])
    lctr_raw = _get(lut, *K["load_center_in"])

    model = {
        # identity
        "model": raw_model or "Unknown Model",
        "_display": raw_model or "Unknown Model",
        "_slug": None,  # set later

        # descriptors
        "series": _clean_or(series, "—"),
        "power": _clean_or(power_norm or power_raw, "—"),
        "drive_type": _clean_or(drive, "—"),
        "controller": _clean_or(controller, "—"),

        # strings with units (UI)
        "capacity": _fmt_lb(cap)           or _clean_or(cap_raw, "Not specified"),
        "turning_radius": _fmt_in(trn)     or _clean_or(trn_raw, "Not specified"),
        "load_center": _fmt_in(lctr)       or _clean_or(lctr_raw, "Not specified"),
        "battery_voltage": _fmt_v(batt)    or _clean_or(batt_raw, "Not specified"),
        "wheel_base": _fmt_in(wbase)       or _clean_or(wbase_raw, "Not specified"),
        "overall_height": _fmt_in(oh)      or _clean_or(oh_raw, "Not specified"),   # Height_in
        "overall_length": _fmt_in(ol)      or _clean_or(ol_raw, "Not specified"),   # Length_in
        "overall_width": _fmt_in(ow)       or _clean_or(ow_raw, "Not specified"),   # Width_in
        "max_lift_height": _fmt_in(mlh)    or _clean_or(mlh_raw, "Not specified"),  # LiftHeight_in

        # numeric cache
        "_capacity_lb": cap, "_turning_in": trn, "_load_center_in": lctr, "_battery_v": batt,
        "_wheel_base_in": wbase, "_overall_width_in": ow, "_overall_length_in": ol,
        "_overall_height_in": oh, "_max_lift_height_in": mlh,

        "workplace": _clean_or(workplace, None),
    }

    # Light, spec-driven enrichment
    why, env_in, env_out, env_special = [], [], [], []
    if "Electric" in model["power"]:
        why += ["Zero local emissions", "Lower routine maintenance vs. IC", "Quiet operation"]
        if "Li-ion" in model["power"]:
            why += ["Opportunity charging; uptime friendly"]
            env_special += ["Cold storage friendly vs. lead-acid"]
    if model["drive_type"] and "three wheel" in str(model["drive_type"]).lower():
        why += ["Tight turning in narrow aisles"]; env_in += ["Narrow aisles, docks, staging"]
    if model["_overall_width_in"] and model["_overall_width_in"] <= 36:
        env_in += ["Very tight aisle layouts"]; why += [f"Compact width ({model['overall_width']})"]
    if model["_turning_in"] and model["_turning_in"] <= 60:
        why += [f"Small turning radius ({model['turning_radius']})"]
    if model["_capacity_lb"]:
        why += [f"Right-sized capacity ({model['capacity']})"]

    if not env_in: env_in = ["General warehouse"]
    if not env_out: env_out = ["Smooth outdoor pavement (light duty)"]
    if not env_special: env_special = ["Not specified"]

    model["indoors"] = ", ".join(dict.fromkeys(env_in))
    model["outdoors"] = ", ".join(dict.fromkeys(env_out))
    model["special_env"] = ", ".join(dict.fromkeys(env_special))
    model["why_wins"] = list(dict.fromkeys(why)) or ["Not specified"]
    return model

@lru_cache(maxsize=1)
def _load_models():
    """Read models.json as list OR {'models': [...]} OR {'data': [...]}."""
    with open(MODELS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    rows = raw if isinstance(raw, list) else raw.get("models", raw.get("data", []))
    models, used = [], set()
    for i, row in enumerate(rows or []):
        m = _normalize_record(row)
        base = m["model"] if m["model"] and m["model"] != "Unknown Model" else (
            row.get("Model") or row.get("Model Name") or f"unknown-{i+1}"
        )
        slug = _slugify(base) or f"unknown-{i+1}"
        if slug in used: slug = f"{slug}-{i+1}"
        used.add(slug)
        m["_slug"] = slug
        m["_display"] = m["model"]
        models.append(m)
    return models

def _find_model_by_slug(slug: str):
    for m in _load_models():
        if m["_slug"] == slug:
            return m
    return None

# -------------------------------- routes -------------------------------------
@app.route("/battlecards")
def battlecards_index():
    models = _load_models()
    q = (request.args.get("q") or "").strip().lower()
    if q:
        models = [m for m in models if q in json.dumps(m).lower()]
    return render_template("battlecards_index.html", models=models, q=q)

@app.route("/battlecards/<slug>")
def battlecard_view(slug):
    model = _find_model_by_slug(slug)
    if not model:
        abort(404)
    return render_template("battlecard.html", model=model)

# ---------------------------- Ask AI for Fit ---------------------------------
@app.route("/api/ai_fit")
def api_ai_fit():
    slug = (request.args.get("model") or "").strip()
    m = _find_model_by_slug(slug)
    if not m:
        return jsonify({"error": "Model not found"}), 404

    # HELI model ground truth
    model_spec = {
        "Brand": "HELI",
        "Model": m["model"],
        "Series": m.get("series", "—"),
        "Power": m.get("power", "—"),
        "Drive Type": m.get("drive_type", "—"),
        "Controller": m.get("controller", "—"),
        "Capacity": m.get("capacity", "Not specified"),
        "Load Center": m.get("load_center", "Not specified"),
        "Turning Radius": m.get("turning_radius", "Not specified"),
        "Overall Width": m.get("overall_width", "Not specified"),
        "Overall Length": m.get("overall_length", "Not specified"),
        "Overall Height": m.get("overall_height", "Not specified"),
        "Wheel Base": m.get("wheel_base", "Not specified"),
        "Max Lift Height": m.get("max_lift_height", "Not specified"),
        "Indoors": m.get("indoors", "Not specified"),
        "Outdoors": m.get("outdoors", "Not specified"),
        "Special Environments": m.get("special_env", "Not specified"),
        "Why Wins (precomputed hints)": "; ".join(m.get("why_wins", [])) or "Not specified",
    }
    spec_lines = "\n".join(f"- {k}: {v}" for k, v in model_spec.items())

    # Numeric cache for HELI (lets the LLM do light math properly)
    heli_nums = {
        "capacity_lb": m.get("_capacity_lb"),
        "turning_in": m.get("_turning_in"),
        "width_in": m.get("_overall_width_in"),
        "max_lift_in": m.get("_max_lift_height_in"),
    }
    heli_num_lines = "\n".join(f"- {k}: {v if v is not None else '—'}" for k, v in heli_nums.items())

    # Build brand-coverage peer set and our own clean HTML table
    peers = find_brand_coverage_peers(m, max_rows=12)
    comp_ctx = _build_comp_context(m, peers)  # uses your existing stats builder
    peer_table_html = _peer_table_html(peers) # consistent spacing & styling

    # Prompt: model writes text and inserts [[PEER_TABLE]] token; we replace it afterward
    system = (
        "You are a forklift sales engineer.\n"
        "Use MODEL_SPEC and MODEL_SPEC_NUMERIC as ground truth for this HELI model.\n"
        "Use COMP_PEERS / PEER_STATS verbatim for competitor info. Do NOT invent numbers.\n"
        "Do not wrap your output in <html> or <body>. Do not add horizontal rules."
    )

    user = f"""MODEL_SPEC
{spec_lines}

MODEL_SPEC_NUMERIC
{heli_num_lines}

{comp_ctx}
Produce HTML with two <section> blocks, each wrapped with a root <div> that has data-block attribute:

<div data-block="fit">…</div>
<div data-block="edge">…</div>

### FIT BLOCK (deep spec briefing)
In <div data-block="fit">:
- 1–2 sentence Overview (who/where/why).
- Key Specs (list): Capacity, Load Center, Turning Radius, Overall Width, Max Lift Height, Power, Drive Type, Controller.
- Why It Wins (5–7 bullets): tie each bullet to a SPEC number or clear attribute.
- Best Environments (3–5 bullets).
- Fit Checklist (5 bullets) referencing concrete specs.
- Watchouts (2–4 bullets; mention “Not specified” where relevant).

### EDGE BLOCK (comparison vs peers)
In <div data-block="edge">:
- One concise paragraph explaining how this HELI model stacks up vs peers using COMP_PEERS/PEER_STATS only.
- Insert this exact token on its own line where the comparison table should appear:
[[PEER_TABLE]]
- Then add 'Where HELI Stands Out' (3–5 bullets). Use numbers to support claims when HELI beats the peer averages; otherwise focus on power/maintenance/environments grounded in MODEL_SPEC.
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.25,
        )
        html = resp.choices[0].message.content.strip()
        # Inject our clean table and tidy spacing/wrappers
        html = html.replace('[[PEER_TABLE]]', peer_table_html)
        html = _strip_root_html(html)
        html = _collapse_table_spacing(html)

        if 'data-block="fit"' not in html or 'data-block="edge"' not in html:
            html = (
                "<div data-block=\"fit\"><p>Briefing unavailable.</p></div>"
                "<div data-block=\"edge\"><p>Competitive edge unavailable.</p></div>"
            )
        return jsonify({"html": html})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# ===================== Competitor Data (non-breaking add-on) =====================
# This section only ADDS functionality. No existing functions are modified.

from functools import lru_cache

# Where the converted competitor sheet lives (see earlier converter script)
COMPETITORS_PATH = os.path.join(BASE_DIR, "data", "heli_comp_models.json")

@lru_cache(maxsize=1)
def _load_competitors():
    """Load pre-converted competitor data (list of dicts). Safe if file is missing."""
    try:
        with open(COMPETITORS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []

def _heli_family(heli_model: dict) -> str:
    """
    Lightweight family inference for your HELI model (used for peer matching).
    Adjust if you want finer grouping.
    """
    p = (heli_model.get("power") or "").lower()
    d = (heli_model.get("drive_type") or "").lower()

    # Electric
    if "electric" in p:
        if "cushion" in d: return "Electric Cushion"
        # If drive_type doesn't include 'cushion', prefer pneumatic bucket by default
        return "Electric Pneumatic"

    # IC
    if "diesel" in p or "lpg" in p or "ic" in p:
        if "cushion" in d: return "IC Cushion"
        return "IC Pneumatic"

    return "Other/Unknown"

def _abs(v): 
    try: 
        return abs(float(v))
    except Exception:
        return 0.0

def _dist(a, b):
    if a is None and b is None: 
        return 0.0
    if a is None or b is None:
        # penalize missing values slightly
        return 10.0
    try:
        return abs(float(a) - float(b))
    except Exception:
        return 10.0

def _as_lb(v):
    # Reuse your existing formatters if available
    s = _fmt_lb(v)
    return s or ("—")

def _as_in(v):
    s = _fmt_in(v)
    return s or ("—")

def find_best_competitors(heli_model: dict, K: int = 3):
    """
    Score competitors by (a) same family, (b) closest capacity, and
    lightly consider turning radius and overall width if present.
    Return top-K competitor dicts.
    """
    comp = _load_competitors()
    if not comp:
        return []

    fam = _heli_family(heli_model)
    cap = heli_model.get("_capacity_lb") or _num_from_text(heli_model.get("capacity"))
    turn = heli_model.get("_turning_in") or _num_from_text(heli_model.get("turning_radius"))
    width = heli_model.get("_overall_width_in") or _num_from_text(heli_model.get("overall_width"))

    scored = []
    for row in comp:
        score = 0.0

        # Family match is very important
        score += 0 if (row.get("family") == fam) else 30

        # Capacity closeness (per 100 lb)
        score += _dist(cap, row.get("capacity_lb")) / 100.0

        # Gentle nudges for geometry similarity
        if turn and row.get("turning_in"):
            score += _dist(turn, row.get("turning_in")) / 50.0
        if width and row.get("width_in"):
            score += _dist(width, row.get("width_in")) / 50.0

        scored.append((score, row))

    scored.sort(key=lambda x: x[0])
    return [r for _, r in scored[:K]]

def _peer_table_html(peers: list):
    """Small, responsive table for the Competitive Edge tab."""
    rows = []
    for p in peers:
        rows.append(
            f"<tr>"
            f"<td>{(p.get('brand') or '')}</td>"
            f"<td>{(p.get('model') or '')}</td>"
            f"<td>{(p.get('family') or '')}</td>"
            f"<td class='num'>{_as_lb(p.get('capacity_lb'))}</td>"
            f"<td class='num'>{_as_in(p.get('turning_in'))}</td>"
            f"<td class='num'>{_as_in(p.get('width_in'))}</td>"
            f"<td>{(p.get('fuel') or '—').title()}</td>"
            f"</tr>"
        )
    if not rows:
        rows = ["<tr><td colspan='7'>No close peers found.</td></tr>"]

    return (
        "<div class='peer-table-wrap'>"
        "<table class='peer-table'>"
        "<thead>"
        "<tr>"
        "<th>Brand</th><th>Model</th><th>Family</th>"
        "<th class='num'>Capacity</th><th class='num'>Turning</th><th class='num'>Width</th><th>Fuel</th>"
        "</tr>"
        "</thead>"
        "<tbody>"
        + "".join(rows) +
        "</tbody></table></div>"
    )

def _peer_stats(peers: list[dict]) -> dict:
    """Compute simple ranges/averages for capacity/turning/width and list fuels."""
    vals = {"capacity_lb": [], "turning_in": [], "width_in": []}
    fuels = set()
    for p in peers or []:
        for k in vals.keys():
            v = p.get(k)
            try:
                if v is None or str(v).strip() == "":
                    continue
                vals[k].append(float(v))
            except Exception:
                pass
        fu = (p.get("fuel") or p.get("power") or "").strip()
        if fu:
            fuels.add(fu.title())

    def stat(arr):
        if not arr:
            return None, None, None
        return min(arr), (sum(arr) / len(arr)), max(arr)

    cmin, cavg, cmax = stat(vals["capacity_lb"])
    tmin, tavg, tmax = stat(vals["turning_in"])
    wmin, wavg, wmax = stat(vals["width_in"])
    return {
        "capacity": {"min": cmin, "avg": cavg, "max": cmax},
        "turning":  {"min": tmin, "avg": tavg, "max": tmax},
        "width":    {"min": wmin, "avg": wavg, "max": wmax},
        "fuels": sorted(fuels),
    }

def _build_comp_context(heli_model: dict, peers: list[dict]) -> str:
    """
    Produce a compact block the LLM can use directly. Includes each peer row
    plus aggregated stats (min/avg/max) for capacity/turning/width and a fuel list.
    """
    if not peers:
        return "COMP_PEERS\n- None found\nPEER_STATS\n- N/A\n"

    stats = _peer_stats(peers)
    lines = []
    lines.append("COMP_PEERS")
    for p in peers:
        brand = p.get("brand", "")
        model = p.get("model", "")
        fam   = p.get("family", "")
        cap   = p.get("capacity_lb", None)
        turn  = p.get("turning_in", None)
        wid   = p.get("width_in", None)
        fuel  = (p.get("fuel") or p.get("power") or "").title()
        cap_s = f"{int(round(float(cap))):,} lb" if cap else "—"
        turn_s= f"{int(round(float(turn))):,} in" if turn else "—"
        wid_s = f"{int(round(float(wid))):,} in" if wid else "—"
        lines.append(f"- {brand} {model} | Family: {fam or '—'} | Capacity: {cap_s} | Turning: {turn_s} | Width: {wid_s} | Fuel: {fuel or '—'}")

    def _fmt_rng(d, unit):
        if not d: return "—"
        mn = f"{int(round(d['min'])):,} {unit}" if d['min'] is not None else "—"
        av = f"{int(round(d['avg'])):,} {unit}" if d['avg'] is not None else "—"
        mx = f"{int(round(d['max'])):,} {unit}" if d['max'] is not None else "—"
        return f"min {mn} | avg {av} | max {mx}"

    lines.append("")
    lines.append("PEER_STATS")
    lines.append(f"- Capacity: {_fmt_rng(stats.get('capacity'), 'lb')}")
    lines.append(f"- Turning: {_fmt_rng(stats.get('turning'), 'in')}")
    lines.append(f"- Width: {_fmt_rng(stats.get('width'), 'in')}")
    fuels = ", ".join(stats.get("fuels", [])) or "—"
    lines.append(f"- Fuels: {fuels}")

    return "\n".join(lines) + "\n"

def _strip_root_html(text: str) -> str:
    """Remove <html>/<body> wrappers, code fences, and horizontal rules."""
    if not isinstance(text, str) or not text.strip():
        return text
    t = text
    # code fences
    t = re.sub(r'```(?:html|HTML)?\s*', '', t).replace('```', '')
    # <html>/<body> wrappers
    t = re.sub(r'(?is)^\s*<html[^>]*>\s*<body[^>]*>\s*', '', t)
    t = re.sub(r'(?is)\s*</body>\s*</html>\s*$', '', t)
    # horizontal rules
    t = re.sub(r'(?m)^\s*[-_]{3,}\s*$', '', t)
    t = re.sub(r'(?is)<hr[^>]*>\s*', '', t)
    return t.strip()

def _collapse_table_spacing(text: str) -> str:
    """Tidy whitespace: collapse blank lines and trim trailing spaces."""
    if not isinstance(text, str) or not text.strip():
        return text
    t = re.sub(r'\n{3,}', '\n\n', text)
    t = re.sub(r'[ \t]+\n', '\n', t)
    return t.strip()

def find_brand_coverage_peers(heli_model: dict, max_rows: int = 10, per_brand: int = 1):
    """
    Pick the best comparable for as many brands as possible (one per brand),
    prioritizing same family and closest capacity/geometry.
    """
    comp = _load_competitors()
    if not comp:
        return []

    fam   = _heli_family(heli_model)
    cap   = heli_model.get("_capacity_lb") or _num_from_text(heli_model.get("capacity"))
    turn  = heli_model.get("_turning_in") or _num_from_text(heli_model.get("turning_radius"))
    width = heli_model.get("_overall_width_in") or _num_from_text(heli_model.get("overall_width"))

    scored = []
    for row in comp:
        brand = (row.get("brand") or "").strip()
        if not brand:
            continue
        score = 0.0
        score += 0 if (row.get("family") == fam) else 20.0           # prefer same family
        score += _dist(cap, row.get("capacity_lb")) / 100.0          # capacity closeness
        if turn and row.get("turning_in"):
            score += _dist(turn, row.get("turning_in")) / 75.0       # geometry nudges
        if width and row.get("width_in"):
            score += _dist(width, row.get("width_in")) / 75.0
        scored.append((brand, score, row))

    best_by_brand = {}
    for brand, score, row in sorted(scored, key=lambda x: x[1]):
        if best_by_brand.get(brand) is None:
            best_by_brand[brand] = row

    peers = list(best_by_brand.values())[:max_rows]
    return peers

@app.get("/api/competitor_peers")
def api_competitor_peers():
    """
    Returns a small HTML table and raw JSON rows for the closest competitor matches.
    Non-breaking: you can optionally fetch this in your edge tab and append the table.
    Usage: /api/competitor_peers?model=<slug>&k=3
    """
    slug = (request.args.get("model") or "").strip()
    k = request.args.get("k", "").strip()
    try:
        K = int(k) if k else 3
    except Exception:
        K = 3

    heli_model = _find_model_by_slug(slug)
    if not heli_model:
        return jsonify({"error": "Model not found"}), 404

    peers = find_best_competitors(heli_model, K=K)
    html = _peer_table_html(peers)

    # Also return the rows if you want to render in the client
    return jsonify({"html": html, "rows": peers})

# --- Chat competitor helpers --------------------------------------------------

def _normkey(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

def _find_heli_model_by_name_loose(name: str):
    """Match a HELI model by name against the normalized models from _load_models()."""
    if not name:
        return None
    target = _normkey(name)
    pool = _load_models()
    # exact (normalized) match
    for m in pool:
        if _normkey(m.get("model")) == target:
            return m
    # close match fallback
    try:
        import difflib
        names = [m.get("model","") for m in pool]
        match = difflib.get_close_matches(name, names, n=1, cutoff=0.82)
        if match:
            for m in pool:
                if m.get("model") == match[0]:
                    return m
    except Exception:
        pass
    return None

def _build_competitor_block_for_model(heli_model_name: str, k: int = 4) -> str:
    """
    Returns a compact, numeric peer list the model can use directly.
    Example:
    COMPETITOR PEERS:
    - CAT GC40K: 8,000 lb; turn 90 in; width 47 in; LPG
    """
    if not heli_model_name:
        return "COMPETITOR PEERS:\n(none)"
    heli = _find_heli_model_by_name_loose(heli_model_name)
    if not heli:
        return "COMPETITOR PEERS:\n(none)"
    peers = find_best_competitors(heli, K=k) or []
    if not peers:
        return "COMPETITOR PEERS:\n(none)"

    lines = ["COMPETITOR PEERS:"]
    for p in peers:
        cap  = _as_lb(p.get("capacity_lb"))
        turn = _as_in(p.get("turning_in"))
        wid  = _as_in(p.get("width_in"))
        fuel = (p.get("fuel") or "—")
        brand = (p.get("brand") or "").strip()
        model = (p.get("model") or "").strip()
        lines.append(f"- {brand} {model}: {cap}; turn {turn}; width {wid}; {fuel}")
    return "\n".join(lines)

# --- Sales Coach helpers -------------------------------------------------

def _coach_detect_submode(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ["roleplay", "role-play", "mock call", "practice", "simulate"]):
        return "roleplay"
    if any(k in t for k in ["critique", "rewrite", "tune", "improve", "score my pitch"]):
        return "critique"
    if any(k in t for k in ["objection", "objections", "handle objections", "pushback"]):
        return "objections"
    if any(k in t for k in ["discovery", "call plan", "meeting plan", "questions to ask"]):
        return "discovery"
    if any(k in t for k in ["email", "voicemail", "linkedin", "dm", "follow-up", "follow up"]):
        return "messaging"
    if any(k in t for k in ["roi", "tco", "total cost", "business case"]):
        return "roi"
    return "coach_general"


def run_sales_coach(user_q: str) -> str:
    mode = _coach_detect_submode(user_q)

    # One system prompt, branching instructions per sub-mode
    base_rules = (
        "You are an expert forklift sales coach for a HELI dealership.\n"
        "Constraints (do NOT echo):\n"
        "- Use short headers exactly as requested for each sub-mode below.\n"
        "- Bullets must start with '- ' (hyphen + space). No other bullet symbols.\n"
        "- Keep spacing tight: no blank lines inside sections; one blank line between sections max.\n"
        "- Never invent pricing, lead times, or competitor exact specs; keep competitor claims generic.\n"
        "- If info is missing, include a short 'Questions to Clarify:' section with 2–3 bullets.\n"
    )

    # Per-submode formatting guides
    if mode == "roleplay":
        guide = (
            "Output ONLY these sections:\n"
            "Scenario:\n"
            "Opening:\n"
            "Roleplay:\n"
            "Branching Responses:\n"
            "Next Actions:\n"
            "\n"
            "Rules for this sub-mode (do NOT echo):\n"
            "- 'Roleplay:' contains alternating lines like '- Rep: …' and '- Prospect: …' (6–10 lines total).\n"
            "- 'Branching Responses:' include 2–3 likely forks ('price push', 'electric skepticism', etc.) with 1–2 lines each.\n"
            "- 'Next Actions:' are 3 concrete steps the rep should take after the call.\n"
        )
    elif mode == "critique":
        guide = (
            "Output ONLY these sections:\n"
            "Diagnosis:\n"
            "Rewrite (≤120 words):\n"
            "Talk Tracks:\n"
            "Next Actions:\n"
            "Questions to Clarify:\n"
            "\n"
            "Rules for this sub-mode (do NOT echo):\n"
            "- 'Diagnosis:' 5–7 bullets (clarity, relevance to buyer, proof, value, CTA, structure).\n"
            "- 'Rewrite (≤120 words):' one tight paragraph, conversational, outcome-led.\n"
            "- 'Talk Tracks:' 3 bullets (short, customer-facing lines).\n"
            "- 'Next Actions:' 3 bullets (immediately usable steps).\n"
        )
    elif mode == "objections":
        guide = (
            "Output ONLY these sections:\n"
            "Objection Handling Pack:\n"
            "Questions to Clarify:\n"
            "\n"
            "Rules for this sub-mode (do NOT echo):\n"
            "- Produce 6–8 lines under 'Objection Handling Pack:' each formatted exactly as\n"
            "  '- <Objection> — Ask: <diagnostic>; Reframe: <benefit>; Proof: <safe fact>; Next: <tiny step>'.\n"
        )
    elif mode == "discovery":
        guide = (
            "Output ONLY these sections:\n"
            "Call Goals:\n"
            "Discovery Questions:\n"
            "Red Flags:\n"
            "Commitment & Next Step:\n"
            "\n"
            "Rules for this sub-mode (do NOT echo):\n"
            "- 'Call Goals:' 3 bullets (what success looks like).\n"
            "- 'Discovery Questions:' 10 bullets grouped implicitly (loads, aisle/mast, duty cycle/power, surface/tires, safety/attachments, timeline/budget).\n"
            "- 'Red Flags:' 3–4 bullets.\n"
            "- 'Commitment & Next Step:' 2 bullets, very specific.\n"
        )
    elif mode == "messaging":
        guide = (
            "Output ONLY these sections:\n"
            "Subject:\n"
            "Email Body:\n"
            "Voicemail (≤15 sec):\n"
            "LinkedIn DM:\n"
            "Next Actions:\n"
            "\n"
            "Rules for this sub-mode (do NOT echo):\n"
            "- 'Subject:' 1 line, value-led.\n"
            "- 'Email Body:' 5–7 bullets or 5–7 short lines (no long paragraphs).\n"
            "- 'Voicemail (≤15 sec):' 2–3 lines the rep can read naturally.\n"
            "- 'LinkedIn DM:' 3–4 short lines.\n"
            "- 'Next Actions:' 2 bullets for follow-up cadence.\n"
        )
    elif mode == "roi":
        guide = (
            "Output ONLY these sections:\n"
            "Business Case Angle:\n"
            "Inputs to Collect:\n"
            "Talk Track:\n"
            "Risks & Mitigations:\n"
            "Next Actions:\n"
            "\n"
            "Rules for this sub-mode (do NOT echo):\n"
            "- No prices. Use variables (e.g., 'fuel_cost_per_hr', 'pm_cost_per_hr').\n"
            "- 'Talk Track:' 4 bullets, outcomes > features.\n"
        )
    else:  # coach_general
        guide = (
            "Output ONLY these sections:\n"
            "Quick Diagnosis:\n"
            "Suggestions:\n"
            "Talk Tracks:\n"
            "Next Actions:\n"
            "Questions to Clarify:\n"
            "\n"
            "Rules for this sub-mode (do NOT echo):\n"
            "- 4–6 bullets under 'Quick Diagnosis:' and 'Suggestions:'.\n"
            "- 'Talk Tracks:' 3 bullets.\n"
            "- 'Next Actions:' 3 bullets.\n"
        )

    system_prompt = {"role": "system", "content": base_rules + guide}
    messages = [system_prompt, {"role": "user", "content": user_q}]

    try:
        resp = client.chat.completions.create(
            model=os.getenv("OAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            max_tokens=700,
            temperature=0.5
        )
        out = resp.choices[0].message.content.strip()
    except Exception as e:
        out = f"❌ Internal error: {e}"

    # Optional: reuse your tidy helper if you have it
    try:
        out = _tidy_formatting(out)  # if defined elsewhere in your app
    except Exception:
        pass
    return out


# ─── Entrypoint ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=int(os.getenv("PORT", 5000)))
