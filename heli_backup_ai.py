# heli_backup_ai.py  ── main web service for Render
import os
import json
import difflib
import tiktoken
from flask import Flask, render_template, request, jsonify, Response
from functools import wraps
from openai import OpenAI
from ai_logic import generate_forklift_context   # ← your helper file

# ─── Flask & OpenAI client ───────────────────────────────────────────────
app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ─── BASIC‑AUTH HANDLING (unchanged) ─────────────────────────────────────
def check_auth(username, password):
    return (
        username == os.getenv("RENDER_USERNAME")
        and password == os.getenv("RENDER_PASSWORD")
    )

def authenticate():
    return Response(
        "Authentication required.", 401,
        {"WWW-Authenticate": 'Basic realm="Login Required"'}
    )

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

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

# ─── Web routes ──────────────────────────────────────────────────────────
@app.route("/")
@requires_auth
def home():
    return render_template("chat.html")

@app.route("/api/chat", methods=["POST"])
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

    # ── SYSTEM PROMPT (your original, plus 1 extra rule) ─────────────────
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
            # NEW LINE – forces GPT to stick to real model codes:
            "Only cite forklift **Model** codes exactly as they appear in the data "
            "(e.g. CPD25, CQD16). Never use only the Series name.\n\n"
            "At the end, include:\n"
            "- <span class=\"section-label\">Sales Pitch Techniques:</span> 1–2 persuasive points.\n"
            "- <span class=\"section-label\">Common Objections:</span> 1–2 common concerns and how to address them.\n\n"
            "<span class=\"section-label\">Example:</span>\n"
            "<span class=\"section-label\">Model:</span>\n"
            "- Heli H2000 Series 5‑7T\n"
            "- Designed for heavy‑duty applications\n\n"
            "<span class=\"section-label\">Power:</span>\n"
            "- Diesel\n"
            "- Provides high torque and durability\n\n"
            "<span class=\"section-label\">Sales Pitch Techniques:</span>\n"
            "- Emphasize Heli’s lower total cost of ownership.\n"
            "- Highlight that standard features are optional on other brands.\n\n"
            "<span class=\"section-label\">Common Objections:</span>\n"
            "- \"Why not Toyota or Crown?\"\n"
            "  → Heli offers similar quality at a better price with faster part availability."
        )
    }

    messages = [
        system_prompt,
        {"role": "user", "content": prompt_ctx}
    ]

    # ── Token‑limit guard (7000 ≈ safe for gpt‑4‑8k) ─────────────────────
    enc = tiktoken.encoding_for_model("gpt-4")
    while sum(len(enc.encode(m["content"])) for m in messages) > 7000 and len(messages) > 2:
        messages.pop(1)  # drop oldest user content (you only send one user message here)

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
