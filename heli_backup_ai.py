# heli_backup_ai.py
import os, json, tiktoken
from flask import Flask, render_template, request, jsonify, Response
from functools import wraps
from openai import OpenAI
from ai_logic import (
    get_account,
    generate_forklift_context,
)

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ─── Basic‑Auth ───────────────────────────────────────────────────
def _ok(u, p):
    return u == os.getenv("RENDER_USERNAME") and p == os.getenv("RENDER_PASSWORD")

def _auth_required(fn):
    @wraps(fn)
    def wrapper(*a, **k):
        auth = request.authorization
        if not auth or not _ok(auth.username, auth.password):
            return Response("Authentication required.", 401,
                            {"WWW-Authenticate": 'Basic realm="Login Required"'})
        return fn(*a, **k)
    return wrapper

# ─── One HTML page ────────────────────────────────────────────────
@app.route("/")
@_auth_required
def home():
    return render_template("chat.html")

# ─── Chat endpoint ────────────────────────────────────────────────
SYSTEM_PROMPT_CONTENT = (
    "You are a helpful, detailed Heli Forklift sales assistant. "
    "When recommending models, format your response as plain text but wrap section headers in a <span class=\"section-label\">...</span> tag. "
    "Use the following sections: Model:, Power:, Capacity:, Tire Type:, Attachments:, Comparison:, Sales Pitch Techniques:, Common Objections:. "
    "List details underneath using hyphens. Leave a blank line between sections. "
    "Indent subpoints for clarity.\n\n"
    "At the end, include:\n"
    "- Sales Pitch Techniques: 1–2 persuasive points.\n"
    "- Common Objections: 1–2 common concerns and how to address them.\n\n"
    "<span class=\"section-label\">Example:</span>\n"
    "<span class=\"section-label\">Model:</span>\n"
    "- Heli H2000 Series 5-7T\n"
    "- Designed for heavy-duty applications\n\n"
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

enc = tiktoken.encoding_for_model("gpt-4")

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_q = (data.get("question") or "").strip()
    if not user_q:
        return jsonify({"response": "Please describe the customer’s needs."}), 400

    account = get_account(user_q)

    # ONLY raw question goes to the filter; profile markup happens inside generate_forklift_context
    prompt_ctx = generate_forklift_context(user_q, account)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_CONTENT},
        {"role": "user",   "content": prompt_ctx},
    ]

    # token‑limit guard (7000 ≈ 8k context – slack)
    while sum(len(enc.encode(m["content"])) for m in messages) > 7000 and len(messages) > 2:
        messages.pop(1)

    try:
        resp = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=600,
            temperature=0.7,
        )
        ai_reply = resp.choices[0].message.content.strip()
    except Exception as e:
        ai_reply = f"❌ Internal error: {e}"

    return jsonify({"response": ai_reply})

# ─── Run local ----------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=int(os.getenv("PORT", 5000)))
