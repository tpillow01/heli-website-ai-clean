"""
Flask app for the Heli AI sales assistant.
- Basic‑auth gate (uses RENDER_USERNAME / RENDER_PASSWORD env vars)
- Pulls account + model data from JSON
- Delegates business logic to ai_logic.py
"""
import os, json, difflib, tiktoken
from functools     import wraps
from flask         import Flask, render_template, request, jsonify, Response
from openai        import OpenAI

# ── internal helpers -----------------------------------------------------
from ai_logic import (
    generate_forklift_context,
    get_account          # <─  restored helper
)

# ── Flask + OpenAI client ------------------------------------------------
app    = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── simple basic‑auth wall ----------------------------------------------
def check_auth(u, p):
    return u == os.getenv("RENDER_USERNAME") and p == os.getenv("RENDER_PASSWORD")

def authenticate():
    return Response("Authentication required.", 401,
                    {"WWW-Authenticate": 'Basic realm="Login Required"'})

def requires_auth(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return fn(*a, **kw)
    return wrapper

# ── preload JSON (only for logging clarity) ------------------------------
with open("accounts.json", "r", encoding="utf-8") as f:
    _accts = json.load(f)
print(f"✅ Loaded {len(_accts)} accounts from JSON")

with open("models.json",   "r", encoding="utf-8") as f:
    _mods = json.load(f)
print(f"✅ Loaded {len(_mods)} models from JSON")

# ── routes ---------------------------------------------------------------
@app.route("/")
@requires_auth
def home():
    return render_template("chat.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    user_q = data.get("question", "").strip()
    if not user_q:
        return jsonify({"response": "Please describe the customer’s needs."}), 400

    # 1) try to identify a customer
    acct = get_account(user_q)  # substring + fuzzy search

    # 2) build prompt context for the LLM
    prompt_ctx = generate_forklift_context(user_q, acct)

    # 3) system prompt (UNCHANGED – exactly what you asked for)
    system_prompt = {
        "role": "system",
        "content": (
            "You are a helpful, detailed Heli Forklift sales assistant. "
            "When recommending models, format your response as plain text but wrap section headers in a "
            "<span class=\"section-label\">...</span> tag. "
            "Use the following sections: Model:, Power:, Capacity:, Tire Type:, Attachments:, Comparison:, "
            "Sales Pitch Techniques:, Common Objections:. "
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
    }

    messages = [
        system_prompt,
        {"role": "user", "content": prompt_ctx}
    ]

    # 4) trim if >7k tokens
    enc = tiktoken.encoding_for_model("gpt-4")
    trim = lambda m: sum(len(enc.encode(x["content"])) for x in m)
    while trim(messages) > 7000 and len(messages) > 2:
        messages.pop(1)

    # 5) call OpenAI
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

# ── entry‑point for gunicorn --------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=int(os.getenv("PORT", 5000)))
