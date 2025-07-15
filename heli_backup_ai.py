# heli_backup_ai.py

import os
import json
import difflib
import tiktoken
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from ai_logic import generate_forklift_context

app = Flask(__name__)
client = OpenAI()  # uses OPENAI_API_KEY env var

# ─── Load password from environment ───────────────────────────────────────
PASSWORD = os.getenv("PASSWORD")
if not PASSWORD:
    raise RuntimeError("PASSWORD environment variable not set")

# ─── Load JSON data ──────────────────────────────────────────────────────
with open("accounts.json", "r", encoding="utf-8") as f:
    accounts_raw = json.load(f)
print(f"✅ Loaded {len(accounts_raw)} accounts")

with open("models.json", "r", encoding="utf-8") as f:
    models_raw = json.load(f)
print(f"✅ Loaded {len(models_raw)} models")


def find_account_by_name(text: str) -> str:
    """
    1) Exact substring match (case‑insensitive) against Account Name
    2) Fuzzy match over all names if no exact
    """
    txt = text.lower()
    names = [acct.get("Account Name", "") for acct in accounts_raw]
    # exact match
    for name in names:
        if name and name.lower() in txt:
            return name
    # fuzzy fallback
    close = difflib.get_close_matches(text, names, n=1, cutoff=0.6)
    return close[0] if close else ""


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/api/chat", methods=["POST"])
def chat_api():
    payload = request.get_json() or {}
    user_q = payload.get("question", "").strip()
    pwd = payload.get("password", "")
    if pwd != PASSWORD:
        return jsonify({"error": "Invalid password"}), 401

    # 2) detect account
    customer = find_account_by_name(user_q)

    # 3) build context
    prompt_ctx = generate_forklift_context(user_q, customer)

    # 4) assemble messages with your original system prompt
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
            "List details underneath using hyphens, leave a blank line\n"
            "between sections, and indent subpoints for clarity.\n\n"
            "At the end, include:\n"
            "- <span class=\"section-label\">Sales Pitch Techniques:</span> 1–2 persuasive points.\n"
            "- <span class=\"section-label\">Common Objections:</span> 1–2 common concerns and how to address them.\n\n"
            "Example:\n"
            "<span class=\"section-label\">Customer Profile:</span>\n"
            "- Company: Acme Co\n"
            "- Industry: Retail\n"
            "- SIC Code: 5311\n\n"
            "<span class=\"section-label\">Model:</span>\n"
            "- Heli H2000 Series 5–7T Electric Forklift\n\n"
            "<span class=\"section-label\">Power:</span>\n"
            "- Diesel\n\n"
            "<span class=\"section-label\">Sales Pitch Techniques:</span>\n"
            "- Emphasize reliability.\n\n"
            "<span class=\"section-label\">Common Objections:</span>\n"
            "- \"Why not Toyota?\"\n"
            "  → Heli offers faster part availability.\n"
        )
    }
    user_msg = {"role": "user", "content": prompt_ctx}

    # 5) token‑limit guard (~7000 tokens for prompt)
    enc = tiktoken.encoding_for_model("gpt-4")
    total = len(enc.encode(system_prompt["content"])) + len(enc.encode(user_msg["content"]))
    max_tokens = 7000
    if total > max_tokens:
        over = total - max_tokens
        trimmed = enc.decode(enc.encode(user_msg["content"])[over:])
        user_msg["content"] = trimmed

    # 6) call OpenAI
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[system_prompt, user_msg],
        max_tokens=1024,
        temperature=0.2
    )

    # 7) return only the assistant’s reply
    answer = resp.choices[0].message.content
    return jsonify({"reply": answer})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
