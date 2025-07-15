# heli_backup_ai.py

import json
import difflib
import tiktoken
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from ai_logic import generate_forklift_context

app = Flask(__name__)
client = OpenAI()  # Uses OPENAI_API_KEY env var

# ─── Load accounts.json ──────────────────────────────────────────────────
with open("accounts.json", "r", encoding="utf-8") as f:
    account_data = json.load(f)
print(f"✅ Loaded {len(account_data)} accounts from JSON")

# ─── Load models.json ───────────────────────────────────────────────────
with open("models.json", "r", encoding="utf-8") as f:
    model_data = json.load(f)
print(f"✅ Loaded {len(model_data)} forklift models from JSON")

# ─── Conversation memory (in‑process) ──────────────────────────────────
conversation_history = []

def find_account_by_name(text: str):
    """
    First tries a direct substring match of any account name in the question text.
    If none found, falls back to fuzzy matching against full account names.
    Returns the account dict or None.
    """
    lower_text = text.lower()
    # 1) direct substring match
    for acct in account_data:
        name = acct.get("Account Name", "")
        if name.lower() in lower_text:
            return acct
    # 2) fallback to fuzzy match on full names
    names = [acct["Account Name"] for acct in account_data]
    match = difflib.get_close_matches(text, names, n=1, cutoff=0.6)
    if match:
        return next(a for a in account_data if a["Account Name"] == match[0])
    return None

@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    global conversation_history

    data = request.get_json() or {}
    user_question = data.get('question', '').strip()
    if not user_question:
        return jsonify({
            'response': 'Please describe the customer’s needs (you can mention a company name here).'
        }), 400

    # 1) Detect customer
    account = find_account_by_name(user_question)
    customer_name = account["Account Name"] if account else ""
    print(f"🔍 Matched customer_name: {customer_name or '<<none>>'}")

    # 2) Build context for the AI
    prompt_context = generate_forklift_context(user_question, customer_name, model_data)
    print("=== PROMPT CONTEXT START ===")
    print(prompt_context)
    print("=== PROMPT CONTEXT END ===")

    # 3) Maintain a short conversation history
    conversation_history.append({"role": "user", "content": user_question})
    if len(conversation_history) > 4:
        conversation_history.pop(0)

    # 4) System prompt
    system_prompt = {
        "role": "system",
        "content": (
            "You are a helpful, detailed Heli Forklift sales assistant.\n"
            "Wrap section headers in <span class=\"section-label\">…</span> tags.\n"
            "Valid sections (in order):\n"
            "Customer Profile:, Model:, Power:, Capacity:, Tire Type:, "
            "Attachments:, Comparison:, Sales Pitch Techniques:, Common Objections:.\n"
            "List each detail with hyphens and leave blank lines between sections.\n\n"
            "Example:\n"
            "<span class=\"section-label\">Customer Profile:</span>\n"
            "- Company: Acme Co\n"
            "- Industry: Retail\n"
            "- SIC Code: 5311\n\n"
            "<span class=\"section-label\">Model:</span>\n"
            "- Heli G Series 3–3.5T Electric Forklift\n\n"
            "<span class=\"section-label\">Power:</span>\n"
            "- Electric\n\n"
            "<span class=\"section-label\">Sales Pitch Techniques:</span>\n"
            "- Emphasize zero emissions.\n\n"
            "<span class=\"section-label\">Common Objections:</span>\n"
            "- \"Electric won’t last a full shift.\"\n"
            "  → Our G Series runs 8–10 hours on a single charge."
        )
    }

    # 5) Assemble and prune tokens
    messages = [system_prompt, {"role": "user", "content": prompt_context}] + conversation_history
    encoding = tiktoken.encoding_for_model("gpt-4")
    def count_tokens(msgs):
        return sum(len(encoding.encode(m["content"])) for m in msgs)

    while count_tokens(messages) > 7000 and len(messages) > 2:
        messages.pop(1)

    # 6) Call Chat API
    try:
        resp = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=600,
            temperature=0.7
        )
        ai_reply = resp.choices[0].message.content.strip()
        conversation_history.append({"role": "assistant", "content": ai_reply})
    except Exception as e:
        print("OpenAI error:", e)
        ai_reply = f"❌ Internal error: {e}"

    return jsonify({'response': ai_reply})


if __name__ == '__main__':
    app.run(debug=True)
