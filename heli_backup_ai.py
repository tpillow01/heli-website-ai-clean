# heli_backup_ai.py

import json
import difflib
import tiktoken
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from ai_logic import generate_forklift_context

app = Flask(__name__)
client = OpenAI()  # Reads OPENAI_API_KEY from env

# Load accounts and models once at startup
with open("accounts.json", "r", encoding="utf-8") as f:
    account_data = json.load(f)
print(f"✅ Loaded {len(account_data)} accounts")

with open("models.json", "r", encoding="utf-8") as f:
    model_data = json.load(f)
print(f"✅ Loaded {len(model_data)} models")

conversation_history = []

def find_account_by_name(text: str):
    lower = text.lower()
    # 1) substring
    for acct in account_data:
        if acct["Account Name"].lower() in lower:
            return acct
    # 2) fuzzy fallback
    names = [a["Account Name"] for a in account_data]
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
    q = data.get("question", "").strip()
    if not q:
        return jsonify({'response': "Please describe the customer's needs."}), 400

    # Detect company
    acct = find_account_by_name(q)
    cust_name = acct["Account Name"] if acct else ""
    print("🔍 Matched company:", cust_name or "<none>")

    # Build context
    ctx = generate_forklift_context(q, cust_name, model_data)
    print("=== AI CONTEXT ===")
    print(ctx)
    print("==================")

    # Short history
    conversation_history = [{"role":"user","content":q}]

    # System prompt
    system = {
        "role": "system",
        "content": (
            "You are a Heli Forklift sales assistant.\n"
            "Wrap headings in <span class='section-label'>…</span> tags.\n"
            "Valid sections: Customer Profile:, Recommended Heli Models:, Model:, Type:, Power:, Capacity (lbs):,\n"
            "Dimensions (in):, Max Lifting Height (in):, Sales Pitch Techniques:, Common Objections:.\n"
            "Use hyphens and blank lines for readability."
        )
    }

    # Assemble messages
    messages = [system, {"role":"user","content":ctx}]

    encoding = tiktoken.encoding_for_model("gpt-4")
    def count_tokens(msgs):
        return sum(len(encoding.encode(m["content"])) for m in msgs)
    while count_tokens(messages) > 7000:
        messages.pop(1)

    # Call OpenAI
    try:
        res = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=500,
            temperature=0.2
        )
        reply = res.choices[0].message.content.strip()
    except Exception as e:
        print("❌ OpenAI error:", e)
        reply = f"Error: {e}"

    return jsonify({'response': reply})

if __name__ == '__main__':
    app.run(debug=True)
