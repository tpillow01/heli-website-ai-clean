import os
import json
import difflib
import tiktoken
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from ai_logic import generate_forklift_context

app = Flask(__name__)
client = OpenAI()  # ✅ Uses environment variable for API key

# ─── Load accounts.json for customer matching ─────────────────────────────────
with open("accounts.json", "r", encoding="utf-8") as f:
    account_data = json.load(f)
print(f"✅ Loaded {len(account_data)} accounts from JSON")

# Conversation memory
dconversation_history = []

# Fuzzy match customer name
def find_account_by_name(name):
    names = [acct.get("Account Name", "") for acct in account_data]
    match = difflib.get_close_matches(name, names, n=1, cutoff=0.6)
    if match:
        return next(acct for acct in account_data if acct["Account Name"] == match[0])
    return None

# Placeholder model filter (unused for now)
def filter_models_for_account(account):
    return []

# Format model blocks for inclusion in prompt
def format_models(models):
    if not models:
        return "- No suitable model matches found."
    blocks = ""
    for m in models[:2]:
        blocks += (
            "<span class=\"section-label\">Suggested Model:</span>\n"
            f"- Model: {m.get('Model')}\n"
            f"- Power: {m.get('Power')}\n"
            f"- Capacity: {m.get('Capacity')}\n"
            f"- Type: {m.get('Type')}\n\n"
        )
    return blocks

@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    global conversation_history

    data = request.get_json()
    if not data:
        return jsonify({'response': 'Invalid request. Please send a JSON body.'}), 400

    user_question = data.get('question', '').strip()
    if not user_question:
        return jsonify({'response': 'Please enter a description of the customer’s needs.'}), 400

    # Attempt to match a customer
    account = find_account_by_name(user_question)
    customer_name = account.get('Account Name') if account else ''

    # Build combined context
    combined_context = generate_forklift_context(user_question, customer_name)
    conversation_history.append({"role": "user", "content": user_question})

    # Keep history short
    if len(conversation_history) > 4:
        conversation_history.pop(0)

    # System prompt defines overall behavior
    system_prompt = {
        "role": "system",
        "content": (
            "You are a helpful, detailed Heli Forklift sales assistant. "
            "When recommending models, format your response as plain text but wrap section headers in a <span class=\"section-label\">...</span> tag. "
            "Use sections: Model:, Power:, Capacity:, Tire Type:, Attachments:, Comparison:, Sales Pitch Techniques:, Common Objections:. "
            "List details with hyphens. Leave blank lines between sections.Indent subpoints.\n\n"
            "At the end, include:\n"
            "- Sales Pitch Techniques: 1–2 persuasive points.\n"
            "- Common Objections: 1–2 common concerns and how to address them."
        )
    }

    messages = [system_prompt, {"role": "user", "content": combined_context}] + conversation_history

    # Token management to avoid overflow
    encoding = tiktoken.encoding_for_model("gpt-4")
    def num_tokens_from_messages(msgs):
        return sum(len(encoding.encode(m["content"])) for m in msgs)

    while num_tokens_from_messages(messages) > 7000 and len(messages) > 2:
        messages.pop(1)

    # Call OpenAI
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=600,
            temperature=0.7
        )
        ai_reply = response.choices[0].message.content.strip()
        conversation_history.append({"role": "assistant", "content": ai_reply})
    except Exception as e:
        print("OpenAI API error:", e)
        ai_reply = "Something went wrong when contacting the AI. Please try again."

    return jsonify({'response': ai_reply})

if __name__ == '__main__':
    app.run(debug=True)
