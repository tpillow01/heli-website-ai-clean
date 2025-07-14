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

# ─── Load models.json for forklift recommendations ─────────────────────────────
with open("models.json", "r", encoding="utf-8") as f:
    model_data = json.load(f)
print(f"✅ Loaded {len(model_data)} forklift models from JSON")

# Conversation memory
conversation_history = []

# Fuzzy match customer name
def find_account_by_name(name):
    names = [acct.get("Account Name", "") for acct in account_data]
    match = difflib.get_close_matches(name, names, n=1, cutoff=0.6)
    if match:
        return next(acct for acct in account_data if acct["Account Name"] == match[0])
    return None

# Filter models based on customer industry
def filter_models_for_account(account):
    industry = account.get("Industry", "").lower()
    results = []
    for m in model_data:
        inds = [i.lower() for i in m.get("Industries", [])]
        if any(ind in industry for ind in inds):
            results.append(m)
    return results

# Format model blocks for inclusion in context
def format_models(models):
    if not models:
        return "- No suitable model matches found."
    blocks = []
    for m in models[:5]:
        blocks.append(
            f"<span class=\"section-label\">Model:</span> {m.get('Model')}\n"
            f"- Power: {m.get('Power')}\n"
            f"- Capacity: {m.get('Capacity')}\n"
            f"- Type: {m.get('Type')}\n"
        )
    return "\n".join(blocks)

@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    global conversation_history

    data = request.get_json()
    if not data:
        return jsonify({'response': 'Invalid request. Please send JSON.'}), 400

    user_question = data.get('question', '').strip()
    if not user_question:
        return jsonify({'response': 'Please enter a description of the customer’s needs.'}), 400

    # Attempt to match customer
    account = find_account_by_name(user_question)
    if account:
        # Use account info and filter models
        filtered = filter_models_for_account(account)
        model_ctx = format_models(filtered)
        acct_ctx = (
            f"Customer Account: {account.get('Account Name')}\n"
            f"Industry: {account.get('Industry')}\n\n"
        )
        combined_context = acct_ctx + model_ctx + "\n\n" + user_question
    else:
        # No account: general recommendations
        general_ctx = format_models(model_data)
        combined_context = "General Recommendations:\n" + general_ctx + "\n\n" + user_question

    # Build AI context and call OpenAI
    conversation_history.append({"role": "user", "content": user_question})
    if len(conversation_history) > 4:
        conversation_history.pop(0)

    system_prompt = {
        "role": "system",
        "content": (
            "You are a Heli Forklift sales assistant. Use only the provided models.json for recommendations and accounts.json when matching companies. "
            "Format section headers in <span class=\"section-label\"> tags and list details with hyphens."
        )
    }
    messages = [system_prompt, {"role": "user", "content": combined_context}] + conversation_history

    # Token management
    encoding = tiktoken.encoding_for_model("gpt-4")
    def num_tokens(msgs):
        return sum(len(encoding.encode(m["content"])) for m in msgs)
    while num_tokens(messages) > 7000 and len(messages) > 2:
        messages.pop(1)

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
        print("OpenAI API error:", e)
        ai_reply = "Error contacting AI."

    return jsonify({'response': ai_reply})

if __name__ == '__main__':
    app.run(debug=True)
