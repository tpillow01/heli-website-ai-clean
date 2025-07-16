# heli_backup_ai.py

import os
import json
import difflib
import tiktoken
from flask import Flask, render_template, request, jsonify, Response
from functools import wraps
from openai import OpenAI
from ai_logic import generate_forklift_context

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ─── Password Authentication ─────────────────────────────────────────────

def check_auth(username, password):
    return (
        username == os.getenv('RENDER_USERNAME')
        and password == os.getenv('RENDER_PASSWORD')
    )

def authenticate():
    return Response(
        'Authentication required.',
        401,
        {'WWW-Authenticate': 'Basic realm="Login Required"'}
    )

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

# ─── Load customer and model data ─────────────────────────────────────────
with open("accounts.json", "r", encoding="utf-8") as f:
    account_data = json.load(f)
print(f"✅ Loaded {len(account_data)} accounts from JSON")

with open("models.json", "r", encoding="utf-8") as f:
    model_data = json.load(f)
print(f"✅ Loaded {len(model_data)} models from JSON")

# Store conversation history
conversation_history = []

# Helper: substring-first then fuzzy match company name

def find_account_by_name(text: str):
    lower = text.lower()
    for acct in account_data:
        name = acct.get("Account Name", "").lower()
        if name in lower:
            return acct
    names = [acct.get("Account Name", "") for acct in account_data]
    match = difflib.get_close_matches(text, names, n=1, cutoff=0.7)
    if match:
        return next(a for a in account_data if a.get("Account Name") == match[0])
    return None

@app.route('/')
@requires_auth
def home():
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    global conversation_history
    data = request.get_json() or {}
    user_question = data.get('question', '').strip()
    if not user_question:
        return jsonify({'response': 'Please enter a description of the customer’s needs.'}), 400

    # Find matching account
    account = find_account_by_name(user_question)

    # Build context input including profile if available
    context_input = user_question
    if account:
        profile = account
        profile_ctx = (
            "<span class=\"section-label\">Customer Profile:</span>\n"
            f"- Company: {profile.get('Account Name','N/A')}\n"
            f"- Industry: {profile.get('Industry','N/A')}\n"
            f"- SIC Code: {profile.get('SIC Code','N/A')}\n"
            f"- Fleet Size: {profile.get('Total Company Fleet Size','N/A')}\n"
            f"- Truck Types: {profile.get('Truck Types at Location','N/A')}\n\n"
        )
        context_input = profile_ctx + user_question

    # Generate the AI context
    prompt_ctx = generate_forklift_context(context_input, account)

    # System prompt
    system_prompt = {
        "role": "system",
        "content": '''You are a helpful, detailed Heli Forklift sales assistant. When recommending models, format your response as plain text but wrap section headers in a <span class="section-label">...</span> tag. Use the following sections: Model:, Power:, Capacity:, Tire Type:, Attachments:, Comparison:, Sales Pitch Techniques:, Common Objections:. List details underneath using hyphens. Leave a blank line between sections. Indent subpoints for clarity.

At the end, include:
- Sales Pitch Techniques: 1–2 persuasive points.
- Common Objections: 1–2 common concerns and how to address them.

<span class="section-label">Example:</span>
<span class="section-label">Model:</span>
- Heli H2000 Series 5-7T
- Designed for heavy-duty applications

<span class="section-label">Power:</span>
- Diesel
- Provides high torque and durability

<span class="section-label">Sales Pitch Techniques:</span>
- Emphasize Heli’s lower total cost of ownership.
- Highlight that standard features are optional on other brands.

<span class="section-label">Common Objections:</span>
- "Why not Toyota or Crown?"
  → Heli offers similar quality at a better price with faster part availability.
'''  }

    # Build messages and trim tokens and trim tokens
    messages = [system_prompt, {"role": "user", "content": prompt_ctx}]
    enc = tiktoken.encoding_for_model("gpt-4")
    def count_tokens(msgs):
        return sum(len(enc.encode(m["content"])) for m in msgs)
    while count_tokens(messages) > 7000 and len(messages) > 2:
        messages.pop(1)

    # Call OpenAI
    ai_reply = ""  # initialize
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

    return jsonify({'response': ai_reply})

if __name__ == '__main__':
    app.run(debug=True, port=int(os.getenv('PORT', 5000)))
