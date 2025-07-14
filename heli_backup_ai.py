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

# ─── Load models.json ONCE for forklift recommendations ────────────────────────
with open("models.json", "r", encoding="utf-8") as f:
    model_data = json.load(f)
print(f"✅ Loaded {len(model_data)} forklift models from JSON")

# Conversation history
conversation_history = []


def find_account_by_name(name):
    names = [acct.get("Account Name", "") for acct in account_data]
    match = difflib.get_close_matches(name, names, n=1, cutoff=0.6)
    if match:
        return next(acct for acct in account_data if acct["Account Name"] == match[0])
    return None


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

    # Fuzzy-match the account
    account = find_account_by_name(user_question)
    customer_name = account["Account Name"] if account else ""

    # Build the combined context (using the single model_data loaded above)
    filtered_context = generate_forklift_context(user_question, customer_name, model_data)

    # Append to history
    conversation_history.append({"role": "user", "content": user_question})
    if len(conversation_history) > 4:
        conversation_history.pop(0)

    # Your detailed system prompt, unchanged
    system_prompt = {
        "role": "system",
        "content": (
            "You are a helpful, detailed Heli Forklift sales assistant. "
            "When recommending models, format your response as plain text but wrap section headers in a "
            "<span class=\"section-label\">...</span> tag. "
            "Use the following sections: Model:, Power:, Capacity:, Tire Type:, Attachments:, Comparison:, "
            "Sales Pitch Techniques:, Common Objections:. List details underneath using hyphens. "
            "Leave a blank line between sections. Indent subpoints for clarity.\n\n"
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
            f"\n\n{filtered_context}"
        )
    }

    # Assemble messages in the correct order
    messages = [system_prompt] + conversation_history

    # Token‐limit safety
    encoding = tiktoken.encoding_for_model("gpt-4")
    def num_tokens(msgs):
        return sum(len(encoding.encode(m["content"])) for m in msgs)

    while num_tokens(messages) > 7000 and len(messages) > 2:
        messages.pop(1)

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
