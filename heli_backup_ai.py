import os
import json
import difflib
import tiktoken
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from ai_logic import generate_forklift_context

app = Flask(__name__)

# Load API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Load customer data
with open("accounts.json", "r", encoding="utf-8") as f:
    account_data = json.load(f)
print(f"✅ Loaded {len(account_data)} accounts from JSON")

# Load model data
with open("models.json", "r", encoding="utf-8") as f:
    model_data = json.load(f)
print(f"✅ Loaded {len(model_data)} models from JSON")

# Store conversation history
conversation_history = []

# Fuzzy match company name
def find_account_by_name(name):
    names = [acct.get("Account Name", "") for acct in account_data]
    match = difflib.get_close_matches(name, names, n=1, cutoff=0.6)
    if match:
        return next(acct for acct in account_data if acct["Account Name"] == match[0])
    return None

# Filter models using industry and truck types
def filter_models_for_account(account):
    industry = account.get("Industry", "").lower()
    truck_types = account.get("Truck Types at Location", "").lower()

    filtered = []
    for model in model_data:
        model_industries = [i.lower() for i in model.get("Industries", [])]
        model_truck_types = [t.lower() for t in model.get("Compatible Truck Types", [])]
        if any(ind in industry for ind in model_industries) or any(tt in truck_types for tt in model_truck_types):
            filtered.append(model)
    return filtered

# Limit models to reduce OpenAI token size
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
    data = request.json
    user_question = data.get('question', '').strip()

    if not user_question:
        return jsonify({'response': 'Please enter a description of the customer’s needs.'})

    account = find_account_by_name(user_question)
    extra_context = ""

    if account:
        filtered_models = filter_models_for_account(account)
        model_blocks = format_models(filtered_models)
        extra_context = f"""
Customer Profile:
- Industry: {account.get("Industry")}
- Fleet Size: {account.get("Total Company Fleet Size")}
- Truck Types: {account.get("Truck Types at Location")}

{model_blocks}
"""
    else:
        model_blocks = format_models(model_data[:2])
        extra_context = (
            "No customer data was found. Proceeding with general forklift recommendations.\n\n"
            + model_blocks
        )

    # Final prompt: user context + input
    user_question = extra_context + "\n\n" + user_question
    filtered_context = generate_forklift_context(user_question)

    # Trim old chat history
    if len(conversation_history) > 2:
        conversation_history = conversation_history[-2:]

    conversation_history.append({"role": "user", "content": user_question})

    # System prompt
    system_prompt = {
        "role": "system",
        "content": (
            "You are a helpful, detailed Heli Forklift sales assistant. "
            "When recommending models, format your response as plain text but wrap section headers in a <span class=\"section-label\">...</span> tag. "
            "Use the following sections: Model:, Power:, Capacity:, Tire Type:, Attachments:, Comparison:, Sales Pitch Techniques:, Common Objections:. "
            "Leave blank lines between sections.\n\n"
            "<span class=\"section-label\">Example:</span>\n"
            "<span class=\"section-label\">Model:</span>\n"
            "- Heli H2000 Series 5-7T\n"
            "- Designed for heavy-duty applications\n\n"
            "<span class=\"section-label\">Power:</span>\n"
            "- Diesel\n\n"
            "<span class=\"section-label\">Sales Pitch Techniques:</span>\n"
            "- Emphasize Heli’s lower total cost of ownership.\n\n"
            "<span class=\"section-label\">Common Objections:</span>\n"
            "- \"Why not Toyota or Crown?\"\n"
            "  → Heli offers similar quality at a better price."
            f"\n\n{filtered_context}"
        )
    }

    messages = [system_prompt] + conversation_history

    # Trim tokens to prevent 429/overload
    encoding = tiktoken.encoding_for_model("gpt-4")

    def num_tokens_from_messages(messages):
        return sum(len(encoding.encode(m["content"])) for m in messages)

    while num_tokens_from_messages(messages) > 7000 and len(messages) > 2:
        messages.pop(1)  # keep system prompt at index 0

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        ai_reply = response.choices[0].message.content.strip()
        conversation_history.append({"role": "assistant", "content": ai_reply})
    except Exception as e:
        ai_reply = f"Error contacting OpenAI: {e}"

    return jsonify({'response': ai_reply})

if __name__ == "__main__":
    app.run(debug=True, port=5004)
