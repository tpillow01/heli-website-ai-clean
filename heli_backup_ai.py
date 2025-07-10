import os
import json
import difflib
import tiktoken
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from ai_logic import generate_forklift_context

app = Flask(__name__)

client = OpenAI()  # ✅ Uses environment variable for API key

# Conversation memory
conversation_history = []

# Fuzzy match customer name
def find_account_by_name(name):
    return None

def filter_models_for_account(account):
    return []

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

    data = request.json
    user_question = data.get('question', '').strip()
    customer_name = data.get('customer', '').strip()

    if not user_question:
        return jsonify({'response': 'Please enter a description of the customer’s needs.'})

    combined_context = generate_forklift_context(user_question, customer_name)

    conversation_history.append({"role": "user", "content": user_question})

    if len(conversation_history) > 4:
        conversation_history.pop(0)

    system_prompt = {
        "role": "system",
        "content": (
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
    }

    messages = [system_prompt, {"role": "user", "content": combined_context}] + conversation_history

    encoding = tiktoken.encoding_for_model("gpt-4")

    def num_tokens_from_messages(messages):
        return sum(len(encoding.encode(m["content"])) for m in messages)

    while num_tokens_from_messages(messages) > 7000 and len(messages) > 2:
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
        ai_reply = f"Error contacting OpenAI: {e}"

    return jsonify({'response': ai_reply})
