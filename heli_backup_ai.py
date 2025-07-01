import os
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from ai_logic import generate_forklift_context

app = Flask(__name__)

# Load API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Store conversation history
conversation_history = []

@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_question = data.get('question', '').strip()

    if not user_question:
        return jsonify({'response': 'Please enter a description of the customer’s needs.'})

    filtered_context = generate_forklift_context(user_question)

    if len(conversation_history) > 10:
        conversation_history.pop(0)

    conversation_history.append({"role": "user", "content": user_question})

    # System prompt (always first)
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
            "Example:\n"
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

    messages = [system_prompt] + conversation_history

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

if __name__ == "__main__":
    app.run(debug=True, port=5004)
