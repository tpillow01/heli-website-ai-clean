import json
import difflib
import tiktoken
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from ai_logic import generate_forklift_context

app = Flask(__name__)
client = OpenAI()  # uses OPENAI_API_KEY env

# ─── Load accounts.json ──────────────────────────────────────────────────
with open("accounts.json", "r", encoding="utf-8") as f:
    account_data = json.load(f)
print(f"✅ Loaded {len(account_data)} accounts")

# ─── Load models.json ───────────────────────────────────────────────────
with open("models.json", "r", encoding="utf-8") as f:
    model_data = json.load(f)
print(f"✅ Loaded {len(model_data)} models")

# Conversation history (if you choose to use it later)
conversation_history = []


def find_account_by_name(text: str):
    """
    Return the first account whose name appears as substring,
    else fuzzy‐match.
    """
    lower_text = text.lower()
    for acct in account_data:
        if acct["Account Name"].lower() in lower_text:
            return acct
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
    user_q = data.get('question', '').strip()
    if not user_q:
        return jsonify({'response': 'Please describe the customer’s needs.'}), 400

    # 1) Detect company
    acct = find_account_by_name(user_q)
    cust_name = acct["Account Name"] if acct else ""
    print(f"🔍 find_account_by_name: matched = '{cust_name}'")

    # 2) Build prompt context
    prompt_context = generate_forklift_context(user_q, cust_name, model_data)
    print("=== PROMPT CONTEXT ===")
    print(prompt_context)
    print("======================")

    # 3) (Optional) track history
    conversation_history.append({"role": "user", "content": user_q})
    if len(conversation_history) > 4:
        conversation_history.pop(0)

    # 4) Combined system prompt
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
            "Attachments:, Comparison:, Sales Pitch Techniques:, "
            "Common Objections:.\n"
            "List details underneath using hyphens and indent subpoints. "
            "Leave a blank line between sections.\n\n"
            "At the end, include:\n"
            "- <span class=\"section-label\">Sales Pitch Techniques:</span> 1–2 persuasive points.\n"
            "- <span class=\"section-label\">Common Objections:</span> 1–2 common concerns and how to address them.\n\n"
            "Example:\n"
            "<span class=\"section-label\">Customer Profile:</span>\n"
            "- Company: Acme Corp\n"
            "- Industry: Retail\n"
            "- SIC Code: 5311\n\n"
            "<span class=\"section-label\">Model:</span>\n"
            "- Heli H2000 Series 5-7T\n\n"
            "<span class=\"section-label\">Power:</span>\n"
            "- Diesel\n\n"
            "<span class=\"section-label\">Sales Pitch Techniques:</span>\n"
            "- Emphasize reliability.\n\n"
            "<span class=\"section-label\">Common Objections:</span>\n"
            "- \"Why not Toyota?\"\n"
            "  → Heli offers faster part availability.\n"
        )
    }

    # 5) Assemble messages
    messages = [
        system_prompt,
        {"role": "user", "content": prompt_context}
    ]
    # You may later append conversation_history here if needed.

    # 6) Token‐limit guard
    encoding = tiktoken.encoding_for_model("gpt-4")
    def count_tokens(msgs):
        return sum(len(encoding.encode(m["content"])) for m in msgs)
    while count_tokens(messages) > 7000 and len(messages) > 2:
        messages.pop(1)

    # 7) Call OpenAI
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
