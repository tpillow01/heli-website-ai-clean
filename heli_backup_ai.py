import difflib
import tiktoken
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
import json
from ai_logic import generate_forklift_context

app = Flask(__name__)
client = OpenAI()  # reads OPENAI_API_KEY from env

# Load JSON just once
with open("accounts.json", "r", encoding="utf-8") as f:
    account_data = json.load(f)

@app.route('/')
def home():
    return render_template('chat.html')

def find_account_by_name(text: str):
    """
    Look for a substring match of any account name in `text`,
    otherwise fallback to fuzzy match.
    """
    lower = text.lower()
    for acct in account_data:
        if acct["Account Name"].lower() in lower:
            return acct
    names = [a["Account Name"] for a in account_data]
    match = difflib.get_close_matches(text, names, n=1, cutoff=0.6)
    if match:
        return next(a for a in account_data if a["Account Name"] == match[0])
    return None

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json() or {}
    user_q = data.get('question','').strip()
    if not user_q:
        return jsonify({'response':'Please describe the customer’s needs.'}), 400

    # 1) detect company
    acct = find_account_by_name(user_q)
    cust_name = acct["Account Name"] if acct else ""

    # 2) build context
    prompt_ctx = generate_forklift_context(user_q, cust_name)

    # 3) build system prompt
    system_prompt = {
        "role":"system",
        "content":(
            "You are a helpful, detailed Heli Forklift sales assistant.\n"
            "When providing customer-specific data, wrap it in a "
            "<span class=\"section-label\">Customer Profile:</span> section.\n"
            "When recommending models, wrap section headers in a "
            "<span class=\"section-label\">...</span> tag.\n"
            "Use these sections in order if present:\n"
            "Customer Profile:, Model:, Power:, Capacity:, Tire Type:, "
            "Attachments:, Comparison:, Sales Pitch Techniques:, Common Objections:.\n"
            "List details underneath using hyphens, leave a blank line\n"
            "between sections, and indent subpoints for clarity.\n\n"
            "At the end, include:\n"
            "- <span class=\"section-label\">Sales Pitch Techniques:</span> 1–2 persuasive points.\n"
            "- <span class=\"section-label\">Common Objections:</span> 1–2 common concerns and how to address them.\n\n"
            "Example:\n"
            "<span class=\"section-label\">Customer Profile:</span>\n"
            "- Company: Acme Co\n"
            "- Industry: Retail\n"
            "- SIC Code: 5311\n\n"
            "<span class=\"section-label\">Model:</span>\n"
            "- Heli H2000 Series 5–7T Electric Forklift\n\n"
            "<span class=\"section-label\">Power:</span>\n"
            "- Diesel\n\n"
            "<span class=\"section-label\">Sales Pitch Techniques:</span>\n"
            "- Emphasize reliability.\n\n"
            "<span class=\"section-label\">Common Objections:</span>\n"
            "- \"Why not Toyota?\"\n"
            "  → Heli offers faster part availability.\n"
        )
    }

    # 4) assemble messages
    messages = [
        system_prompt,
        {"role":"user","content":prompt_ctx}
    ]

    # 5) token‐limit guard
    encoding = tiktoken.encoding_for_model("gpt-4")
    def count_tokens(ms): 
        return sum(len(encoding.encode(m["content"])) for m in ms)
    while count_tokens(messages) > 7000 and len(messages) > 2:
        messages.pop(1)

    # 6) call OpenAI
    try:
        resp = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=600,
            temperature=0.7
        )
        ai_reply = resp.choices[0].message.content.strip()
    except Exception as e:
        print("OpenAI error:",e)
        ai_reply = f"❌ Internal error: {e}"

    return jsonify({'response':ai_reply})

if __name__=='__main__':
    app.run(debug=True)
