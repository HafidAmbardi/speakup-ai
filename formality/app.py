import os
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Set environment variables
os.environ["TRANSFORMERS_CACHE"] = "/tmp"
os.environ["HF_HOME"] = "/tmp"

app = Flask(__name__)

# Load model and tokenizer once at startup
MODEL_NAME = "s-nlp/roberta-base-formality-ranker"
tokenizer = None
model = None

def load_model():
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()  # Set model to evaluation mode

# Load model at startup
load_model()

# Define formality range responses
FORMALITY_RESPONSES = {
    "very_informal": "Your speech formality score is {}/100, quite informal! Your tone is very casual, which could be great for personal conversations with friends or informal settings. However, in professional or formal situations, you may want to avoid using overly relaxed language. Consider using more respectful and polished language when addressing colleagues, clients, or superiors in business settings.",
    "informal": "Your speech formality score is {}/100, a bit informal but showing improvement! While your tone is still casual, it's better suited for friendly exchanges or informal conversations. However, for interviews, meetings, or any formal discussion, it's a good idea to increase the level of professionalism in your language. Using more polite phrases and professional vocabulary can enhance your communication in these contexts.",
    "balanced": "Your speech formality score is {}/100, a balanced mix of casual and formal! You're in a comfortable middle ground, which works well for most social interactions and some professional settings. It's appropriate for emails or casual discussions with colleagues. However, if you're presenting to a high-level audience or in formal meetings, consider polishing your language slightly by avoiding colloquialisms and being more direct in your communication.",
    "formal": "Your speech formality score is {}/100, mostly formal with a hint of informality! This tone is well-suited for professional emails, presentations, and discussions in business meetings. Your language shows respect and professionalism while still remaining approachable. For more formal settings like conferences or addressing senior executives, you may want to avoid casual phrases and adopt more sophisticated language to leave a stronger impression.",
    "very_formal": "Your speech formality score is {}/100, highly formal and polished! Your tone is perfect for formal meetings, academic presentations, or professional interactions where respect and authority are key. It's appropriate for official correspondence, addressing clients or superiors, and speaking at conferences. Keep in mind that, in more casual settings or among peers, this level of formality might feel a bit too stiff. Tailor your approach depending on your audience, but you're definitely on track for professional success."
}

def get_formality_category(formal_percent):
    """Determine formality category based on percentage"""
    if formal_percent <= 20:
        return "very_informal"
    elif formal_percent <= 40:
        return "informal"
    elif formal_percent <= 60:
        return "balanced"
    elif formal_percent <= 80:
        return "formal"
    else:
        return "very_formal"

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Formality Classifier API is running! Use /predict to classify text."})

@app.route("/predict", methods=["POST"])
def predict_formality():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Text input is required"}), 400

        text = data["text"].strip()
        if not text:
            return jsonify({"error": "Text cannot be empty"}), 400

        # Tokenize input
        encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

        with torch.no_grad():
            logits = model(**encoding).logits
        score = logits.softmax(dim=1)[:, 1].item()

        formal_percent = round(score * 100)
        informal_percent = 100 - formal_percent
        
        category = get_formality_category(formal_percent)
        detailed_response = FORMALITY_RESPONSES[category].format(formal_percent)

        return jsonify({
            "formality_score": round(score, 3),
            "formal_percent": formal_percent,
            "informal_percent": informal_percent,
            "classification": detailed_response
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
