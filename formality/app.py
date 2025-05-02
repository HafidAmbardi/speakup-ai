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

        # Predict formality score
        with torch.no_grad():
            logits = model(**encoding).logits
        score = logits.softmax(dim=1)[:, 1].item()

        # Convert score to formality classification
        formal_percent = round(score * 100)
        informal_percent = 100 - formal_percent

        return jsonify({
            "formality_score": round(score, 3),
            "formal_percent": formal_percent,
            "informal_percent": informal_percent,
            "classification": f"Your speech is {formal_percent}% formal and {informal_percent}% informal."
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
