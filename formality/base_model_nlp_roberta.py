from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "s-nlp/roberta-base-formality-ranker"

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Use the model for inference, for example:
text = "This is a test sentence."
inputs = tokenizer(text, return_tensors="pt")

# Make a prediction
outputs = model(**inputs)
print(outputs)

model.save_pretrained("D:/Programming/aten/roberta-formality-ranker")
tokenizer.save_pretrained("D:/Programming/aten/roberta-formality-ranker")
