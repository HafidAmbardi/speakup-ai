from flask import Flask, request, jsonify
import whisper
import tempfile
import os
import re

# Set cache dir so Hugging Face Space has write access
os.environ["XDG_CACHE_HOME"] = "/tmp"

app = Flask(__name__)
model = whisper.load_model("tiny") 

# List of common filler words to detect
FILLER_WORDS = [
    "um", "umm", "hm", "uh", "ah", "er", "hmm", "hmmm" "like", "you know", "well"
]

def is_filler_word(word):
    word = word.lower().strip('.,!?')
    return word in FILLER_WORDS

def format_word(word):
    # Remove punctuation and check if it's a filler word
    clean_word = word.strip('.,!?')
    if is_filler_word(clean_word):
        return f"[{clean_word.upper()}]"
    return clean_word

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    audio_file = request.files['file']
    if audio_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        audio_file.save(tmp)
        tmp_path = tmp.name

    # Create initial prompt with filler words
    initial_prompt = "umm, uh, ah, er, hm, hmm, um, hmmm, like, well"
    
    result = model.transcribe(
        tmp_path,
        initial_prompt=initial_prompt
    )
    
    # Process each segment and split into words
    chunks = []
    for segment in result["segments"]:
        words = segment["text"].strip().split()
        if not words:
            continue
            
        # Calculate time per word
        segment_duration = segment["end"] - segment["start"]
        time_per_word = segment_duration / len(words)
        
        # Create word chunks with timestamps
        for i, word in enumerate(words):
            start_time = segment["start"] + (i * time_per_word)
            end_time = start_time + time_per_word
            
            chunks.append({
                "text": format_word(word),
                "timestamp": [round(start_time, 2), round(end_time, 2)]
            })
    
    # Clean up the temporary file
    os.unlink(tmp_path)
    
    return jsonify({"chunks": chunks})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)