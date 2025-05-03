from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import threading
import time
import platform
import logging
import signal
import sys
import warnings

# Suppress specific tokenizer warning
warnings.filterwarnings("ignore", message=".*The current process just got forked.*")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_DIR = "models"
model_files = {
    "tokenizer_data": os.path.join(MODEL_DIR, "tokenizer_data.pk"),
    "tokenizer_grammarly": os.path.join(MODEL_DIR, "tokenizer_grammarly.pk"),
    "model_grammarly": os.path.join(MODEL_DIR, "model_grammarly.pk"),
    "whisper_model": os.path.join(MODEL_DIR, "whisper_model.pt"),
}

app = Flask(__name__)
CORS(app)

tokenizer_grammarly = None
model_grammarly = None
tokenizer_data = None
model_whisper = None
models_loaded = False
is_loading = False
loading_error = None

loading_lock = threading.Lock()

def signal_handler(signum, frame):
    logger.info("Received shutdown signal")
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)

def load_models():
    global tokenizer_grammarly, model_grammarly, tokenizer_data, model_whisper, models_loaded, is_loading, loading_error

    if is_loading or models_loaded:
        return

    with loading_lock:
        if is_loading or models_loaded:
            return
        is_loading = True

    try:
        import pickle
        import torch
        import whisper
        from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
        from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
        from whisper.model import Whisper

        # Add Whisper to safe globals
        torch.serialization.add_safe_globals([Whisper])

        # Load models directly from local files
        logger.info("Loading models from local files...")
        
        # Load tokenizer and model for grammarly
        with open(model_files["tokenizer_grammarly"], 'rb') as f:
            tokenizer_grammarly = pickle.load(f)
            
        # Load the T5 model from the pickle file
        with open(model_files["model_grammarly"], 'rb') as f:
            model_grammarly = pickle.load(f)
        
        # Load tokenizer for data
        with open(model_files["tokenizer_data"], 'rb') as f:
            tokenizer_data = pickle.load(f)
        
        # Load whisper model
        logger.info("Loading Whisper model...")
        model_whisper = torch.load(model_files["whisper_model"], map_location="cpu", weights_only=False)
        model_whisper = model_whisper.to("cpu")
        
        models_loaded = True
        logger.info("All models loaded successfully")
        
    except Exception as e:
        loading_error = str(e)
        logger.error(f"Error loading models: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        is_loading = False

def get_memory_usage_mb():
    if platform.system() != "Windows":
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    else:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)

def levenshteinRecursive(str1, str2, m, n):
    if m == 0:
        return n
    if n == 0:
        return m
    if str1[m - 1] == str2[n - 1]:
        return levenshteinRecursive(str1, str2, m - 1, n - 1)
    return 1 + min(
        levenshteinRecursive(str1, str2, m, n - 1),
        min(
            levenshteinRecursive(str1, str2, m - 1, n),
            levenshteinRecursive(str1, str2, m - 1, n - 1))
    )

@app.route('/_ah/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint."""
    global models_loaded, is_loading, loading_error

    status = {
        "status": "healthy",
        "models_loaded": models_loaded,
        "is_loading": is_loading,
        "memory_usage_mb": get_memory_usage_mb()
    }

    if loading_error:
        status["loading_error"] = loading_error
        return jsonify(status), 503

    if not models_loaded and not is_loading:
        # Start loading models if not already loading
        thread = threading.Thread(target=load_models)
        thread.daemon = True
        thread.start()
        return jsonify(status), 202  # Accepted but not ready

    return jsonify(status), 200

@app.route('/start-loading', methods=['GET'])
def start_loading():
    global models_loaded, is_loading

    if not models_loaded and not is_loading:
        thread = threading.Thread(target=load_models)
        thread.daemon = True
        thread.start()
        return jsonify({"message": "Model loading started"}), 200
    elif models_loaded:
        return jsonify({"message": "Models already loaded"}), 200
    else:
        return jsonify({"message": "Models are currently loading"}), 200

@app.route('/speech2text', methods=['POST'])
def transcribe():
    global models_loaded, is_loading, model_whisper, model_grammarly, tokenizer_grammarly, tokenizer_data

    if not models_loaded and not is_loading:
        thread = threading.Thread(target=load_models)
        thread.daemon = True
        thread.start()
        return jsonify({"error": "Models are still loading. Please try again in a moment."}), 503

    if not models_loaded:
        return jsonify({"error": "Models are still loading. Please try again in a moment."}), 503

    from pydub import AudioSegment
    import re

    # Get content type from request
    content_type = request.headers.get('Content-Type', '').lower()
    
    # Determine file extension based on content type
    if 'audio/m4a' in content_type:
        input_ext = 'm4a'
    elif 'audio/wav' in content_type:
        input_ext = 'wav'
    elif 'audio/mp3' in content_type:
        input_ext = 'mp3'
    else:
        # Try to detect format from file content if content-type is not specified
        input_ext = 'm4a'  # default to m4a if not specified

    input_file = f"temp_audio.{input_ext}"
    output_file = "temp_audio.wav"
    files_to_cleanup = []

    try:
        if request.data:
            # Save the uploaded file
            with open(input_file, "wb") as f:
                f.write(request.data)
            files_to_cleanup.append(input_file)
            
            # Convert to WAV if needed
            if input_ext != 'wav':
                audio = AudioSegment.from_file(input_file, format=input_ext)
                audio.export(output_file, format="wav")
                files_to_cleanup.append(output_file)
            else:
                output_file = input_file  # Use the WAV file directly
            
            # Transcribe the audio
            result = model_whisper.transcribe(output_file)

            sentences = re.split(r'(?<=[.!?])\s+', result["text"])
            sentence_pairs = []
            total_distance = 0
            corrections_count = 0

            for sentence in sentences:
                if not sentence.strip():
                    continue

                input_text_sample = f"Fix grammatical errors in this sentence: {sentence}"
                input_ids = tokenizer_grammarly(input_text_sample, return_tensors="pt").input_ids
                outputs = model_grammarly.generate(input_ids, max_length=256)
                edited_sentence = tokenizer_grammarly.decode(outputs[0], skip_special_tokens=True)

                tokens_original = tokenizer_data.split(sentence)
                tokens_corrected = tokenizer_data.split(edited_sentence)

                tokens_original = list(filter(lambda a: a != '', tokens_original))
                tokens_corrected = list(filter(lambda a: a != '', tokens_corrected))

                sentence_dist = levenshteinRecursive(tokens_original, tokens_corrected,
                                                len(tokens_original), len(tokens_corrected))

                total_distance += sentence_dist
                if sentence_dist > 0:
                    corrections_count += 1

                sentence_pairs.append({
                    "original": sentence,
                    "corrected": edited_sentence,
                    "distance": sentence_dist
                })

            avg_distance = total_distance / len(sentence_pairs) if sentence_pairs else 0

            output = {
                "sentence_pairs": sentence_pairs,
                "stats": {
                    "average_distance": avg_distance,
                    "sentences_corrected": corrections_count,
                    "total_sentences": len(sentence_pairs)
                }
            }

            return jsonify(output)
            
        else:
            return jsonify({"error": "No audio data received"}), 400
            
    except Exception as e:
        logger.error(f"Error processing audio file: {str(e)}")
        return jsonify({"error": f"Error processing audio file: {str(e)}"}), 400
        
    finally:
        # Clean up any temporary files
        for file_path in files_to_cleanup:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {file_path}: {str(e)}")

@app.route('/detailed-health', methods=['GET'])
def detailed_health_check():
    global models_loaded, is_loading, loading_error, tokenizer_grammarly, model_grammarly, tokenizer_data, model_whisper

    status = {
        "status": "healthy",
        "models_loaded": models_loaded,
        "is_loading": is_loading,
        "memory_usage_mb": get_memory_usage_mb(),
        "model_objects": {
            "tokenizer_data": "loaded" if tokenizer_data is not None else "not loaded",
            "tokenizer_grammarly": "loaded" if tokenizer_grammarly is not None else "not loaded",
            "model_whisper": "loaded" if model_whisper is not None else "not loaded",
            "model_grammarly": "loaded" if model_grammarly is not None else "not loaded"
        }
    }

    if loading_error:
        status["loading_error"] = loading_error

    import sys
    status["python_version"] = sys.version

    try:
        import whisper
        status["whisper_version"] = whisper.__version__
    except:
        status["whisper_version"] = "unknown"

    return jsonify(status), 200

@app.route('/_ah/warmup', methods=['GET'])
def warmup():
    """Handle warmup requests from Cloud Run."""
    if not models_loaded and not is_loading:
        thread = threading.Thread(target=load_models)
        thread.daemon = True
        thread.start()
    return jsonify({"status": "warming up"}), 200

if __name__ == '__main__':
    # Start loading models in a background thread
    thread = threading.Thread(target=load_models)
    thread.daemon = True
    thread.start()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)), debug=False)