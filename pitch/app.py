from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import os
import tempfile
from werkzeug.utils import secure_filename
import atexit
import shutil
from scipy import stats

app = Flask(__name__)
CORS(app)

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def remove_noise(y, sr):
    # Compute the spectrogram
    D = librosa.stft(y)
    
    # Compute the magnitude spectrogram
    magnitude = np.abs(D)
    
    # Compute the phase spectrogram
    phase = np.angle(D)
    
    # Compute the median filter along the time axis
    median_filtered = librosa.decompose.nn_filter(magnitude,
                                                aggregate=np.median,
                                                metric='cosine')
    
    # Reconstruct the denoised spectrogram
    D_denoised = median_filtered * np.exp(1j * phase)
    
    # Convert back to time domain
    y_denoised = librosa.istft(D_denoised)
    
    return y_denoised

def calculate_silent_ratio(y, sr, threshold_db=-40):
    # Compute RMS energy
    rms = librosa.feature.rms(y=y)[0]
    
    # Convert to dB
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    
    # Calculate silent ratio
    silent_frames = np.sum(rms_db < threshold_db)
    total_frames = len(rms_db)
    silent_ratio = silent_frames / total_frames
    
    return silent_ratio

def analyze_pitch_fluctuation(pitches):
    # Remove zero and negative pitches
    valid_pitches = pitches[pitches > 0]
    
    if len(valid_pitches) < 2:
        return {
            "is_monotone": True,
            "fluctuation_score": 0.0,
            "pitch_range": 0.0
        }
    
    # Calculate pitch range
    pitch_range = np.max(valid_pitches) - np.min(valid_pitches)
    
    # Calculate standard deviation of pitch changes
    pitch_changes = np.diff(valid_pitches)
    std_dev = np.std(pitch_changes)
    
    # Calculate fluctuation score (normalized between 0 and 1)
    max_expected_std = 100  # This is a heuristic value
    fluctuation_score = min(std_dev / max_expected_std, 1.0)
    
    # Determine if speech is monotone (threshold is arbitrary)
    is_monotone = fluctuation_score < 0.2
    
    return {
        "is_monotone": bool(is_monotone),
        "fluctuation_score": float(fluctuation_score),
        "pitch_range": float(pitch_range)
    }

def analyze_pitch(audio_path):
    # Load audio file
    y, sr = librosa.load(audio_path)
    
    # Remove background noise
    y_denoised = remove_noise(y, sr)
    
    # Calculate silent ratio
    silent_ratio = calculate_silent_ratio(y_denoised, sr)
    
    # Extract pitch using librosa's pitch tracking
    pitches, magnitudes = librosa.piptrack(y=y_denoised, sr=sr)
    
    # Get the most prominent pitch at each time frame
    pitch_times = []
    all_pitches = []
    frame_times = librosa.frames_to_time(np.arange(pitches.shape[1]), sr=sr)
    
    # Process at 0.5 second intervals
    interval = 0.5  # half second intervals
    current_time = 0.0
    duration = librosa.get_duration(y=y_denoised, sr=sr)
    
    while current_time < duration:
        # Find the closest frame to our target time
        frame_idx = np.argmin(np.abs(frame_times - current_time))
        
        # Get the pitch at this frame
        index = magnitudes[:, frame_idx].argmax()
        pitch = pitches[index, frame_idx]
        
        if pitch > 0:  # Only include valid pitches
            pitch_times.append({
                "pitch": float(pitch),
                "timestamp": float(current_time)
            })
            all_pitches.append(float(pitch))
        
        current_time += interval
    
    # Analyze pitch fluctuation
    pitch_analysis = analyze_pitch_fluctuation(np.array(all_pitches))
    
    return {
        "pitchFluctuation": pitch_times,
        "silentRatio": float(silent_ratio),
        "pitchAnalysis": pitch_analysis
    }

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, secure_filename(file.filename))
    
    try:
        # Save the uploaded file
        file.save(temp_file_path)
        
        # Analyze the audio
        analysis_results = analyze_pitch(temp_file_path)
        return jsonify(analysis_results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up the temporary directory and its contents
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"Warning: Could not remove temporary directory: {e}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port) 