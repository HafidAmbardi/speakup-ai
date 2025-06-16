import pickle
import torch
import os

print("Starting model preloading script...")

# Define model files - now loading directly from local files
model_files = [
    "model_grammarly.pk", 
    "tokenizer_data.pk", 
    "tokenizer_grammarly.pk"
]

# Check if files exist
for file_name in model_files:
    if os.path.exists(file_name):
        print(f"✓ Found {file_name}")
    else:
        print(f"✗ Missing {file_name}")
        raise FileNotFoundError(f"Required model file {file_name} not found")

# Load the models
print("Loading tokenizer_data.pk...")
with open("tokenizer_data.pk", "rb") as f:
    tokenizer_data = pickle.load(f)

print("Loading tokenizer_grammarly.pk...")
with open("tokenizer_grammarly.pk", "rb") as f:
    tokenizer_grammarly = pickle.load(f)

print("Loading whisper model from package...")
import whisper
whisper_model = whisper.load_model("small")
print("Successfully loaded whisper model from package")

print("Loading model_grammarly.pk...")
with open("model_grammarly.pk", "rb") as f:
    model_grammarly = pickle.load(f)
    
# Save in optimized format
print("Saving optimized models...")
torch.save(tokenizer_data, "tokenizer_data_preloaded.pt", _use_new_zipfile_serialization=True)
torch.save(tokenizer_grammarly, "tokenizer_grammarly_preloaded.pt", _use_new_zipfile_serialization=True)
torch.save(model_grammarly, "model_grammarly_preloaded.pt", _use_new_zipfile_serialization=True)
torch.save(whisper_model, "whisper_model_preloaded.pt", _use_new_zipfile_serialization=True)

print("All models preloaded and saved successfully!")

# Add verification at the end
print("Verifying saved models...")
try:
    test_load = torch.load("tokenizer_data_preloaded.pt", weights_only=False)
    print("✓ tokenizer_data_preloaded.pt verified")
except Exception as e:
    print(f"✗ Error verifying tokenizer_data_preloaded.pt: {e}")

try:
    test_load = torch.load("tokenizer_grammarly_preloaded.pt", weights_only=False)
    print("✓ tokenizer_grammarly_preloaded.pt verified")
except Exception as e:
    print(f"✗ Error verifying tokenizer_grammarly_preloaded.pt: {e}")

try:
    test_load = torch.load("model_grammarly_preloaded.pt", weights_only=False)
    print("✓ model_grammarly_preloaded.pt verified")
except Exception as e:
    print(f"✗ Error verifying model_grammarly_preloaded.pt: {e}")

try:
    test_load = torch.load("whisper_model_preloaded.pt", weights_only=False)
    print("✓ whisper_model_preloaded.pt verified")
except Exception as e:
    print(f"✗ Error verifying whisper_model_preloaded.pt: {e}")