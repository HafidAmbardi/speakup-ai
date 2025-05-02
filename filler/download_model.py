from transformers import AutoModel, AutoProcessor
import os

model_id = "nyrahealth/CrisperWhisper"
target_dir = os.path.join("C:", "Users", "Hafid", "Desktop", "Git", "speakup-ai", "filler")

model = AutoModel.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

model.save_pretrained(target_dir)
processor.save_pretrained(target_dir)