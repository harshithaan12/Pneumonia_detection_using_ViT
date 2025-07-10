import os
from flask import Flask, render_template, request
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

app = Flask(__name__)

# Use raw string or forward slashes for Windows paths
model_dir = r"C:\Users\Harshitha\Downloads\pneumonia_webapp\pneumonia_vit_model"

# Load model & processor
model = ViTForImageClassification.from_pretrained(model_dir, local_files_only=True)
processor = ViTImageProcessor.from_pretrained(model_dir, local_files_only=True)
model.eval()

# **Define your own class names in the correct order**:
class_names = ["NORMAL", "PNEUMONIA"]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files.get('image')
    if not file:
        return "No image uploaded", 400

    # Load & preprocess
    img = Image.open(file.stream).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")

    # Inference
    with torch.no_grad():
        logits = model(**inputs).logits
        pred_idx = logits.argmax(-1).item()

    # Map index → human label
    predicted_label = class_names[pred_idx]

    return render_template('result.html', prediction=predicted_label)

if __name__ == '__main__':
    app.run(debug=True)
