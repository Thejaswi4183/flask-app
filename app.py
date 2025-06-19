from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from datetime import datetime
import os
import gdown
from dotenv import load_dotenv

# Initialize Flask app
app = Flask(__name__)

# Load environment variables from .env
load_dotenv()

# File IDs from Google Drive (Render reads from environment directly)
KERAS_FILE_ID = os.getenv("KERAS_FILE_ID")
PKL_FILE_ID = os.getenv("PKL_FILE_ID")

# Model-related global vars
model = None
tokenizer = None
max_len = 100

# Download function
def download_file_if_missing(file_id, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", filename, quiet=False)
        print(f"{filename} downloaded.")

# Load model and tokenizer
def load_model_and_tokenizer():
    global model, tokenizer
    download_file_if_missing(KERAS_FILE_ID, "final_multimodal_model.keras")
    download_file_if_missing(PKL_FILE_ID, "text_tokenizer.pkl")

    print("Loading model and tokenizer...")
    model = tf.keras.models.load_model("final_multimodal_model.keras")
    with open("text_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    print("Model and tokenizer loaded.")

# Preprocessing functions
def preprocess_image(image_file):
    file_bytes = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224)) / 255.0
    return np.expand_dims(img, axis=0)

def preprocess_text(text):
    seq = tokenizer.texts_to_sequences([text])
    return pad_sequences(seq, maxlen=max_len, padding="post")

# Health check
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "API running"}), 200

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    try:
        file = request.files["file"]
        text = request.form.get("text", "")

        image_input = preprocess_image(file)
        text_input = preprocess_text(text)

        age_pred = model.predict([image_input, text_input])[0][0]
        current_year = datetime.now().year
        age = current_year - age_pred

        return jsonify({"age": f"{float(age):.2f}"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Load model only once when server starts
with app.app_context():
    load_model_and_tokenizer()

# Entry point for gunicorn
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
