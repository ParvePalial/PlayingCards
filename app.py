# server/app.py
import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import joblib
import cv2
import numpy as np
from skimage.feature import hog
from flask_cors import CORS

MODEL_PATH = "card_classifier.joblib"  # path to your saved model
IMG_SIZE = (200, 200)                   # must match training IMG_SIZE

app = Flask(__name__)
CORS(app)  # allow cross-origin requests from the RN app (adjust in production)

# Load model once
clf, encoder = joblib.load(MODEL_PATH)
print("Model loaded:", MODEL_PATH)

def preprocess_image_bytes(file_bytes):
    # Convert bytes -> numpy image (cv2)
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image")
    img = cv2.resize(img, IMG_SIZE)
    # Extract HOG features (same parameters as training)
    features = hog(img,
                   orientations=9,
                   pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2),
                   block_norm='L2-Hys')
    return features

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "no file part"}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({"error": "no selected file"}), 400

    try:
        data = f.read()
        features = preprocess_image_bytes(data)
        pred_idx = clf.predict([features])[0]
        pred_label = encoder.inverse_transform([pred_idx])[0]

        # Optional: get confidence/probability if clf supports it
        prob = None
        if hasattr(clf, "predict_proba"):
            prob = float(np.max(clf.predict_proba([features])))

        return jsonify({
            "label": str(pred_label),
            "probability": prob
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
