from flask import Flask, render_template, request, send_from_directory
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

MODEL_PATH = os.path.join(BASE_DIR, "models", "vegetable_model.h5")
IMG_HEIGHT = 128
IMG_WIDTH = 128

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Class names (from training folders)
train_dir = os.path.join(BASE_DIR, "code", "Vegetable Images", "train")
class_names = sorted(os.listdir(train_dir))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']

    if file:
        filename = file.filename
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(image_path)

        # Preprocess image
        img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions)
        confidence = float(np.max(predictions)) * 100

        label = class_names[class_index]

        return render_template(
            'logout.html',
            label=label,
            confidence=round(confidence, 2),
            image_file=filename
        )

    return "No image uploaded"

if __name__ == '__main__':
    app.run(debug=True)
