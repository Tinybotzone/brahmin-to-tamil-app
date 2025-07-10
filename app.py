from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import json

app = Flask(__name__, static_folder='../frontend', static_url_path='')

# Load model and character map
model = tf.keras.models.load_model('keras_model.h5')
with open('char_map.json', 'r', encoding='utf-8') as f:
    char_map = json.load(f)

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    try:
        image = Image.open(file.stream).convert('RGB')
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        prediction = model.predict(image_array)
        predicted_index = int(np.argmax(prediction))
        tamil_character = char_map.get(str(predicted_index), '?')

        return jsonify({'character': tamil_character})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

