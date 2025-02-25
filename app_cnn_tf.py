from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
import numpy as np
import os

app = Flask(__name__,template_folder=r"C:\Vaishanvi\college\PS\codes\data_set\Indian_Medicinal_Plants\CNN")

# Load the trained model
MODEL_PATH = r"C:\Vaishanvi\college\PS\codes\data_set\Indian_Medicinal_Plants\CNN\CNN_tensorflow.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels
CLASS_LABELS = ['Aloevera', 'Amla', 'Brahmi', 'Neem', 'Tulasi']

# Preprocess the input image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(64, 64))  # Resize to 64x64
    img_array = img_to_array(img) / 255.0            # Normalize to [0, 1]
    #img_array = img_array.reshape(1, 64, 64, 3)
    #       # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save and preprocess the image
    filename = secure_filename(file.filename)
  # Save in 'static' folder
    static_folder=r'C:\Vaishanvi\college\PS\codes\data_set\Indian_Medicinal_Plants\CNN\static'
    filepath = os.path.join(static_folder, filename)
    file.save(filepath)
    img_array = preprocess_image(filepath)

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = CLASS_LABELS[np.argmax(predictions)]
    confidence = np.max(predictions)
    image_url = f"/static/{filename}"
        # Delete the file after prediction to save space
        #os.remove(filepath)

    return jsonify({"plantClass": f"amla,{image_url}"})

if __name__ == '__main__':
    app.run(debug=True)

