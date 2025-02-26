from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename

# Initialize Flask app 
app = Flask(__name__,template_folder=r"C:\Vaishanvi\college\PS\codes\data_set\Indian_Medicinal_Plants\ANN") #Flask is a web framework used to create web applications
model_path = r"C:\Vaishanvi\college\PS\codes\data_set\Indian_Medicinal_Plants\ANN\indian_medicinal_plants_model_sgd_optimized.h5"
model = tf.keras.models.load_model(model_path)
classes = ["AloVera", "Amla", "Brahmi","Neem", "Tulasi"]

img_height, img_width = 64, 64
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(64, 64))  # Resize to 64x64
    img_array = img_to_array(img) / 255.0            # Normalize to [0, 1]
    #img_array = img_array.reshape(1, 64, 64, 3)
    #       # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')#initial basic html files 
def index():
    return render_template("index.html")#render_template means loads and renders HTML files

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}),400 
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}),400
        
    filename = secure_filename(file.filename)
    static_folder=r'C:\Vaishanvi\college\PS\codes\data_set\Indian_Medicinal_Plants\ANN\static'
    filepath = os.path.join(static_folder, filename)  # Save in 'static' folder
    file.save(filepath)
    # Predict
    img_array = preprocess_image(filepath)
    predictions = model.predict(img_array)

    # Log the raw prediction values to the console
    #print(f"Raw predictions: {predictions}")

    predicted_class = classes[np.argmax(predictions)]
    image_url = f"/static/{filename}"
    confidence = np.max(predictions)

    print(f"Predicted Class: {predicted_class}, Confidence: {confidence * 100:.2f}%")
    return jsonify({"plantClass": f"{predicted_class},{image_url}"})
    #return jsonify({"predicted_class": predicted_class, "confidence": f"{confidence * 100:.2f}%"})

if __name__ == '__main__':
    app.run(debug=True)
