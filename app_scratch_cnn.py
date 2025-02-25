from flask import Flask, request, render_template,jsonify
from werkzeug.utils import secure_filename
from keras.preprocessing import image
import numpy as np
import os
#import tensorflow as tf  # Add this line to import TensorFlow
import joblib
import pickle


# Initialize Flask application
app = Flask(__name__,template_folder=r"C:\Vaishanvi\college\PS\codes\data_set\Indian_Medicinal_Plants\CNN")

# Load the pre-trained model
model_path = r"C:\Vaishanvi\college\PS\codes\data_set\Indian_Medicinal_Plants\CNN\final\CNN_final.pkl"
model = joblib.load(model_path)


# Class labels for prediction
classes = ['AloVera', 'Amla','Brahmi', 'Neem', 'Tulasi']

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image part"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        # Secure the filename
        filename = secure_filename(file.filename)
  # Save in 'static' folder
        static_folder=r'C:\Vaishanvi\college\PS\codes\data_set\Indian_Medicinal_Plants\CNN\static'
        filepath = os.path.join(static_folder, filename)
        file.save(filepath)
        
        # Preprocess the image
        img = image.load_img(filepath, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Predict using the model
        num=model.forward(img_array)
        n=np.argmax(num, axis=1)
        predicted_class = ["AloeVera", "Amla","Brahmi", "Neem", "Tulasi"]
        pre = predicted_class[n[0]]

        image_url = f"/static/{filename}"
        # Delete the file after prediction to save space
        #os.remove(filepath)

        return jsonify({"plantClass": f"{pre},{image_url}"})
    return "Something went wrong"

if __name__ == '__main__':
    app.run(debug=True)



