# from flask import Flask, request, render_template, jsonify
# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.preprocessing import image

# app = Flask(__name__)

# # Load the trained model
# MODEL_PATH = r'C:\Users\chada\OneDrive\Desktop\project(tensorflow)\plant_model.h5'
# model = tf.keras.models.load_model(MODEL_PATH)

# # Class labels
# CLASS_LABELS = ['Tulasi', 'Neem', 'AloeVera', 'Amla']

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'})
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No file selected'})

#     # Preprocess the image
#     img = image.load_img(file, target_size=(150, 150))
#     img_array = image.img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     # Make prediction
#     prediction = model.predict(img_array)
#     predicted_class = CLASS_LABELS[np.argmax(prediction)]
#     confidence = np.max(prediction)

#     return jsonify({'class': predicted_class, 'confidence': f'{confidence:.2f}'})

# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, request, render_template
# from keras.preprocessing import image
# import numpy as np
# import os

# app = Flask(__name__)

# # Load your pre-trained model
# model = keras.models.load_model(r'C:\Users\chada\OneDrive\Desktop\project(tensorflow)\model\plant_model.h5')

# # Class labels for prediction
# classes = ['Tulasi', 'Neem', 'AloeVera', 'Amla']

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return "No file part"
#     file = request.files['file']
#     if file.filename == '':
#         return "No selected file"
#     if file:
#         # Save the file to a temporary location
#         filename = secure_filename(file.filename)
#         filepath = os.path.join('static', filename)  # Save in 'static' folder
#         file.save(filepath)
        
#         # Preprocess the image
#         img = image.load_img(filepath, target_size=(150, 150))
#         img_array = image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array /= 255.0

#         # Predict using the model
#         prediction = model.predict(img_array)
#         predicted_class = classes[np.argmax(prediction)]

#         # Delete the file after prediction to save space
#         os.remove(filepath)

#         # Return result
#         return render_template('index.html', prediction=predicted_class, uploaded_image=filepath)
#     return "Something went wrong"

# if __name__ == '__main__':
#     app.run(debug=True)


#
# from flask import Flask, request, render_template
# from werkzeug.utils import secure_filename
# from keras.preprocessing import image
# import numpy as np
# import os
# import tensorflow as tf  # Add this line to import TensorFlow

# # Initialize Flask application
# app = Flask(__name__)

# # Load the pre-trained model
# model_path = r'C:\Users\lala deepika\Desktop\project_tensorflow\model\plant_model_ann.h5'
# model = tf.keras.models.load_model(model_path)  # Use tf.keras instead of keras

# # Class labels for prediction
# classes = ['AloVera', 'Amla', 'Neem', 'Tulasi']

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return "No file part"
#     file = request.files['file']
#     if file.filename == '':
#         return "No selected file"
#     if file:
#         # Secure the filename
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(r'C:\Users\lala deepika\Desktop\project_tensorflow\static', filename)  # Save in 'static' folder
#         file.save(filepath)
        
#         # Preprocess the image
#         img = image.load_img(filepath, target_size=(128, 128))
#         img_array = image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array /= 255.0

#         # Predict using the model
#         prediction = model.predict(img_array)
#         predicted_class = classes[np.argmax(prediction)]

#         # Delete the file after prediction to save space
#         #os.remove(filepath)

#         # Return result
#         return render_template('index.html', prediction=predicted_class, uploaded_image=filepath)
#     return "Something went wrong"

# if __name__ == '__main__':
#     app.run(debug=True)


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
    return render_template('index_tensor.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
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

    return render_template('index_tensor.html', prediction=predicted_class, confidence=f"{confidence:.2f}")

if __name__ == '__main__':
    app.run(debug=True)

