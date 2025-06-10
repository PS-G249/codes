IDENTIFICATION OF INDIAN MEDICINAL PLANTS USING CNN

📌 Overview
Plant identification is crucial in agriculture, herbal medicine, and biodiversity conservation. Traditional AI-based systems rely on datasets like Flavia, Folio, and PlantVillage, which feature isolated leaves on uniform backgrounds. However, these controlled datasets do not reflect real-world conditions.

Our project aims to bridge this gap by using a dataset of whole plant images taken in natural settings with diverse backgrounds such as different soil textures, vehicles, and other environmental elements. This approach enhances the model’s applicability in practical scenarios.

📂 Dataset
Our dataset consists of plant images captured in a nursery environment, categorized into five classes:

🌱 Aloe Vera
🍈 Amla
🌿 Brahmi
🍃 Neem
🌿 Tulasi

Data Augmentation
To increase the dataset size, we applied data augmentation:
✅ Original dataset: 425 images
✅ Augmented dataset: 1,275 images (3× increase)

Rotation: Each image was rotated 90° clockwise
Flipping: Each image was flipped vertically

Final Split:
Training images: 1,020
Testing images: 255

🛠️ Model Development
We implemented CNN and ANN models using only NumPy, TensorFlow & PyTorch, experimenting with different configurations:

Hyperparameter Variations
Activation Functions: Tanh, ReLU, LeakyReLU
Optimization Techniques: Vanilla Gradient Descent, Mini-Batch Gradient Descent, Adam Optimizer
Epochs: 10, 15
Learning Rates: 0.005, 0.01
We ensured identical architectures for both ANN and CNN models to facilitate fair comparison.

🏆 Best Performing Model
📌 CNN (PyTorch) with LeakyReLU, Adam Optimizer, 10 epochs, and a 0.001 learning rate achieved 90% accuracy—the highest among all tested configurations.

🚀 Features
✅ Real-world dataset with varied backgrounds
✅ Comparison of ANN and CNN models
✅ Implemented in PyTorch & TensorFlow
✅ Interactive UI using Flask for real-time plant classification

🔧 Tech Stack
Python Libraries: NumPy, OpenCV, TensorFlow, PyTorch, Pillow, Matplotlib
Web Framework: Flask (for backend and UI linking)
Frontend: HTML, JavaScript

 Conclusion
This project provides a realistic plant identification system by leveraging a custom dataset with diverse backgrounds and optimizing CNN performance. The findings highlight the impact of data representation, augmentation, and hyperparameter tuning on model accuracy.
