from flask import Flask, request, render_template,jsonify
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms
from PIL import Image
import os

# Parameters
model_path = r"C:\Vaishanvi\college\PS\codes\data_set\Indian_Medicinal_Plants\CNN\CNN_pytorch_v1.pth"
class_names = ['AloeVera','Amla', 'Brahmi', 'Neem', 'Tulasi']

# Load model
class PlantCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(PlantCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(64 * 16 * 16, 64)
        self.fc2 = torch.nn.Linear(64, num_classes)
        self.LeakyReLU = torch.nn.LeakyReLU()

    def forward(self, x):
        x = self.pool(self.LeakyReLU(self.conv1(x)))
        x = self.pool(self.LeakyReLU(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.LeakyReLU(self.fc1(x))
        x = self.fc2(x)
        return x

model = PlantCNN(num_classes=len(class_names))
model.load_state_dict(torch.load(model_path))
model.eval()

# Flask app
app = Flask(__name__,template_folder=r"C:\Vaishanvi\college\PS\codes\data_set\Indian_Medicinal_Plants\CNN")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

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
        
        image = Image.open(file).convert('RGB')
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            label = class_names[predicted.item()]

        # Predict using the model

        image_url = f"/static/{filename}"
        # Delete the file after prediction to save space
        #os.remove(filepath)

        return jsonify({"plantClass": f"{label},{image_url}"})
    return "Something went wrong"

if __name__ == '__main__':
    app.run(debug=True)
