import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


# Parameters
data_dir = r"C:\Vaishanvi\college\PS\codes\data_set\Indian_Medicinal_Plants\CNN\increased_dataset_split\increased_dataset_split"  # Dataset directory
save_model_path = r"C:\Vaishanvi\college\PS\codes\data_set\Indian_Medicinal_Plants\CNN\CNN_pytorch_test.pth"  # Save model path
num_classes = 5
epochs = 10
batch_size = 24
learning_rate = 0.001

# Data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalization
])

# Load datasets
train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Define CNN model
class PlantCNN(nn.Module):
    def __init__(self, num_classes):
        super(PlantCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.LeakyReLU= nn.LeakyReLU()

    def forward(self, x):
        x = self.pool(self.LeakyReLU(self.conv1(x)))
        x = self.pool(self.LeakyReLU(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.LeakyReLU(self.fc1(x))
        x = self.fc2(x)
        return x
    
    '''def __init__(self, num_classes):
        super(PlantCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64,8,kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.fc1 = nn.Linear(8 * 8 * 8, 64)
        self.fc2 = nn.Linear(8*8*8, num_classes)
        self.LeakyReLU= nn.LeakyReLU()

    def forward(self, x):
        x = self.pool(self.LeakyReLU(self.conv1(x)))
        x = self.pool(self.LeakyReLU(self.conv2(x)))
        x = self.pool(self.LeakyReLU(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        #x = self.LeakyReLU(self.fc1(x))
        x = self.fc2(x)
        return x'''

# Initialize model, loss, and optimizer
torch.manual_seed(42)
model = PlantCNN(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []


# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(correct_train / total_train)

    # Validation
    model.eval()
    running_test_loss = 0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_test += (predicted == labels).sum().item()
            total_test += labels.size(0)

    test_losses.append(running_test_loss / len(test_loader))
    test_accuracies.append(correct_test / total_test)

    print(f"Epoch {epoch + 1}/{epochs}, "
          f"Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, "
          f"Train Accuracy: {train_accuracies[-1]:.4f}, Test Accuracy: {test_accuracies[-1]:.4f}")

    
    # Evaluation loop with confusion matrix
all_preds = []
all_labels = []

model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        
        # Collect predictions and labels for confusion matrix
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Convert to numpy arrays
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Generate confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:\n", cm)


# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.classes)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(r"C:\Vaishanvi\college\PS\codes\data_set\Indian_Medicinal_Plants\CNN\confusion_matrix_test.png",dpi=300, bbox_inches="tight")
plt.show()

# Classification report
report = classification_report(all_labels, all_preds, target_names=train_dataset.classes)
print("\nClassification Report:\n", report)

def calculate_metrics(cm, all_labels, all_preds, num_classes):
    metrics = {}
    for i in range(num_classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FN + FP)
        
        accuracy = (TP + TN) / cm.sum()
        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
        
        # Calculate AUC for each class
        bin_labels = (all_labels == i).astype(int)  # Binary labels for class `i`
        bin_preds = (all_preds == i).astype(int)    # Binary predictions for class `i`
        auc = roc_auc_score(bin_labels, bin_preds) if len(set(bin_labels)) > 1 else 0
        
        metrics[f"Class {i}"] = {
            "Accuracy": accuracy,
            "Specificity": specificity,
            "AUC": auc,
        }
    return metrics

# Calculate and display additional metrics
metrics = calculate_metrics(cm, all_labels, all_preds, num_classes)
print("\nMetrics for Each Class:")
for class_name, values in metrics.items():
    print(f"{class_name}: {values}")


metrics_save_path = "performance_metrics_test.txt"
with open(metrics_save_path, "w") as f:
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(cm))
    f.write("\n\nClassification Report:\n")
    f.write(report)
print(f"Performance metrics saved to {metrics_save_path}")

# Plot Loss Graph
plt.plot(range(1, epochs + 1), train_losses, label="train", color="blue")
plt.plot(range(1, epochs + 1), test_losses, label="test", color="orange")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Over Epochs")
plt.legend()  # Add legend
loss_plot_path = "loss_graph_v1.png"
plt.show()
plt.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Loss graph saved to {loss_plot_path}")

# Plot Accuracy Graph
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs + 1), train_accuracies, label="train", color="blue")
plt.plot(range(1, epochs + 1), test_accuracies, label="test", color="orange")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy Over Epochs")
plt.legend()  # Add legend
plt.show()
accuracy_plot_path = "accuracy_graph_v1.png"
plt.savefig(accuracy_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Accuracy graph saved to {accuracy_plot_path}")


# Save the model
os.makedirs(os.path.dirname(save_model_path), exist_ok=True)  # Ensure directory exists
torch.save(model.state_dict(), save_model_path)
print(f"Model saved as {save_model_path}")
