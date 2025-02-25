import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
import numpy as np

# Dataset paths
DATASET_PATH = r"C:\Vaishanvi\college\PS\codes\data_set\Indian_Medicinal_Plants\CNN\increased_dataset_split\increased_dataset_split"
TRAIN_PATH = f"{DATASET_PATH}\\train"
TEST_PATH = f"{DATASET_PATH}\\test"

# Image data generators
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load data
train_data = train_datagen.flow_from_directory(
    TRAIN_PATH, target_size=(64, 64), batch_size=24, class_mode='categorical'
)
test_data = test_datagen.flow_from_directory(
    TEST_PATH, target_size=(64, 64), batch_size=24, class_mode='categorical'
)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), input_shape=(64,64,3)),
    LeakyReLU(alpha=0.1),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3)),
    LeakyReLU(alpha=0.1),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64),
    LeakyReLU(alpha=0.1),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.005), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_data, validation_data=test_data, epochs=20,verbose=1)


# Evaluate on the test data
test_loss, test_accuracy = model.evaluate(test_data, verbose=0)
# Save the model
MODEL_PATH = r'C:\Vaishanvi\college\PS\codes\data_set\Indian_Medicinal_Plants\CNN\CNN_tensorflow.h5'
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")


# Print training and testing accuracy
final_training_accuracy = history.history['accuracy'][-1]
final_testing_accuracy = test_accuracy
print(f"Final Training Accuracy: {final_training_accuracy:.2f}")
print(f"Final Testing Accuracy: {final_testing_accuracy:.2f}")
# Plot training and validation loss
# plt.figure(figsize=(8, 6))
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Loss vs Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid()
# plt.show()

# # Plot training and validation accuracy
# plt.figure(figsize=(8, 6))
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Accuracy vs Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.grid()
# plt.show()

