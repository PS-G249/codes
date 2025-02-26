
import tensorflow as tf
import numpy as np #Handles arrays and numerical computations
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential#For sequential layers
from tensorflow.keras.layers import Dense, Flatten, Dropout,Activation #Dense-Full connected layer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os #Used for file operations(creating directories,saving files etc..)

# Dataset Paths & Parameters
train_dir = r"C:\Vaishanvi\college\PS\codes\data_set\Indian_Medicinal_Plants\CNN\increased_dataset_split\increased_dataset_split\train"
test_dir = r"C:\Vaishanvi\college\PS\codes\data_set\Indian_Medicinal_Plants\CNN\increased_dataset_split\increased_dataset_split\test"
classes = ["AloVera", "Amla", "Brahmi", "Neem", "Tulasi"]
img_height, img_width = 64, 64
#batch_size_train= 1020
#batch_size_test=255
batch_size=24 #mini batch gradient descent is used  #Generally batch_size is the number of images processes in one training step
learning_rate = 0.005  # It is a hyperparameter that controls how much a model updates its weights suring tarining

# Data Preparation with Augmentation
train_gen = ImageDataGenerator(  #Image data generator prepares images for training or testing
    rescale=1.0 / 255 #Normalizes pixel values to the range [0,1]
    # rotation_range=20,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # horizontal_flip=True
)
test_gen = ImageDataGenerator(rescale=1.0 / 255)

#Set seed for reproducibility
SEED = 42  # You can change this to any fixed number
# random.seed(SEED)
#np.random.seed(SEED)
tf.random.set_seed(SEED)
#os.environ['PYTHONHASHSEED'] = str(SEED)

#Loading the dataset 
train_data = train_gen.flow_from_directory(
    train_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical'
)#Here it reads images from train_dir and test_dir and resizes them to (64,64) and loads images in bathes of 24
#and uses categorical mode which means Labels are one_hot encoded.
test_data = test_gen.flow_from_directory(
    test_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical'
)

# Defining the newral network Model with He Initialization
def build_model():
    model = Sequential([
        Flatten(input_shape=(img_height, img_width, 3)),#Converts the 3d image into 1D array
        Dense(768, kernel_initializer='he_uniform'),#he_uniform is generally a weight initialiation method
        Activation('tanh'),
        #ReLU(),
        #LeakyReLU(alpha=0.01),
        Dense(96, kernel_initializer='he_uniform'),#Second fully connected layer #he-uniform=perfect form of initialization formula=np.random.randint(input_size,output_size)*np.sqrt(2/input_size)
        Activation('tanh'),
        #ReLU(),
        #LeakyReLU(alpha=0.01),
        Dense(len(classes), activation='softmax')#Output layer with 5 neurons(one per class),using softmax to predicy probabilities
    ])
    return model

model = build_model()

# Compile Model with SGD
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
              loss='categorical_crossentropy',#Loss function for multi class classification
              metrics=['accuracy'])#Model evaluated accuracy during training

# Train Model for More Epochs
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=10
)

# Save & Evaluate Model
model.save("indian_medicinal_plants_model_sgd_optimized.h5")
test_loss, test_acc = model.evaluate(test_data)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Ensure the save directory exists
save_dir = r"C:\Vaishanvi\college\PS\codes\data_set\Indian_Medicinal_Plants\ANN"
os.makedirs(save_dir, exist_ok=True)

# Plot Loss vs. Epochs
# plt.figure(figsize=(8, 6))
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Loss vs. Epochs')
# plt.legend()
# plt.grid()
# loss_graph_path = os.path.join(save_dir, "loss_vs_epochs(6).png")
# plt.savefig(loss_graph_path)  # Save the plot
# plt.close() 

# # Plot Accuracy vs. Epochs
# plt.figure(figsize=(8, 6))
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.title('Accuracy vs. Epochs')
# plt.legend()
# plt.grid()
# accuracy_graph_path = os.path.join(save_dir, "accuracy_vs_epochs(6).png")
# plt.savefig(accuracy_graph_path)  # Save the plot
# plt.close()  # Close the plot
