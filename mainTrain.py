import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt 
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc

''' CHELSI PATEL (22BAI10005)'''

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define constants
IMAGE_DIRECTORY = 'dataset/'
INPUT_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 10
NUM_CLASSES = 2  # Binary classification

# Supported image extensions
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

# Initialize dataset and labels
dataset = []
labels = []

def load_images(folder, label):
    """
    Loads and preprocesses images from a specified folder.

    Args:
        folder (str): Path to the image folder.
        label (int): Label to assign to the images.

    Returns:
        None
    """
    for image_name in os.listdir(folder):
        if image_name.lower().endswith(SUPPORTED_EXTENSIONS):
            image_path = os.path.join(folder, image_name)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Unable to read image '{image_path}'. Skipping.")
                continue
            # Convert BGR (OpenCV default) to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Resize image
            image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
            dataset.append(image)
            labels.append(label)

# Load 'no' tumor images
no_tumor_folder = os.path.join(IMAGE_DIRECTORY, 'no')
load_images(no_tumor_folder, label=0)

# Load 'yes' tumor images
yes_tumor_folder = os.path.join(IMAGE_DIRECTORY, 'yes')
load_images(yes_tumor_folder, label=1)

# Convert lists to numpy arrays
dataset = np.array(dataset)
labels = np.array(labels)

print(f"Total images loaded: {len(dataset)}")
print(f"Class distribution: {np.bincount(labels)}")

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    dataset, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"x_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Normalize the image data to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert labels to categorical (one-hot encoding)
y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
y_test = to_categorical(y_test, num_classes=NUM_CLASSES)

# Model Building
model = Sequential()

# First Convolutional Block
model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second Convolutional Block
model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Third Convolutional Block
model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Fully Connected Layers
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES))  # 2 neurons for binary classification with softmax

model.add(Activation('softmax'))

# Compile the model
model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy']
)

# Display model architecture
model.summary()

# Train the model
history = model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_test, y_test),
    shuffle=True  # Enable shuffling
)

loss, accuracy = model.evaluate(test_images, test_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")
# Save the trained model
model.save('BrainTumor10epochsCategorical.h5')
print("Model saved as 'BrainTumor10epochsCategorical.h5'")


# Plot accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_plot.eps', format='eps', dpi=300)
plt.show()


# Plot loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot.eps', format='eps', dpi=300)
plt.show()


# Predict classes for the test set
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert one-hot to class indices
y_true = np.argmax(y_test, axis=1)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Tumor', 'Tumor'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.eps', format='eps', dpi=300)
plt.show()

print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=['No Tumor', 'Tumor']))

# Compute ROC curve and AUC for each class
fpr = {}
tpr = {}
roc_auc = {}
for i in range(NUM_CLASSES):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve
plt.figure(figsize=(10, 7))
for i in range(NUM_CLASSES):
    plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guessing")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('roc_curve.eps', format='eps', dpi=300)
plt.show()


def random_test(x_test, y_test, model, num_samples=5):

    plt.figure(figsize=(15, num_samples * 3))
    
    for i in range(num_samples):
        # Randomly select an index
        random_index = random.randint(0, len(x_test) - 1)
        
        # Extract the test image and its ground truth
        test_image = x_test[random_index]
        ground_truth_label = np.argmax(y_test[random_index])  # Convert one-hot to class index
        
        # Predict the class
        prediction = model.predict(np.expand_dims(test_image, axis=0))
        predicted_label = np.argmax(prediction)
        
        # Map labels to human-readable names
        label_map = {0: "No Tumor", 1: "Tumor"}
        ground_truth_text = label_map[ground_truth_label]
        predicted_text = label_map[predicted_label]
        
        # Plot the image
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(test_image)
        plt.axis('off')
        plt.title(
            f"True: {ground_truth_text}\nPredicted: {predicted_text}",
            color="green" if ground_truth_label == predicted_label else "red",
        )
    
    plt.tight_layout()
    plt.savefig('random_test.eps', format='eps', dpi=300)  # Save as EPS
    plt.show()

# Run the random test
random_test(x_test, y_test, model, num_samples=5)

