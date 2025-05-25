import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, Add
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet152
from sklearn.utils import resample


# Define dataset paths
base_path = "/home/rf/Desktop/CNN_exp/"
classes = {"10_BPM_out/Filtered": 0, "15_BPM_out/Filtered": 1, "20_BPM_out/Filtered": 2, "Empty_out/Filtered": 3, "No_Breathing_out/Filtered": 4}
num_classes = len(classes)

window_size=1350
step_size=90
num_subcarriers=10
max_samples_per_class=300

# Apply augmentation to each training sample
def augment_signal(data):
    # data = add_gaussian_noise(data)
    return data

# Load CSI dataset
def load_csi_data(base_path, window_size=1350, step_size=90, num_subcarriers=10, max_samples_per_class=300):
    data, labels = [], []
    
    for class_name, label in classes.items():
        class_path = os.path.join(base_path, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: Directory {class_path} not found!")
            continue
        
        class_data, class_labels = [], []
        for file in os.listdir(class_path):
            file_path = os.path.join(class_path, file)
            if file.endswith('.csv'):
                # df = pd.read_csv(file_path, header=None, nrows=2700)  # Limit to first 2700 rows
                df = pd.read_csv(file_path, header=None)
                
                csi_values = df.iloc[:, :num_subcarriers].values
                csi_values = np.expand_dims(csi_values, axis=-1)  # Add a channel dimension

                num_windows = (csi_values.shape[0] - window_size) // step_size + 1  # Number of windows
                for i in range(num_windows):
                    window = csi_values[i * step_size:i * step_size + window_size]
                    class_data.append(window)
                    class_labels.append(label)

        # Ensure equal number of samples per class
        if len(class_data) > max_samples_per_class:
            class_data, class_labels = resample(class_data, class_labels, n_samples=max_samples_per_class, random_state=42)

        data.extend(class_data)
        labels.extend(class_labels)

    return np.array(data), np.array(labels)

# Load data
X, y = load_csi_data(base_path, window_size=window_size, step_size=step_size, max_samples_per_class=max_samples_per_class, num_subcarriers=num_subcarriers)

# Save the data
# np.savez("5_classes_15s_samples_dataset.npz", x=X, y=y)

# Ensure consistency in sequence length (1350 time steps for each sample)
max_length = max(sample.shape[0] for sample in X)
X = np.array([np.pad(sample, ((0, max_length - sample.shape[0]), (0, 0), (0,0)), mode='constant') for sample in X])

# One-hot encode labels for 3 classes
y = to_categorical(y, num_classes=num_classes)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y, shuffle=True)

X_train = np.array([augment_signal(sample) for sample in X_train])
X_test = np.array([augment_signal(sample) for sample in X_test])

# # Modify the input shape to (1350, 10, 3) by repeating the single channel (1) to create 3 channels
# X_train = np.repeat(X_train, 3, axis=-1)  # Repeat along the channel axis (create 3 channels)
# X_test = np.repeat(X_test, 3, axis=-1)  # Repeat along the channel axis (create 3 channels)

# # Resize the input data to meet the minimum required size for ResNet (32x32)
# X_train_resized = tf.image.resize(X_train, (32, 32))  # Resize to 32x32
# X_test_resized = tf.image.resize(X_test, (32, 32))  # Resize to 32x32

# print(X_train.shape)

# Define the ResNet model function
def create_residual_model(input_shape):
    # Input layer
    input_layer = layers.Input(shape=input_shape)
    
    # First convolutional block
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding="same", kernel_regularizer=l2(0.01))(input_layer)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Second convolutional block
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding="same", kernel_regularizer=l2(0.01))(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Residual block with fewer filters
    residual = layers.Conv2D(filters=128, kernel_size=(3, 3), activation=None, padding="same")(x)
    x = layers.Add()([x, residual])
    x = layers.ReLU()(x)
    
    # Flatten and dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    
    # Output layer with 3 classes (Empty, Normal, No Breathing)
    # output_layer = layers.Dense(2, activation='softmax')(x)
    output_layer = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create the model
    model = models.Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Create and compile the ResNet model
model = create_residual_model([1350, 10, 1])
# optimizer = Adam(learning_rate=0.0001)
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test), shuffle=True)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Plot Loss
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

# Save the model
model.save("csi_breathing_residual_BPM.h5")

# Predict on the test set
y_pred = model.predict(X_test)

# Convert predictions from one-hot encoded to class labels
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Generate the confusion matrix
cm = confusion_matrix(y_test_labels, y_pred_labels)

# Print the confusion matrix
print("Confusion Matrix:")
print(cm)

# Plot the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["10 BPM", "15 BPM", "20 BPM", "Empty Room", "No Breathing"], yticklabels=["10 BPM", "15 BPM", "20 BPM", "Empty Room", "No Breathing"])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Identify misclassified samples
misclassified_idxs = np.where(y_pred_labels != y_test_labels)[0]

# Plot misclassified CSI heatmaps
num_mistakes = len(misclassified_idxs)
print(f"Number of misclassified samples: {num_mistakes}")

if num_mistakes > 0:
    fig, axes = plt.subplots(nrows=min(10, num_mistakes), figsize=(20, 20))
    for i, idx in enumerate(misclassified_idxs[:5]):
        ax = axes[i] if num_mistakes > 1 else axes
        ax.imshow(X_test[idx].squeeze().T, aspect='auto', cmap='viridis')
        ax.set_title(f"Actual: {y_test_labels[idx]}, Predicted: {y_pred_labels[idx]}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Subcarriers")
    plt.tight_layout()
    plt.show()

def load_and_preprocess_file(file_path, window_size=1350, step_size=90, num_subcarriers=10):
    # Step 1: Load the CSV file
    df = pd.read_csv(file_path, header=None)  # Assuming no header in your CSV
    
    # Step 2: Extract the relevant CSI data (first 10 subcarriers)
    csi_values = df.iloc[:, :num_subcarriers].values
    
    # Step 3: Reshape the data into overlapping windows
    num_windows = (csi_values.shape[0] - window_size) // step_size + 1  # Number of windows
    
    windows = []
    
    for i in range(num_windows):
        window = csi_values[i * step_size:i * step_size + window_size]
        windows.append(window)
    
    # Convert windows to a numpy array
    windows = np.array(windows)
    
    # Step 4: Add the channel dimension (reshape to (1350, 10, 1) for each sample)
    windows = np.expand_dims(windows, axis=-1)  # Adds the channel dimension
    
    # Step 5: Ensure the input is ready for model inference
    return windows  # Shape should be (num_windows, 1350, 10, 1)

# # Example usage
# file_path = '/home/rf/Desktop/Filtered/Normal1_filtered_dataset.csv'  # Replace with your actual file path
# processed_data = load_and_preprocess_file(file_path)

# # Now, the data is ready for inference
# model = tf.keras.models.load_model('csi_breathing_resnet152.h5')  # Load your saved model
# predictions = model.predict(processed_data)

# # Get the predicted class for each window
# predicted_classes = np.argmax(predictions, axis=1)

# print("Predicted classes for the windows:")
# for i, pred in enumerate(predicted_classes):
#     print(f"Window {i + 1}: Predicted class = {pred}")