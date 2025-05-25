import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, Add
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
from tensorflow.keras.callbacks import ModelCheckpoint

# Define dataset paths
base_path = "/home/rf/Desktop/CNN_exp/"
# Separate classes for each BPM
classes = {
    "10_BPM_out/Filtered": 0, 
    "11_BPM_out/Filtered": 1, 
    "12_BPM_out/Filtered": 1,  
    "13_BPM_out/Filtered": 1, 
    "14_BPM_out/Filtered": 2, 
    "15_BPM_out/Filtered": 2,  
    "16_BPM_out/Filtered": 2, 
    "17_BPM_out/Filtered": 3, 
    "18_BPM_out/Filtered": 3,  
    "19_BPM_out/Filtered": 3, 
    "20_BPM_out/Filtered": 4,  
    "Empty_out/Filtered": 5, 
}

class_names = {
    "< 10 BPM": 0,
    "11 - 13 BPM": 1,
    "14 - 16 BPM": 2,
    "17 - 19 BPM": 3,
    "> 20 BPM": 4,
    "Empty Room": 5,
}

num_classes = len(set(classes.values()))

window_size = 900
step_size = 90
num_subcarriers = 10
max_samples_per_class = 450  # Max samples per class for training and testing

# Function to load and preprocess the data
def load_csi_data(base_path, window_size=900, step_size=90, num_subcarriers=10, max_samples_per_class=400):
    data, labels = [], []
    
    # Define which classes should have resampling
    intermediate_bpm_classes = [1, 2, 3]  # 11-19 BPM combined
    # max_samples_per_class = 400
    
    # Iterate through each class and load the data
    for class_name, label in classes.items():
        class_path = os.path.join(base_path, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: Directory {class_path} not found!")
            continue
        
        class_data, class_labels = [], []
        for file in os.listdir(class_path):
            file_path = os.path.join(class_path, file)
            if file.endswith('.csv'):
                df = pd.read_csv(file_path, header=None)
                csi_values = df.iloc[:, :num_subcarriers].values
                num_windows = (csi_values.shape[0] - window_size) // step_size + 1
                
                for i in range(num_windows):
                    window = csi_values[i * step_size:i * step_size + window_size]
                    class_data.append(window)
                    class_labels.append(label)
        
        print("Class name", class_name, ":", np.array(class_data).shape, "Label:", label)
        
        if(label == 0 or label ==4 or label ==5):
            max_samples_per_class =  402
        else:
            max_samples_per_class = 134

        # Ensure we don't exceed max_samples_per_class
        if len(class_data) > max_samples_per_class:
            # if label in intermediate_bpm_classes:
            #      class_data, class_labels = resample(class_data, class_labels, n_samples=max_samples_per_class//3, random_state=42)
            # else:
            class_data, class_labels = resample(class_data, class_labels, n_samples=max_samples_per_class, random_state=42)
            print("2)Class name", class_name, ":", np.array(class_data).shape)
        # Append the data and labels
        data.extend(class_data)
        labels.extend(class_labels)
    
    return np.array(data), np.array(labels)

# Load the dataset
X, y = load_csi_data(base_path, window_size, step_size, num_subcarriers, max_samples_per_class)

# One-hot encode labels for all classes
y = to_categorical(y, num_classes=num_classes)

# Print the dataset shape
print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)

# Split the dataset into training and testing sets (50/50 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)

# X_train = np.array([augment_signal(sample) for sample in X_train])
# X_test = np.array([augment_signal(sample) for sample in X_test])

# # Modify the input shape to (1350, 10, 3) by repeating the single channel (1) to create 3 channels
# X_train = np.repeat(X_train, 3, axis=-1)  # Repeat along the channel axis (create 3 channels)
# X_test = np.repeat(X_test, 3, axis=-1)  # Repeat along the channel axis (create 3 channels)

# # Resize the input data to meet the minimum required size for ResNet (32x32)
# X_train_resized = tf.image.resize(X_train, (32, 32))  # Resize to 32x32
# X_test_resized = tf.image.resize(X_test, (32, 32))  # Resize to 32x32

# print(X_train.shape)

# Defined 1D LSTM CNN Model
def create_1d_cnn_model(input_shape):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Create and compile the 1D CNN model
model = create_1d_cnn_model(input_shape=(window_size, num_subcarriers))

# Define checkpoint callback
checkpoint_filepath = '/tmp/ckpt/checkpoint.model.keras'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
)

# Print the model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test), shuffle=True,)

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
model.save("csi_br_cnn_1dconv.h5")

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

# Compute the accuracy per class (correct predictions / total samples per class)
class_accuracies = np.diag(cm) / np.sum(cm, axis=1)

# Print the class accuracies
print("Class-wise Accuracy (%):")
for i, class_name in enumerate(class_names.keys()):
    print(f"{class_name}: {class_accuracies[i] * 100:.2f}%")

# Compute and plot confusion matrix with accuracy percentages
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # Normalize to percentage

plt.figure(figsize=(12, 10))
sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues', 
            xticklabels=[list(class_names.keys())[list(class_names.values()).index(i)] for i in range(num_classes)],
            yticklabels=[list(class_names.keys())[list(class_names.values()).index(i)] for i in range(num_classes)])
plt.title('Confusion Matrix with Accuracy Percentage')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# # Identify misclassified samples
# misclassified_idxs = np.where(y_pred_labels != y_test_labels)[0]

# # Plot misclassified CSI heatmaps
# num_mistakes = len(misclassified_idxs)
# print(f"Number of misclassified samples: {num_mistakes}")

# # Identify misclassified samples
# misclassified_idxs = np.where(y_pred_labels != y_test_labels)[0]

# # Plot misclassified CSI heatmaps
# num_mistakes = len(misclassified_idxs)
# print(f"Number of misclassified samples: {num_mistakes}")

# if num_mistakes > 0:
#     fig, axes = plt.subplots(nrows=min(10, num_mistakes), figsize=(20, 20))
#     for i, idx in enumerate(misclassified_idxs[:5]):
#         ax = axes[i] if num_mistakes > 1 else axes
#         ax.imshow(X_test[idx].T, aspect='auto', cmap='viridis')
#         ax.set_title(f"Actual: {y_test_labels[idx]}, Predicted: {y_pred_labels[idx]}")
#         ax.set_xlabel("Time Step")
#         ax.set_ylabel("Subcarriers")
#     plt.tight_layout()
#     plt.show()

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

# Example usage
file_path = '/home/rf/Desktop/CNN_exp/10_BPM_out/10_BPM_2_filtered_dataset.csv'  # Replace with your actual file path
processed_data1 = load_and_preprocess_file(file_path, window_size=window_size, step_size=step_size, num_subcarriers=num_subcarriers)

# Now, the data is ready for inference
model = tf.keras.models.load_model('csi_breathing_residual_BPM.h5')  # Load your saved model
predictions = model.predict(processed_data1)

# Get the predicted class for each window
predicted_classes = np.argmax(predictions, axis=1)

print("Predicted classes for the windows:")
for i, pred in enumerate(predicted_classes):
    print(f"Window {i + 1}: Actual class = 0, Predicted class = {pred}")

# Example usage
file_path = '/home/rf/Desktop/CNN_exp/15_BPM_out/15_BPM_2_filtered_dataset.csv'  # Replace with your actual file path
processed_data2 = load_and_preprocess_file(file_path, window_size=window_size, step_size=step_size, num_subcarriers=num_subcarriers)

predictions = model.predict(processed_data2)

# Get the predicted class for each window
predicted_classes = np.argmax(predictions, axis=1)

print("Predicted classes for the windows:")
for i, pred in enumerate(predicted_classes):
    print(f"Window {i + 1}: Actual class = 1, Predicted class = {pred}")

# Example usage
file_path = '/home/rf/Desktop/CNN_exp/20_BPM_out/20_BPM_2_filtered_dataset.csv'  # Replace with your actual file path
processed_data3 = load_and_preprocess_file(file_path, window_size=window_size, step_size=step_size, num_subcarriers=num_subcarriers)

predictions = model.predict(processed_data3)

# Get the predicted class for each window
predicted_classes = np.argmax(predictions, axis=1)

print("Predicted classes for the windows:")
for i, pred in enumerate(predicted_classes):
    print(f"Window {i + 1}: Actual class = 2, Predicted class = {pred}")
