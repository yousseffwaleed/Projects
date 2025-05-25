import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.utils import resample

# Define dataset paths and classes for BPM ranges
base_path = "/home/rf/Desktop/CNN_exp/"

# Grouped classes
classes = {
    "10_BPM_out/Filtered": 0, "11_BPM_out/Filtered": 0, "12_BPM_out/Filtered": 0,  # Group 10-12 BPM
    "13_BPM_out/Filtered": 1, "14_BPM_out/Filtered": 1, "15_BPM_out/Filtered": 1,  # Group 13-15 BPM
    "16_BPM_out/Filtered": 2, "17_BPM_out/Filtered": 2, "18_BPM_out/Filtered": 2,  # Group 16-18 BPM
    "19_BPM_out/Filtered": 3, "20_BPM_out/Filtered": 3,  # Group 19-20 BPM
    "Empty_out/Filtered": 4, "No_Breathing_out/Filtered": 5  # Empty room and no breathing
}
num_classes = len(set(classes.values()))


window_size = 1350
step_size = 90
num_subcarriers = 10
max_samples_per_class = 300  # Number of samples per class to keep consistent

# Apply augmentation to each training sample (currently no augmentation applied)
def augment_signal(data):
    return data

# Load CSI data
def load_csi_data(base_path, window_size=1350, step_size=90, num_subcarriers=10, max_samples_per_class=300):
    data, labels = [], []
    grouped_data = {i: [] for i in range(num_classes)}
    
    # Load data from files
    for class_name, label in classes.items():
        class_path = os.path.join(base_path, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: Directory {class_path} not found!")
            continue
        
        for file in os.listdir(class_path):
            file_path = os.path.join(class_path, file)
            if file.endswith('.csv'):
                df = pd.read_csv(file_path, header=None)
                csi_values = df.iloc[:, :num_subcarriers].values
                csi_values = np.expand_dims(csi_values, axis=-1)

                # Extract sliding window samples
                num_windows = (csi_values.shape[0] - window_size) // step_size + 1
                for i in range(num_windows):
                    window = csi_values[i * step_size:i * step_size + window_size]
                    grouped_data[label].append(window)
    
    # Determine the minimum samples across all grouped classes
    min_samples = min(len(samples) for samples in grouped_data.values())
    min_samples = min(min_samples, max_samples_per_class)  # Cap at max_samples_per_class
    
    # Balance dataset by resampling each class to have the same number of samples
    for label, samples in grouped_data.items():
        balanced_samples = resample(samples, n_samples=min_samples, random_state=42, replace=False) if len(samples) > min_samples else samples
        data.extend(balanced_samples)
        labels.extend([label] * len(balanced_samples))
    
    return np.array(data), np.array(labels)

X, y = load_csi_data(base_path, window_size=window_size, step_size=step_size, max_samples_per_class=max_samples_per_class, num_subcarriers=num_subcarriers)

# Save the data
# np.savez("5_classes_15s_samples_dataset.npz", x=X, y=y)

# Ensure consistency in sequence length (1350 time steps for each sample)
max_length = max(sample.shape[0] for sample in X)
# X = np.array([np.pad(sample, ((0, max_length - sample.shape[0]), (0, 0)), mode='constant') for sample in X])

# One-hot encode labels for 6 classes (updated to 6 classes)
y = to_categorical(y, num_classes=num_classes)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y, shuffle=True)

X_train = np.array([augment_signal(sample) for sample in X_train])
X_test = np.array([augment_signal(sample) for sample in X_test])

# Define the 1D CNN-LSTM model
def create_cnn_lstm_model(input_shape):
    model = Sequential([
        # 1D CNN layers
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        # LSTM layers
        LSTM(64, return_sequences=True),
        Dropout(0.25),
        LSTM(128),
        Dropout(0.25),
        
        # Fully connected layers
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create and compile the model
model = create_cnn_lstm_model(input_shape=(1350, 10))

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
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["11-13 BPM", "12-14 BPM", "15-17 BPM", "18-20 BPM", "Empty Room", "No Breathing"], yticklabels=["11-13 BPM", "12-14 BPM", "15-17 BPM", "18-20 BPM", "Empty Room", "No Breathing"])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Identify misclassified samples
misclassified_idxs = np.where(y_pred_labels != y_test_labels)[0]

# Plot misclassified CSI heatmaps
num_mistakes = len(misclassified_idxs)
print(f"Number of misclassified samples: {num_mistakes}")

# if num_mistakes > 0:
#     fig, axes = plt.subplots(nrows=min(5, num_mistakes), figsize=(6, 6))
#     for i, idx in enumerate(misclassified_idxs[:5]):
#         ax = axes[i] if num_mistakes > 1 else axes
#         ax.imshow(X_test[idx].squeeze().T, aspect='auto', cmap='viridis')
#         ax.set_title(f"Actual: {y_test_labels[idx]}, Predicted: {y_pred_labels[idx]}")
#         ax.set_xlabel("Time Step")
#         ax.set_ylabel("Subcarriers")
#     plt.tight_layout()
#     plt.show()

# Example for file inference (uncomment if required)
# file_path = '/home/rf/Desktop/Filtered/Normal1_filtered_dataset.csv'  # Replace with actual file path
# processed_data = load_and_preprocess_file(file_path)
# predictions = model.predict(processed_data)
# predicted_classes = np.argmax(predictions, axis=1)