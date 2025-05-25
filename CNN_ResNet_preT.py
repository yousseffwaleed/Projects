import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, losses
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample

# Define dataset paths
base_path = "/home/rf/Desktop/CNN_exp/"

# Separate classes for each BPM
classes = {
    "10_BPM_out/Filtered": 0, 
    "11_BPM_out/Filtered": 1, 
    "12_BPM_out/Filtered": 2,  
    "13_BPM_out/Filtered": 3, 
    "14_BPM_out/Filtered": 4, 
    "15_BPM_out/Filtered": 5,  
    "16_BPM_out/Filtered": 6, 
    "17_BPM_out/Filtered": 7, 
    "18_BPM_out/Filtered": 8,  
    "19_BPM_out/Filtered": 9, 
    "20_BPM_out/Filtered": 10,  
    "Empty_out/Filtered": 11, 
    "No_Breathing_out/Filtered": 12  
}
num_classes = len(set(classes.values()))

# Hyperparameters
window_size = 1350
step_size = 90
num_subcarriers = 10
max_samples_per_class = {
    0: 405,   # 10 BPM
    10: 405,  # 20 BPM
    1: 225,   # 11 BPM
    2: 225,   # 12 BPM
    3: 225,   # 13 BPM
    4: 225,   # 14 BPM
    5: 225,   # 15 BPM
    6: 225,   # 16 BPM
    7: 225,   # 17 BPM
    8: 225,   # 18 BPM
    9: 225,   # 19 BPM
    11: 300,  # Empty Room
    12: 300   # No Breathing
}

def load_csi_data(base_path, window_size=1350, step_size=90, num_subcarriers=10, max_samples_per_class=None):
    data, labels = [], []
    grouped_data = {label: [] for label in classes.values()}
    
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
    
    # Balance dataset by resampling each class
    for label, samples in grouped_data.items():
        # Determine max samples for this class
        max_samples = max_samples_per_class.get(label, len(samples))
        
        # Resample or truncate
        if len(samples) > max_samples:
            balanced_samples = resample(samples, n_samples=max_samples, random_state=42, replace=False)
        else:
            balanced_samples = samples
        
        data.extend(balanced_samples)
        labels.extend([label] * len(balanced_samples))
    
    return np.array(data), np.array(labels)

# Load and preprocess data
X, y = load_csi_data(base_path, window_size=window_size, step_size=step_size, 
                     max_samples_per_class=max_samples_per_class, num_subcarriers=num_subcarriers)

# Ensure consistent input shape by padding
max_length = max(sample.shape[0] for sample in X)
X = np.array([np.pad(sample, ((0, max_length - sample.shape[0]), (0, 0), (0,0)), mode='constant') for sample in X])

# Repeat across channels for compatibility with ResNet
X_repeat = np.repeat(X, 3, axis=-1)

# Resize for ResNet input
X_resized = tf.image.resize(X_repeat, (32, 32))

# Custom TensorFlow train_test_split
def tf_train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Custom train-test split function compatible with TensorFlow tensors
    
    Args:
    X (numpy.ndarray or tf.Tensor): Input features
    y (numpy.ndarray or tf.Tensor): Input labels
    test_size (float): Proportion of the dataset to include in the test split
    random_state (int, optional): Seed for random number generation
    
    Returns:
    Tuple of numpy arrays: X_train, X_test, y_train, y_test
    """
    # Ensure inputs are numpy arrays
    X = np.asarray(X)
    y = np.asarray(y)
    
    # Get total number of samples
    total_samples = X.shape[0]
    
    # Compute test and train sample counts
    test_samples = int(total_samples * test_size)
    train_samples = total_samples - test_samples
    
    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate random indices
    indices = np.random.permutation(total_samples)
    
    # Split indices
    train_indices = indices[:train_samples]
    test_indices = indices[train_samples:]
    
    # Split data
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

# Split data into training and testing
X_train, X_test, y_train, y_test = tf_train_test_split(
    X_resized, y, 
    test_size=0.2, 
    random_state=42
)

def prototypical_loss(y_true, y_pred):
    """
    Prototypical Network Loss using TensorFlow operations
    
    Args:
    y_true: True labels tensor
    y_pred: Predicted embeddings tensor
    
    Returns:
    Scalar loss value
    """
    # Convert y_true to tensor if it's not already
    y_true = tf.cast(y_true[0], tf.int32)
    
    # Compute unique classes
    unique_classes = tf.unique(y_true)[0]
    n_way = tf.shape(unique_classes)[0]
    
    # Compute prototypes (class means)
    def get_prototypes(embeddings, labels):
        prototypes = {}
        for label in tf.unstack(unique_classes):
            # Create a mask for the current class
            mask = tf.equal(labels, label)
            
            # Compute mean of embeddings for this class
            class_embeddings = tf.boolean_mask(embeddings, mask)
            prototypes[label] = tf.reduce_mean(class_embeddings, axis=0)
        
        return prototypes
    
    # Compute distances between embeddings and prototypes
    def compute_distances(embeddings, prototypes):
        distances = {}
        for label, proto in prototypes.items():
            # Compute squared Euclidean distance
            distances[label] = tf.reduce_sum(tf.square(embeddings - proto), axis=-1)
        return distances
    
    # Compute loss
    def loss(embeddings, labels):
        # Get prototypes for each unique class
        prototypes = get_prototypes(embeddings, labels)
        
        # Compute distances to prototypes
        distances = compute_distances(embeddings, prototypes)
        
        # Prepare logits by negating distances
        logits = -tf.stack([distances[label] for label in unique_classes], axis=-1)
        
        # Compute sparse softmax cross-entropy loss
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.cast(labels, tf.int32), 
                logits=logits
            )
        )
    
    return loss(y_pred, y_true)

def create_prototypical_network(input_shape):
    # Embedding network
    inputs = layers.Input(shape=input_shape)
    
    # ResNet152 base model as feature extractor
    base_model = tf.keras.applications.ResNet152(
        weights='imagenet', 
        include_top=False, 
        input_shape=input_shape
    )
    base_model.trainable = False
    
    # Feature extraction layers
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    embeddings = layers.Dense(64, activation=None)(x)  # Final embedding layer
    
    model = Model(inputs=inputs, outputs=embeddings)
    
    return model

# Create and compile the prototypical network
embedding_network = create_prototypical_network((32, 32, 3))

# Custom compilation with prototypical loss
embedding_network.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=prototypical_loss
)

# Train the embedding network
history = embedding_network.fit(
    X_train, y_train, 
    epochs=20, 
    batch_size=32, 
    validation_split=0.2
)

# Rest of the code remains the same as in the previous implementation...
# (Compute embeddings, visualize, classify, etc.)

# Compute embeddings for all samples
train_embeddings = embedding_network.predict(X_train)
test_embeddings = embedding_network.predict(X_test)

# Visualize training history
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.title('Embedding Distribution')
plt.scatter(train_embeddings[:, 0], train_embeddings[:, 1], c=y_train, cmap='viridis')
plt.colorbar(label='Class')
plt.tight_layout()
plt.show()

# Compute prototypes for each class
def compute_prototypes(embeddings, labels):
    prototypes = {}
    for label in np.unique(labels):
        prototypes[label] = np.mean(embeddings[labels == label], axis=0)
    return prototypes

train_prototypes = compute_prototypes(train_embeddings, y_train)

# Classification using nearest prototype
def classify_with_prototypes(embeddings, prototypes):
    predictions = []
    for emb in embeddings:
        # Compute distances to each prototype
        distances = {label: np.linalg.norm(emb - proto) for label, proto in prototypes.items()}
        predictions.append(min(distances, key=distances.get))
    return np.array(predictions)

# Predict on test set
y_pred = classify_with_prototypes(test_embeddings, train_prototypes)
y_test_labels = y_test

# Compute and plot confusion matrix
cm = confusion_matrix(y_test_labels, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[list(classes.keys())[list(classes.values()).index(i)] for i in range(num_classes)],
            yticklabels=[list(classes.keys())[list(classes.values()).index(i)] for i in range(num_classes)])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Compute accuracy
accuracy = np.mean(y_pred == y_test_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the embedding network
embedding_network.save("csi_prototypical_network.h5")