import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
# The path to your dataset.
# This path is now corrected to point to the PetImages folder.
DATA_DIR = 'train/kagglecatsanddogs_5340/PetImages'
IMG_SIZE = 100 # All images will be resized to 100x100 pixels.

# --- 2. DATA PREPARATION AND LOADING ---
def create_dataset(data_dir):
    """
    Loads images from subfolders, resizes them, flattens them, and creates labels.
    Returns:
        images (np.array): A numpy array of flattened image data.
        labels (np.array): A numpy array of labels (0 for cats, 1 for dogs).
    """
    images = []
    labels = []
    
    # Check if the data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}. Please check your folder structure.")
        return None, None

    # Loop through the 'cats' and 'dogs' subdirectories
    for category in ['Cat', 'Dog']: # Corrected category names to match your folder names
        path = os.path.join(data_dir, category)
        label = 0 if category == 'Cat' else 1
        
        print(f"Processing {category} images from {path}...")
        
        # Iterate over each image file
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            try:
                # Read the image in grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    # Resize the image to a fixed size
                    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    
                    # Flatten the 2D image array into a 1D vector
                    images.append(img_resized.flatten())
                    labels.append(label)
            except Exception as e:
                # Handle potential issues with corrupted images
                print(f"Error processing {img_path}: {e}")
                
    # Convert lists to numpy arrays for efficient processing
    return np.array(images), np.array(labels)

# Load the dataset
X, y = create_dataset(DATA_DIR)

if X is None or y is None:
    print("Dataset could not be loaded. Exiting.")
    exit()

print(f"Total images loaded: {len(X)}")
print(f"Image vector size: {X.shape[1]}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# --- 3. MODEL TRAINING ---
# Create an SVM classifier with a Radial Basis Function (RBF) kernel
print("\nCreating and training the SVM model...")
svm_model = SVC(kernel='rbf', C=1, gamma='scale', verbose=True) # verbose=True to show progress
svm_model.fit(X_train, y_train)
print("Training complete!")

# --- 4. MODEL EVALUATION ---
# Make predictions on the test set
print("\nMaking predictions on the test set...")
y_pred = svm_model.predict(X_test)

# Calculate and print the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Generate and print a classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Cat', 'Dog']))

# --- 5. VISUALIZATION (Optional) ---
# Visualize a few test images and their predicted labels
def visualize_predictions(X_test, y_test, y_pred, num_images=5):
    """
    Shows a few test images with their true and predicted labels.
    """
    plt.figure(figsize=(15, 8))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        
        # Reshape the flattened vector back into a 2D image
        img = X_test[i].reshape(IMG_SIZE, IMG_SIZE)
        
        plt.imshow(img, cmap='gray')
        plt.title(f"True: {'Dog' if y_test[i] else 'Cat'}\nPred: {'Dog' if y_pred[i] else 'Cat'}",
                  color='green' if y_test[i] == y_pred[i] else 'red')
        plt.axis('off')
    plt.show()

# Visualize the first 5 predictions from the test set
print("\nVisualizing a few predictions...")
visualize_predictions(X_test, y_test, y_pred)