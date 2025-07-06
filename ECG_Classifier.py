import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def load_images():
    data_dir = r"D:\ECG_Project\ECG_Dataset"
    categories = ['Abnormal_Heartbeat', 'History_of_MI', 'Myocardial_Infarction', 'Normal']
    
    images = []
    labels = []
    
    for label, category in enumerate(categories):
        folder = os.path.join(data_dir, category)
        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # assuming grayscale images
            if img is not None:
                img = cv2.resize(img, (128, 128))  # resize images as needed
                images.append(img)
                labels.append(label)
            else:
                print(f"Warning: Unable to read image {img_path}")

    X = np.array(images)
    y = np.array(labels)
    
    # Add a channel dimension
    X = X.reshape(-1, 128, 128, 1)
    
    return X, y

# Step 1: Load the images
X, y = load_images()
print(f"Loaded {len(X)} images with labels.")

# Step 2: Flatten image data for classical ML
X_flat = X.reshape(X.shape[0], -1)

# Step 3: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)

# Step 4: Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Step 5: Make predictions and evaluate
y_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the trained model
model_path = "D:/ECG_Project/ecg_rf_model.pkl"
joblib.dump(clf, model_path)
print(f"Model saved to {model_path}")
