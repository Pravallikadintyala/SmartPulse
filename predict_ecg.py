import os
import cv2
import numpy as np
import joblib

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image {image_path}")
    img = cv2.resize(img, (128, 128))
    img = img.reshape(1, -1)  # flatten to 1D array, shape (1, 128*128)
    return img

# Load the trained model
model_path = "D:/ECG_Project/ecg_rf_model.pkl"
clf = joblib.load(model_path)

# Path to a new ECG image you want to classify
new_image_path = r"D:\ECG_Project\ECG_Dataset\Abnormal_Heartbeat\HB(24).jpg" 

# Preprocess the image
X_new = preprocess_image(new_image_path)

# Predict
prediction = clf.predict(X_new)[0]

# Map prediction number to class name
categories = ['Abnormal_Heartbeat', 'History_of_MI', 'Myocardial_Infarction', 'Normal']
predicted_class = categories[prediction]

print(f"Predicted class for the image is: {predicted_class}")
