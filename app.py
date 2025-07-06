from flask import Flask, render_template, request, redirect, url_for, session
import cv2
import numpy as np
import os
import joblib
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for session management

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "ecg_rf_model.pkl")
model = joblib.load(model_path)

categories = ['Abnormal_Heartbeat', 'History_of_MI', 'Myocardial_Infarction', 'Normal']

def preprocess_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = img.reshape(1, -1)
    return img

@app.route('/')
def login():
    if 'logged_in' in session and session['logged_in']:
        return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def do_login():
    # Accept any username/password — no validation
    username = request.form.get('email')
    password = request.form.get('password')

    # Just set session logged_in to True and redirect
    session['logged_in'] = True
    return redirect(url_for('home'))

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/home')
def home():
    if 'logged_in' not in session or not session['logged_in']:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'logged_in' not in session or not session['logged_in']:
        return redirect(url_for('login'))

    if 'ecgFile' not in request.files:
        return "No file uploaded", 400

    file = request.files['ecgFile']
    if file.filename == '':
        return "No selected file", 400

    filename = secure_filename(file.filename)
    upload_dir = os.path.join(BASE_DIR, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, filename)
    file.save(file_path)

    img = preprocess_image(file_path)
    prediction = model.predict(img)[0]
    result = categories[prediction]

    return render_template('result.html', prediction=result)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:
            return render_template('register.html', error="Passwords do not match")

        # TODO: Save user credentials in DB or file
        # For now, redirect to login page
        return redirect(url_for('login'))

    return render_template('register.html')


if __name__ == '__main__':
    app.run(debug=True)
