from flask import Flask, render_template, request, redirect, url_for, flash, session
import cv2
import time
import numpy as np
import os
from collections import Counter
import time
from keras.models import load_model
from flask import Response
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a secure random key in a production environment

def preprocess_image(img, target_shape=(224, 224)):
    resized_img = cv2.resize(img, target_shape)
    return resized_img
def color_normalization(eye_region):
    lab = cv2.cvtColor(eye_region, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    filtered_eye = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return filtered_eye

def preprocess_eye_region(eye_region, target_shape=(224, 224)):
    resized_eye = cv2.resize(eye_region, target_shape)
    return resized_eye


def detect_pupil(eye_region, frame_counter):
    preprocessed_eye = preprocess_eye_region(eye_region)

    gray_eye = cv2.cvtColor(preprocessed_eye, cv2.COLOR_BGR2GRAY)

    _, thresholded = cv2.threshold(gray_eye, 75, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5, 5), np.uint8)
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresholded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        valid_contours = [contour for contour in contours if cv2.contourArea(contour) > 75]

        if valid_contours:
            largest_contour = max(valid_contours, key=cv2.contourArea)

            x_pupil, y_pupil, w_pupil, h_pupil = cv2.boundingRect(largest_contour)

          
            if 10 < w_pupil < 50 and 10 < h_pupil < 50:
                cv2.rectangle(eye_region, (x_pupil, y_pupil), (x_pupil + w_pupil, y_pupil + h_pupil), (0, 255, 0), 2)

                new_frame_name = f"frame_{frame_counter}_pupil_detected.png"
                cv2.imwrite(os.path.join(output_folder_pupil, new_frame_name), cv2.cvtColor(eye_region, cv2.COLOR_BGR2RGB))
                return True

  
    new_frame_name = f"frame_{frame_counter}_no_pupil_detected.png"
    cv2.imwrite(os.path.join(output_folder_pupil, new_frame_name), cv2.cvtColor(eye_region, cv2.COLOR_BGR2RGB))
    return False
vgg_model_path = "VGG_RNR_model.h5"
vgg_model = load_model(vgg_model_path)

efficientnet_model_path = 'efficientnet_custom_c_n_model_MRL.h5'
efficientnet_model = load_model(efficientnet_model_path)

cap = cv2.VideoCapture(0)

# Global variables for sleep duration adjustment
prev_sleep_hours = 0
working_hours = 0
medication_taken = False
sleep_duration = 0
# Load your models
vgg_model_path = "VGG_RNR_model.h5"
vgg_model = load_model(vgg_model_path)

efficientnet_model_path = 'efficientnet_custom_c_n_model_MRL.h5'
efficientnet_model = load_model(efficientnet_model_path)
# Define your preprocessing functions here
total_personalization_time = 60  
batch_duration = 10
current_timestamp = time.time()
open_eye_counts = 0
closed_eye_counts = {}
start_frame_time = time.time()

# Load driver information from the driver_database.txt file
def load_driver_database():
    driver_database = []
    with open('driver_database.txt', 'r') as file:
        for line in file:
            sl_no, driver_id, driver_name, appointed_district, driver_age = line.strip().split(',')
            driver_info = {
                "sl_no": int(sl_no),
                "driver_id": int(driver_id),
                "driver_name": driver_name,
                "appointed_district": appointed_district,
                "driver_age": int(driver_age),
            }
            driver_database.append(driver_info)
    return driver_database

# Sample registered drivers (replace with a database in a production environment)
registered_drivers = {}

# Load driver database
driver_database = load_driver_database()

# Function to check if driver is registered
def is_driver_registered(driver_id, password):
    with open('registered_drivers.txt', 'r') as file:
        for line in file:
            line = line.strip().split(',')
            if int(line[0]) == driver_id and line[1] == password:
                return True
    return False
def get_driver_details(driver_id):
    for driver in driver_database:
        if driver['driver_id'] == driver_id:
            return driver
    return None
# Function to save driver ID and password to a text file
def save_credentials(driver_id, password):
    with open('driver_credentials.txt', 'a') as file:
        file.write(f"{driver_id},{password}\n")

# Function to check if driver ID and password match
def authenticate(driver_id, password):
    with open('driver_credentials.txt', 'r') as file:
        for line in file:
            line = line.strip().split(',')
            if int(line[0]) == driver_id and line[1] == password:
                return True
    return False


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        driver_name = request.form['driver_name']
        driver_id = int(request.form['driver_id'])
        phone_number = request.form['phone_number']
        password = request.form['password']

        # Save the driver's credentials to the text file
        save_credentials(driver_id, password)
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))

    return render_template('register.html')
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        driver_id = int(request.form['driver_id'])
        password = request.form['password']

        # Check if authentication is successful
        if authenticate(driver_id, password):
            session['driver_id'] = driver_id
            flash('Successful login!')
            return redirect(url_for('input_collection'))
        else:
            flash('Invalid credentials. Please try again.')

    return render_template('login.html')

# Driver interface route
@app.route('/driver_interface/<driver_id>')
def driver_interface(driver_id):
    # Retrieve driver details for rendering in the interface
    driver_details = registered_drivers.get(driver_id, {}).get('driver_details', {})
    return render_template('driver_interface.html', driver_id=driver_id, driver_details=driver_details)
@app.route('/')
def home():
    return render_template('home.html')
# Add this import at the beginning of your Flask application script

# Route for collecting input to adjust sleep duration

@app.route('/execute_personalization')
def execute_personalization():
    global sleep_duration

    # Load models and initialize variables
    vgg_model_path = "VGG_RNR_model.h5"
    efficientnet_model_path = 'efficientnet_custom_c_n_model_MRL.h5'
    cap = cv2.VideoCapture(0)
    total_personalization_time = 60  
    batch_duration = 10
    max_open_eye_count = 0  # Initialize max_open_eye_count
    max_closed_eye_count = 0  # Initialize max_closed_eye_count

    start_personalization_time = time.time()

    def generate_messages():
        max_open_eye_count = 0  # Initialize max_open_eye_count
        max_closed_eye_count = 0  # Initialize max_closed_eye_count

        while time.time() - start_personalization_time < total_personalization_time:
            start_batch_time = time.time()

            open_eye_counts_batch = []
            closed_eye_counts_batch = []

            # Introduce delay based on sleep duration
            time.sleep(sleep_duration)

            while time.time() - start_batch_time < batch_duration:
                ret, frame = cap.read()

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

                faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
                for (x, y, w, h) in faces:
                    roi_gray = gray_frame[y:y+h, x:x+w]
                    eyes = eye_cascade.detectMultiScale(roi_gray)
                    for (ex, ey, ew, eh) in eyes:
                        eye_region = frame[y+ey:y+ey+eh, x+ex:x+ex+ew]

                        # Preprocess eye image
                        preprocessed_eye = preprocess_image(eye_region, target_shape=(224, 224))

                        # Predictions with VGG model
                        input_data_vgg = np.expand_dims(preprocessed_eye, axis=0)
                        prediction_vgg = vgg_model.predict(input_data_vgg)
                        final_prediction_vgg = 1 if prediction_vgg[0, 1] > 0.50 else 0

                        # Predictions with EfficientNet model based on VGG predictions
                        if final_prediction_vgg == 0:
                            input_data_efficientnet = np.expand_dims(preprocessed_eye, axis=0)
                            prediction_efficientnet = efficientnet_model.predict(input_data_efficientnet)
                        else:
                            filtered_eye = color_normalization(preprocessed_eye)
                            input_data_efficientnet = np.expand_dims(filtered_eye, axis=0)
                            prediction_efficientnet = efficientnet_model.predict(input_data_efficientnet)

                        # Handle predictions from EfficientNet model
                        final_prediction_efficientnet = 1 if prediction_efficientnet[0, 1] > 0.5 else 0
                        if final_prediction_efficientnet == 0:
                            closed_eye_counts_batch.append(1) # Update closed eye count
                        elif final_prediction_efficientnet == 1:
                            open_eye_counts_batch.append(1) # Update open eye count

            # Update max open and closed eye counts
            if open_eye_counts_batch:
                max_open_eye_count = max(max_open_eye_count, Counter(open_eye_counts_batch).most_common(1)[0][1])

            if closed_eye_counts_batch:
                max_closed_eye_count = max(max_closed_eye_count, Counter(closed_eye_counts_batch).most_common(1)[0][1])

            yield f"data: Personalization batch: Closed eye count: {len(closed_eye_counts_batch)}, Open eye count: {len(open_eye_counts_batch)}\n\n"

        yield f"data: Max Open eye count: {max_open_eye_count}, Max Closed eye count: {max_closed_eye_count}\n\n"
        yield "data: Personalization phase completed. Starting monitoring phase...\n\n"

    return Response(generate_messages(), mimetype='text/event-stream')

@app.route('/input_collection', methods=['GET', 'POST'])
def input_collection():
    global prev_sleep_hours, working_hours, medication_taken, sleep_duration

    if request.method == 'POST':
        # Collect input data
        prev_sleep_hours = int(request.form.get('prev_sleep_hours'))
        working_hours = int(request.form.get('working_hours'))
        medication_taken = request.form.get('medication_taken', '').lower() == 'yes'

        # Adjust sleep duration based on input data
        sleep_duration = adjust_sleep_duration(prev_sleep_hours, working_hours, medication_taken)

        # Redirect to the personalization phase
        return redirect(url_for('execute_personalization'))

    # If the request method is GET (initial render of the form), initialize input values
    prev_sleep_hours = 0
    working_hours = 0
    medication_taken = False
    sleep_duration = 0

    # Render the template for collecting input
    return render_template('input_collection.html', prev_sleep_hours=prev_sleep_hours, working_hours=working_hours, medication_taken=medication_taken, sleep_duration=sleep_duration)

# Define the function to adjust sleep duration
def adjust_sleep_duration(prev_sleep_hours, working_hours, medication_taken):
    if prev_sleep_hours >= 6 and 8 <= working_hours <= 12 and not medication_taken:
        return 0.50
    else:
        return 0.30

if __name__ == '__main__':
    app.run(debug=True)