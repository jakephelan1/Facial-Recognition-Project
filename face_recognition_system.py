import sys
import dlib
import cv2
import time
import imutils
import numpy as np
from operator import itemgetter
import math
import dnn
import joblib
from convolutional_neural_net import predict_proba, preprocess_image_for_tracking
import os
import time

def test_known_images(model, data_dir, label_dict):
    for label, person in label_dict.items():
        person_dir = os.path.join(data_dir, person)
        if os.path.isdir(person_dir):
            for img_name in os.listdir(person_dir)[:5]:  
                img_path = os.path.join(person_dir, img_name)
                img = cv2.imread(img_path)
                processed_image = preprocess_image_for_tracking(img)
                probabilities = predict_proba(model, processed_image)
                predicted_class = np.argmax(probabilities)
                confidence = probabilities[predicted_class]
                predicted_name = label_dict.get(predicted_class, "Unknown")
                print(f"Known image test - Actual: {person}, Predicted: {predicted_name}, Confidence: {confidence:.2f}")

def grabcut_foreground_extraction(image, rect):
    if image.shape[0] < 10 or image.shape[1] < 10:
        print("Image too small for GrabCut, returning original image")
        return image

    try:
        mask = np.zeros(image.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        foreground = image * mask2[:, :, np.newaxis]
        return foreground
    except cv2.error as e:
        print(f"GrabCut error: {str(e)}")
        return image

def test_known_image(model, image_path, actual_label):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    processed_img = preprocess_image_for_tracking(img)
    probabilities = predict_proba(model, processed_img)
    predicted_class = np.argmax(probabilities)
    confidence = probabilities[predicted_class]
    predicted_name = label_dict.get(predicted_class, "Unknown")
    print(f"Test Image - Actual: {actual_label}, Predicted: {predicted_name}, Confidence: {confidence:.2f}")
    print(f"Raw probabilities: {probabilities}")

def debug_print_shape(name, array):
    if array is not None:
        print(f"{name} shape: {array.shape}, dtype: {array.dtype}, min: {np.min(array)}, max: {np.max(array)}")
    else:
        print(f"{name} is None")

def add_proportional_padding(x, y, w, h, image_width, image_height, padding_scale_w, padding_scale_top, padding_scale_bottom):
    padding_w = int(w * padding_scale_w)
    padding_top = int(h * padding_scale_top)
    padding_bottom = int(h * padding_scale_bottom)

    x_new = max(x - padding_w, 0)
    y_new = max(y - padding_top, 0)
    w_new = min(w + 2 * padding_w, image_width - x_new)
    h_new = min(h + padding_top + padding_bottom, image_height - y_new)

    return x_new, y_new, w_new, h_new

print("Loading model...")
model, label_dict = joblib.load("cnn_model_and_labels.pkl")
print("Model loaded.")
print("Label Dictionary:", label_dict)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(1)

CONFIDENCE_THRESHOLD = 0.3
PADDING_SCALE_W = 0.15
PADDING_SCALE_TOP = 0.15
PADDING_SCALE_BOTTOM = 0.1

start_time = time.time()
max_run_time = 60

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(200, 200))

    for (x, y, w, h) in faces:
        x, y, w, h = add_proportional_padding(x, y, w, h, frame.shape[1], frame.shape[0], PADDING_SCALE_W, PADDING_SCALE_TOP, PADDING_SCALE_BOTTOM)
        face_image = gray[y:y+h, x:x+w]
       
        processed_face_image = preprocess_image_for_tracking(face_image)
        
        if processed_face_image.shape != (1, 1, 96, 96):
            processed_face_image = processed_face_image.reshape(1, 1, 96, 96)
        
        display_image = processed_face_image.squeeze()
        display_image = (display_image * 255).astype(np.uint8)

        cv2.imshow('Preprocessed Image', display_image)
        cv2.waitKey(1)
        
        probabilities = predict_proba(model, processed_face_image)
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]

        predicted_name = label_dict.get(predicted_class, "Unknown")
        color = (0, 255, 0) if confidence > CONFIDENCE_THRESHOLD else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        label = f"{predicted_name}: {confidence:.2f}"
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        print(f"Predicted: {predicted_name}, Confidence: {confidence:.2f}")
        print(f"Raw probabilities: {probabilities}")
        print("--------------------")

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

