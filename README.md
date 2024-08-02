# Facial Recognition Project

Created by: [Jake Phelan](https://github.com/jakephelan1)

![0E5785E6-837E-4EDA-9B51-6F97BC3A33A7_1_201_a](https://github.com/user-attachments/assets/1dfaec45-f881-4fd1-bcb0-dbe9b24e2051)

## Project Description
This project implements a facial recognition system using my custom machine learning library. It includes functionality for training a Convolutional Neural Network (CNN) on a dataset of facial images, as well as real-time face detection and recognition using a webcam feed.

## Features
- Trains a custom CNN model on a dataset of facial images
- Implements real-time face detection using OpenCV
- Performs facial recognition on detected faces
- Displays recognition results with confidence scores
- Supports data augmentation for improved model performance
- Includes dropout layers for regularization
- Implements early stopping and learning rate reduction for efficient training

## Tools Used
- **Python:** Primary programming language
- **NumPy:** Numerical computing and array operations
- **SciPy:** Matrix calculations for forward and backward passes 
- **OpenCV (cv2):** Image processing and computer vision tasks
- **dlib:** Face detection and facial landmark prediction
- **scikit-learn:** Used for train-test split
- **joblib:** Model serialization and deserialization
- **imutils:** Convenience functions for OpenCV operations
- **rembg:** Background removal from images

## Setup and Installation
1. **Clone the Repository**
   ```bash
   git clone [Your Repository URL]
   cd [Your Repository Name]
   ```

2. **Create a Virtual Environment**
   ```bash
   python3 -m venv myenv
   source myenv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Required Data**
   - Download the shape predictor file if not already downloaded: `shape_predictor_68_face_landmarks.dat`
   - Place it in the root directory of the project

5. **Prepare Your Dataset**
   - Organize your facial images in the `data` directory
   - Each person should have their own subdirectory named after them
   - For efficient recognition, the data should include 150-200 images of the persons face at different angles/lighting

6. **Train the Model**
   ```bash
   python convolutional_neural_net.py
   ```

7. **Run the Face Recognition System**
   ```bash
   python face_recognition_system.py
   ```

## Usage
After running the face recognition system, it will access your webcam and start detecting and recognizing faces in real-time. The recognized faces will be displayed with bounding boxes and labels showing the predicted name and confidence score.

To exit the program, press 'q' while the webcam window is active.

## Note
This project is for educational purposes only. Ensure you have the right to use and process any facial images in your dataset.
