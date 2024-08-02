import numpy as np
from keras.utils import to_categorical
import joblib
from layers import Dense, Convolutional, Flatten, Dropout, MaxPooling
from activations import ReLU, Softmax
from losses import categorical_cross_entropy, categorical_cross_entropy_prime
from sklearn.model_selection import train_test_split
import logging
import os
import cv2
from scipy.ndimage import rotate
from rembg import remove
from imutils import face_utils
import dlib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

def remove_background_with_rembg(image):
    return remove(image)

def align_face(image, left_eye, right_eye):
    left_eye = np.array(left_eye).astype(int)
    right_eye = np.array(right_eye).astype(int)

    eye_center = ((left_eye[0] + right_eye[0]) // 2,
                  (left_eye[1] + right_eye[1]) // 2)


    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))

    desired_right_eye_x = 1.0 - 0.35 

    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desired_dist = (desired_right_eye_x - 0.35)
    desired_dist *= 96  
    scale = desired_dist / dist

    eye_center_float = (float(eye_center[0]), float(eye_center[1]))
    try:
        M = cv2.getRotationMatrix2D(eye_center_float, angle, scale)
    except Exception as e:
        print(f"Error in cv2.getRotationMatrix2D: {str(e)}")
        print(f"eye_center_float: {eye_center_float}, angle: {angle}, scale: {scale}")
        raise

    tX = 96 * 0.5
    tY = 96 * 0.35
    M[0, 2] += (tX - eye_center[0])
    M[1, 2] += (tY - eye_center[1])

    output = cv2.warpAffine(image, M, (96, 96), flags=cv2.INTER_CUBIC)

    return output

def preprocess_image(image, target_size=(96, 96)):
    if isinstance(image, str):
        img = cv2.imread(image)
    elif len(image.shape) == 2:
        img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img = image.copy()
    
    if img is None:
        raise ValueError(f"Unable to read or process image")

    img_bg_removed = remove_background_with_rembg(img)


    shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(shape_predictor_path):
        gray = cv2.cvtColor(img_bg_removed, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(gray, target_size)
    else:
        try:
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor(shape_predictor_path)

            gray = cv2.cvtColor(img_bg_removed, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)

            if len(rects) == 0:
                print("No face detected, using the whole image")
                face = cv2.resize(gray, target_size)
            else:
                rect = rects[0]

                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                left_eye = shape[36:42].mean(axis=0).astype("int")
                right_eye = shape[42:48].mean(axis=0).astype("int")
                
                face = align_face(gray, left_eye, right_eye)
        except Exception as e:
            print(f"Face alignment failed: {str(e)}")
            print("Using fallback method: resizing whole image")
            face = cv2.resize(gray, target_size)

    normalized = face.astype(np.float32) / 255.0

    return normalized.reshape(1, *target_size)

def preprocess_image_for_tracking(image, target_size=(96, 96)):
    if isinstance(image, str):
        img = cv2.imread(image)
    elif len(image.shape) == 2:
        img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img = image.copy()
    
    if img is None:
        raise ValueError(f"Unable to read or process image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, target_size)

    normalized = face.astype(np.float32) / 255.0

    return normalized.reshape(1, *target_size)

def preprocess_facial_data(data_dir):
    images = []
    labels = []
    label_dict = {}
    
    valid_subdirs = [d for d in sorted(os.listdir(data_dir)) if os.path.isdir(os.path.join(data_dir, d))]
    
    for label, person_name in enumerate(valid_subdirs):
        person_dir = os.path.join(data_dir, person_name)
        label_dict[label] = person_name
        person_images = 0
        
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            
            if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    processed_image = preprocess_image(image_path)
                    images.append(processed_image)
                    labels.append(label)
                    person_images += 1
                    
                    print(f"{person_images} images preprocessed for {person_name}")
                        
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")
        
        print(f"Processed {person_images} images for {person_name}")
    
    if not images:
        print("No images were processed.")
        return None, None, None

    try:
        images = np.array(images)
    except Exception as e:
        print(f"Error converting images to numpy array: {str(e)}")
        return None, None, None

    if images.shape[1:] not in [(96, 96), (1, 96, 96)]:
        print(f"Unexpected image shape: {images.shape[1:]}. Expected (96, 96) or (1, 96, 96)")
        return None, None, None

    if images.shape[1:] == (96, 96):
        images = images.reshape(-1, 1, 96, 96)

    labels = np.array(labels)
    
    num_classes = len(label_dict)
    labels_one_hot = to_categorical(labels, num_classes)
    
    if len(images) != len(labels_one_hot):
        print("Mismatch between number of images and labels")
        return None, None, None
    
    return images, labels_one_hot, label_dict

def create_network(num_classes):
    return [
        Convolutional((1, 96, 96), 3, 64),    
        ReLU(),
        MaxPooling(2, 2),                     
        Convolutional((64, 47, 47), 3, 128),  
        ReLU(),
        MaxPooling(2, 2),                     
        Convolutional((128, 22, 22), 3, 256), 
        ReLU(),
        Convolutional((256, 20, 20), 3, 256), 
        ReLU(),
        MaxPooling(2, 2),                     
        Convolutional((256, 9, 9), 3, 512),   
        ReLU(),
        Convolutional((512, 7, 7), 3, 512),   
        ReLU(),
        MaxPooling(2, 2),                     
        Flatten(),                            
        Dense(2048, 1024),
        ReLU(),
        Dropout(0.5),
        Dense(1024, 512),
        ReLU(),
        Dropout(0.5),
        Dense(512, num_classes),
        Softmax()
    ]

def clip_and_apply_gradients(network, learning_rate, max_norm=1.0):
    gradients = []
    for layer in network:
        if isinstance(layer, Convolutional):
            gradients.extend([layer.kernels_gradient, layer.biases_gradient])
        elif isinstance(layer, Dense):
            gradients.extend([layer.weights_gradient, layer.bias_gradient])

    total_norm = np.sqrt(sum(np.sum(np.square(grad)) for grad in gradients if grad is not None))
    clip_coef = min(max_norm / total_norm + 1e-6, 1.0) if total_norm > max_norm else 1.0

    for layer in network:
        if hasattr(layer, 'apply_gradients'):
            layer.apply_gradients(learning_rate * clip_coef)

def reduce_lr_on_plateau(current_lr, val_acc_history, factor=0.5, patience=3, min_lr=1e-6):
    if len(val_acc_history) <= patience:
        return current_lr

    if val_acc_history[-1] > max(val_acc_history[:-1]):
        logging.info(f"Validation accuracy improved. Maintaining learning rate at {current_lr:.6f}")
        return current_lr
  
    best_epoch = val_acc_history.index(max(val_acc_history))
    epochs_since_improvement = len(val_acc_history) - 1 - best_epoch

    if epochs_since_improvement >= patience:
        new_lr = max(current_lr * factor, min_lr)
        logging.info(f"Validation accuracy not improving for {patience} epochs. Reducing learning rate from {current_lr:.6f} to {new_lr:.6f}")
        return new_lr
    else:
        return current_lr

def predict(network, x_batch):
    for layer in network:
        if hasattr(layer, 'set_training_mode') and layer.training == True:
            layer.set_training_mode(False)
    
    output = x_batch
    for layer in network:
        output = layer.forward(output)
    return np.argmax(output, axis=1)

def predict_proba(network, image):
    if isinstance(image, str) or (isinstance(image, np.ndarray) and image.shape != (1, 1, 96, 96)):
        image = preprocess_image(image)
    
    for layer in network:
        if hasattr(layer, 'set_training_mode') and layer.training == True:
            layer.set_training_mode(False)
    
    output = image
    for layer in network:
        output = layer.forward(output)
    return output[0]

def evaluate(network, x_test, y_test, batch_size):
    correct_predictions = 0
    for i in range(0, len(x_test), batch_size):
        x_batch = x_test[i:i + batch_size]
        y_batch = y_test[i:i + batch_size]
        predictions = predict(network, x_batch)
        correct_predictions += np.sum(predictions == np.argmax(y_batch, axis=1))
    accuracy = correct_predictions / len(x_test)
    logging.info(f"Accuracy on test set: {accuracy * 100:.2f}%")


def augment_image(image):
    angle = np.random.uniform(-15, 15)
    image = rotate(image.squeeze(), angle, reshape=False)
    
    image = image * (0.8 + np.random.rand() * 0.4)
    
    mean = np.mean(image)
    image = (image - mean) * (0.8 + np.random.rand() * 0.4) + mean
    
    noise = np.random.normal(0, 0.05, image.shape)
    image = image + noise
    
    return np.clip(image, 0, 1)

def create_augmented_batch(x_batch, y_batch, augment_ratio=0.5):
    batch_size = len(x_batch)
    num_augment = int(batch_size * augment_ratio)
    
    x_aug = np.array([augment_image(img) for img in x_batch[:num_augment]])
    x_aug = x_aug.reshape(-1, 1, 96, 96)
    
    x_mixed = np.concatenate([x_aug, x_batch[num_augment:]], axis=0)
    
    return x_mixed, y_batch

def train(network, x_train, y_train, epochs, batch_size, learning_rate, x_val, y_val, lr_patience=3, early_stopping_patience=7):
    val_acc_history = []
    best_val_acc = 0
    epochs_without_improvement = 0

    for e in range(epochs):
        epoch_error = 0
        batch_count = 0
        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            
            x_augmented, y_augmented = create_augmented_batch(x_batch, y_batch)
            
            output = x_augmented
            for layer in network:
                if hasattr(layer, 'set_training_mode'):
                    layer.set_training_mode(True)
                output = layer.forward(output)
            
            batch_error = categorical_cross_entropy(y_augmented, output)
            epoch_error += batch_error
            batch_count += 1
            
            grad = categorical_cross_entropy_prime(y_augmented, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)
            
            clip_and_apply_gradients(network, learning_rate)
            
            print(f"Epoch {e + 1}/{epochs}, Batch {batch_count}, Error: {batch_error:.6f}")
        
        average_epoch_error = epoch_error / batch_count

        val_predictions = predict(network, x_val)
        val_accuracy = np.mean(val_predictions == np.argmax(y_val, axis=1))
        val_acc_history.append(val_accuracy)

        logging.info(f"Epoch {e + 1}/{epochs}, Average Error: {average_epoch_error:.6f}, Validation Accuracy: {val_accuracy:.4f}, lr = {learning_rate:.6f}")
        
        if np.isnan(average_epoch_error):
            print("NaN error encountered. Stopping training.")
            break

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered. No improvement for {early_stopping_patience} epochs.")
            break

        learning_rate = reduce_lr_on_plateau(learning_rate, val_acc_history, patience=lr_patience)

    return network, val_acc_history

def main():
    data_dir = 'data'
    result = preprocess_facial_data(data_dir)

    if result[0] is None:
        print("Error in preprocessing data. Exiting.")
        exit()

    x, y, label_dict = result
    logging.info("Preprocessing complete. Check the 'preprocessed_data' directory for new images.")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    assert y_train.shape[1] == 5, f"Unexpected shape of y_train: {y_train.shape}"
    assert y_val.shape[1] == 5, f"Unexpected shape of y_val: {y_val.shape}"
    assert y_test.shape[1] == 5, f"Unexpected shape of y_test: {y_test.shape}"

    epochs = 20  
    batch_size = 16
    learning_rate = 0.01
    network = create_network(5)

    train(network, x_train, y_train, epochs, batch_size, learning_rate, x_val, y_val)
    joblib.dump((network, label_dict), "cnn_model_and_labels.pkl")

    evaluate(network, x_test, y_test, batch_size)

if __name__ == "__main__":
    main()
