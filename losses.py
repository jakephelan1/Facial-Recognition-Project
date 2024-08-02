import numpy as np
from keras.utils import to_categorical

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

def bianry_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)

def categorical_cross_entropy(y_true, y_pred):
    if y_true.ndim == 1:
        y_true = to_categorical(y_true, num_classes=y_pred.shape[1])
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(1, -1)
    
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

def categorical_cross_entropy_prime(y_true, y_pred):
    if y_true.ndim == 1:
        y_true = to_categorical(y_true, num_classes=y_pred.shape[1])
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(1, -1)
    
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    
    return (y_pred - y_true) / y_pred.shape[0]