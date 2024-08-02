import numpy as np
from layers import Layer

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x) ** 2

def sigmoid(x):
    return 1 / (1 + np.exp(-x) + 1e-15)

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1-s)

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

def identity(x):
    return x

class Activation:
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))
    
    def apply_gradients(self, learning_rate):
        pass

class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)

class Sigmoid(Activation):
    def __init__(self):
        super().__init__(sigmoid, sigmoid_prime)

class ReLU(Activation):
    def __init__(self):
        super().__init__(relu, relu_prime)

class Softmax(Activation):
    def __init__(self):
        super().__init__(softmax, identity)

    def backward(self, output_gradient, learning_rate):
        return output_gradient