from scipy.signal import correlate2d
import numpy as np

class Layer:
    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError

    def apply_gradients(self, learning_rate):
        raise NotImplementedError
    
class Flatten(Layer):
    def forward(self, input):
        self.input_shape = input.shape
        return input.reshape(input.shape[0], -1)
    
    def backward(self, output_gradient, learning_rate):
        return output_gradient.reshape(self.input_shape)

    def apply_gradients(self, learning_rate):
        pass

class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape) * np.sqrt(2. / (kernel_size * kernel_size * input_depth))
        self.biases = np.zeros((depth, 1, 1))

    def forward(self, input):
        self.input = input
        batch_size = input.shape[0]
        self.output = np.zeros((batch_size, *self.output_shape))
        
        for b in range(batch_size):
            for i in range(self.depth):
                for j in range(self.input_depth):
                    self.output[b, i] += correlate2d(self.input[b, j], self.kernels[i, j], 'valid')
                self.output[b, i] += self.biases[i]
        
        return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input.shape)
        biases_gradient = np.zeros(self.biases.shape)

        batch_size = self.input.shape[0]
        for b in range(batch_size):
            for i in range(self.depth):
                for j in range(self.input_depth):
                    kernels_gradient[i, j] += correlate2d(self.input[b, j], output_gradient[b, i], 'valid')
                    input_gradient[b, j] += correlate2d(output_gradient[b, i], self.kernels[i, j], 'full')
                biases_gradient[i] += np.sum(output_gradient[b, i])

        self.kernels_gradient = kernels_gradient / batch_size
        self.biases_gradient = biases_gradient / batch_size
        return input_gradient

    def apply_gradients(self, learning_rate):
        self.kernels -= learning_rate * self.kernels_gradient
        self.biases -= learning_rate * self.biases_gradient
    
class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.bias = np.zeros((1, output_size))

    def forward(self, input):
        self.input = input
        return np.dot(self.input, self.weights) + self.bias
    
    def backward(self, output_gradient, learning_rate):
        self.weights_gradient = np.dot(self.input.T, output_gradient)
        self.bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        return np.dot(output_gradient, self.weights.T)

    def apply_gradients(self, learning_rate):
        self.weights -= learning_rate * self.weights_gradient
        self.bias -= learning_rate * self.bias_gradient
    
class Dropout(Layer):
    def __init__(self, rate):
        self.rate = rate
        self.scale = 1 / (1 - rate)
        self.training = True
    
    def forward(self, input):
        if self.training:
            self.mask = np.random.binomial(1, 1 - self.rate, size=input.shape) * self.scale
            return input * self.mask
        else:
            return input
    
    def backward(self, output_gradient, learning_rate):
        if self.training:
            return output_gradient * self.mask
        else:
            return output_gradient

    def apply_gradients(self, learning_rate):
        pass

    def set_training_mode(self, training):
        self.training = training

class MaxPooling(Layer):
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
    
    def forward(self, input):
        self.input = input
        self.batch_size, self.c, self.h, self.w = input.shape
        self.out_h = (self.h - self.pool_size) // self.stride + 1
        self.out_w = (self.w - self.pool_size) // self.stride + 1
        
        output = np.zeros((self.batch_size, self.c, self.out_h, self.out_w))
        
        for i in range(self.out_h):
            for j in range(self.out_w):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                output[:, :, i, j] = np.max(input[:, :, h_start:h_end, w_start:w_end], axis=(2, 3))
        
        return output
    
    def backward(self, output_gradient, learning_rate):
        input_gradient = np.zeros_like(self.input)
        
        for i in range(self.out_h):
            for j in range(self.out_w):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                
                window = self.input[:, :, h_start:h_end, w_start:w_end]
                mask = (window == np.max(window, axis=(2, 3), keepdims=True))
                
                input_gradient[:, :, h_start:h_end, w_start:w_end] += \
                    mask * output_gradient[:, :, i:i+1, j:j+1]
        
        return input_gradient

    def apply_gradients(self, learning_rate):
        pass
