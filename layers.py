from ast import Param
import numpy as np
from parameter import Parameter
from utils import *


class Layer:
    def __init__(self, **kwargs) -> None:
        pass
    
    def forward(self, input: np.ndarray):
        pass

    def backward(self, gradient):
        pass

    def __call__(self, input: np.ndarray):
        return self.forward(input)
    
    def update_weights(self, a):
        pass


class Dense(Layer):
    def __init__(self, inputSize: int, outputSize: int, seed=0) -> None:
        super().__init__()
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.layerType = "Dense"
        self.weights = Parameter((outputSize, inputSize), seed=seed)

    def forward(self, input: np.ndarray):
        input.grad_fn.append(self)
        return linear(input, self.weights)

    def backward(self, grad):
        self.weights.grad = self.weights.T @ grad
        return self.weights

    def zero_grad(self):
        self.weights.grad = np.zeros(self.inputSize)

    def update_weights(self, a):
        self.weights = self.weights - a * self.weights.grad


class Bias(Layer):
    def __init__(self, size, seed=0):
        super().__init__()
        self.size = size
        self.weights = Parameter((size,), seed=seed)
        self.layerType = "Bias"

    def forward(self, input: Parameter):
        input.grad_fn.append(self)
        return add(input, self.weights)
    
    def backward(self, grad):
        self.weights.grad = np.ones(self.size) * grad
        return self.weights.grad
    
    def zero_grad(self):
        self.weights.grad = np.zeros(self.size)

    def update_weights(self, a):
        self.weights = self.weights - a * self.weights.grad


class ReLU(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.input = None

    def forward(self, input: np.ndarray):
        input.grad_fn.append(self)
        self.input = input
        return forward_ReLU(input)
    
    def backward(self, gradient):
        return deriv_ReLU(self.input) * gradient

    def zero_grad(self):
        self.input = None

class Sigmoid(Layer):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        return forward_sigmoid(input)

    def backward(self, gradient):
        return deriv_sigmoid(input)
    
    def zero_grad(self):
        return
