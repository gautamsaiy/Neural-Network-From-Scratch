import numpy as np

def softmax(arr: list, normalizationFunc=np.exp):
    if type(arr) != list and type(arr) != np.ndarray:
        raise TypeError("Input is not of type list or np.ndarray")
    normalized = normalizationFunc(arr)
    return normalized / np.sum(normalized)

def rmse(predictions, actual):
    return np.sqrt(np.mean(np.square(np.subtract(predictions, actual))))

def linear(input, weights):
    return input @ weights.T

def add(x, y):
    return x + y.T

def forward_ReLU(input: np.ndarray):
    return np.maximum(0, input)

def deriv_ReLU(input: np.ndarray):
    return (input > 0) * 1

def deriv_sigmoid(input: np.ndarray):
    sig = forward_sigmoid(input)
    return sig * (1 - sig)

def forward_sigmoid(input: np.ndarray):
    return 1 / (1 + np.exp(-input))

def GELU(input: np.ndarray):
    return .5 * input * (1 + np.tanh(np.sqrt(2/np.pi) * (input + .044715 * np.power(input, 3))))

