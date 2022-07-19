import numpy as np

def softmax(arr: list, normalizationFunc=np.exp):
    if type(arr) != list and type(arr) != np.ndarray:
        raise TypeError("Input is not of type list or np.ndarray")
    normalized = normalizationFunc(arr)
    return normalized / np.sum(normalized)

def rmse(predictions, actual):
    if type(predictions) != list and type(predictions) != np.ndarray:
        raise TypeError("prediction must be of type list or np.ndarray")
    if type(actual) != list and type(actual) != np.ndarray:
        raise TypeError("actual must be of type list or np.ndarray")
    if len(predictions) != len(actual):
        raise ValueError("predictions and actual must be arrays of the same length")
    return np.sqrt(np.mean(np.square(np.subtract(predictions, actual))))

def linear(input, weights, bias=None):
    if bias is None:
        return input @ weights.T
    return input @ weights.T + bias

def ReLU(input: np.ndarray):
    return np.maximum(0, input)

def sigmoid(input: np.ndarray):
    return 1 / (1 + np.exp(input))

def GELU(input: np.ndarray):
    return .5 * input * (1 + np.tanh(np.sqrt(2/np.pi) * (input + .044715 * np.power(input, 3))))
