from platform import release
from turtle import forward
import numpy as np
from utils import *

class Layer:
    def __init__(self, inputSize: int, outputSize: int, bias: bool=True, presets: dict=None, activation=ReLU) -> None:
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.activation = activation
        self._initWeights(presets, bias)
        self.layerType = 'Layer'
    
    def _initWeights(self, presetWeights: dict, bias: bool=True):
        self.bias = None
        if presetWeights != None:
            self.weights = presetWeights['weights']
            if bias: 
                self.bias = presetWeights['bias']
        else:
            sqrtk = np.sqrt(1 / self.inputSize)
            self.weights = np.random.uniform(-sqrtk, sqrtk, (self.outputSize, self.inputSize))
            if bias:
                self.bias = np.random.uniform(-sqrtk, sqrtk, self.outputSize)
        self.numWeights = np.prod(self.weights.shape)
        if bias:
            self.numWeights += np.prod(self.bias.shape)

    def forward(self, input: np.ndarray):
        pass

    def __call__(self, input: np.ndarray):
        return self.forward(input)
    
    def __str__(self) -> str:
        return "{0}(in={1}, out={2}, bias={3})".format(self.layerType, self.inputSize, self.outputSize, self.bias is not None)


class Dense(Layer):
    def __init__(self, inputSize: int, outputSize: int, bias: bool = True, presets: dict = None, activation=ReLU) -> None:
        super().__init__(inputSize, outputSize, bias, presets, activation)
        self.layerType = "Dense"

    def forward(self, input: np.ndarray):
        return self.activation(linear(input, self.weights, self.bias))
