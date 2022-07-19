import numpy as np
from layers import *

class NeuralNet:
    def __init__(self) -> None:
        pass

    def forward(self, input):
        pass

    def back(self):
        pass

    def description():
        pass


class Sequential(NeuralNet):
    def __init__(self) -> None:
        self.layers = []
        self.inputSize = None
        self.outputSize = None
    
    def addLayer(self, layer: Layer):
        if not isinstance(layer, Layer):
            raise TypeError("Must add an object of type Layer")
        if len(self.layers) == 0:
            self.layers.append(layer)
            self.inputSize = layer.inputSize
            self.outputSize = layer.outputSize
        else:
            if self.outputSize != layer.inputSize:
                raise ValueError("Input Size of {0} must be of size {1}".format(layer, self.outputSize))
            else:
                self.layers.append(layer)
                self.outputSize = layer.outputSize

    def description(self):
        desc = "Sequential(Architecture: {0}, Input Size: {1}, Output Size: {2})".format([str(layer) for layer in self.layers], self.inputSize, self.outputSize)
        return desc

    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        return input

