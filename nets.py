from collections import OrderedDict
import numpy as np
from parameter import Parameter
from layers import Layer


class NeuralNet:
    def __init__(self) -> None:
        self.training = True
        self.parameters = OrderedDict()

    def forward(self, input):
        pass

    def add_parameter(self, name, param):
        assert isinstance(name, str), "name must be of type str"
        assert isinstance(param, Layer), "param must be of type Layer"
        self.parameters[name] = param

    def __setattr__(self, __name: str, __value) -> None:
        if isinstance(__value, Layer):
            self.add_parameter(__name, __value)
        else:
            super().__setattr__(__name, __value)
    
    def __getattribute__(self, __name: str):
        try:
            return super().__getattribute__(__name)
        except:
            try:
                return self.parameters[__name]
            except:
                raise AttributeError

    def __delattr__(self, __name: str) -> None:
        if __name in self.parameters.keys():
            del self.parameters[__name]
        else:
            super().__delattr__(__name)
            
    def __call__(self, input):
        return self.forward(input)
