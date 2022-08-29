import numpy as np
from tensors import *
from utils import *


class Layer:
    def __init__(self, forward_func, input_size, output_size, layer_type="Layer") -> None:
        self.weights = None
        self.forward_func = forward_func
        self.input_size, self.output_size = input_size, output_size
        self.layer_type = layer_type
    
    def forward(self, *args):
       return self.forward_func(*args, self.weights)
    
    def update_weights(self, a):
        if self.requires_grad: self.weights -= a * self.grad

    def __call__(self, *args):
        return self.forward(*args)

    def __repr__(self) -> str:
        return f"{self.layer_type}: ({self.input_size}, {self.output_size})"


class Dense(Layer):
    def __init__(self, input_size: int, output_size: int, data=None, seed=0) -> None:
        super().__init__(forward_func=forward_dense, 
            input_size=input_size, output_size=output_size)
        self.weights = Tensor((input_size, output_size)) if data is None else Tensor(data)
    
class Bias(Layer):
    def __init__(self, size, data=None, seed=0):
        super().__init__(forward_func=forward_bias,
            input_size=size, output_size=size)
        self.weights = Tensor((1, size)) if data is None else Tensor(data)

class ReLU(Layer):
    def __init__(self) -> None:
        super().__init__(forward_func=forward_ReLU, input_size=None, output_size=None)

class Sigmoid(Layer):
    def __init__(self) -> None:
        super().__init__(forward_func=forward_sigmoid, input_size=None, output_size=None)
