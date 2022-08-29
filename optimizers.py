import numpy as np
from tensors import *


class Optimizer:
    def __init__(self, params) -> None:
        assert isinstance(params, dict), "params must be of a dictionary of Parameters"
        self.parameters = params
    
    def step(self, lr):
        for p in self.parameters.values():
            if p.weights is not None and isinstance(p.weights, Tensor) and p.weights.requires_grad:
                p.weights -= lr * p.weights.grad
    
    def zero_grad(self):
        for p in self.parameters.values():
            if p.weights is not None and isinstance(p.weights, Tensor) and p.weights.requires_grad:
                p.weights.grad = np.zeros(p.weights.shape)
