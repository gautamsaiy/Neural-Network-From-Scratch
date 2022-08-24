import numpy as np


class Optimizer:
    def __init__(self, params) -> None:
        assert isinstance(params, dict), "params must be of a dictionary of Parameters"
        self.parameters = params
    
    def step(self, lr):
        for p in self.parameters.values():
            p.update_weights(lr)
    
    def zero_grad(self):
        for p in self.parameters.values():
            p.zero_grad()
