from tensors import *
from utils import *

class Loss:
    def __init__(self, forward_func, backward_func) -> None:
        self.forward_func = forward_func
        self.backward_func = backward_func
    
    def forward(self, predictions, actual):
        self.predictions = predictions
        self.actual = actual
        self.loss = self.forward_func(self.predictions, self.actual)
        return self.loss
    
    def backward(self):
        self.grad = self.backward_func(self.predictions, self.actual)
        self.predictions.backward(self.grad)

    def __call__(self, predictions, actual):
        return self.forward(predictions, actual)
    
class MSE(Loss):
    def __init__(self) -> None:
        super().__init__(forward_func=forward_mse, backward_func=backward_mse)
