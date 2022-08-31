import numpy as np
from tensors import *


class Optimizer:
    def __init__(self, params) -> None:
        self.parameters = params

    def zero_grad(self):
        for p in self.parameters.values():
            if p.weights is not None and isinstance(p.weights, Tensor) and p.weights.requires_grad:
                p.weights.grad = np.zeros(p.weights.shape)

    def step(self, magnitude):
        for p in self.parameters.values():
            if p.weights is not None and isinstance(p.weights, Tensor) and p.weights.requires_grad:
                p.weights -= magnitude * p.weights.grad


class SGD(Optimizer):
    def __init__(self, params, lr) -> None:
        super().__init__(params)
        self.lr = lr

    def step(self):
        return super().step(self.lr)

class Adam(Optimizer):
    def __init__(self, params, lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0) -> None:
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {n: 0 for n in self.parameters.keys()}
        self.v = {n: 0 for n in self.parameters.keys()}
        self.v_max = 0
        self.iteration = 0

    def step(self):
        self.iteration += 1
        for n, p in self.parameters.items():
            if p.weights is not None and isinstance(p.weights, Tensor) and p.weights.requires_grad:
                self.m[n] = self.beta1 * self.m[n] + (1 - self.beta1) * p.weights.grad
                self.v[n] = self.beta2 * self.v[n] + (1 - self.beta2) * p.weights.grad ** 2
                m_hat = self.m[n] / (1 - self.beta1 ** self.iteration)
                v_hat = self.v[n] / (1 - self.beta2 ** self.iteration)
                p.weights -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
