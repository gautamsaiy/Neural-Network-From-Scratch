import numpy as np



def forward_ReLU(input: np.ndarray, weights):
    return input * (input > 0).view(np.ndarray)

def forward_dense(input: np.ndarray, weights):
    return input @ weights

def forward_bias(input: np.ndarray, weights):
    return input + weights


def grad_matmul(a, b, grad):
    return (grad * np.ones((a.shape[0], b.shape[1]))) @ b.T, a.T @ (grad * np.ones((a.shape[0], b.shape[1])))

def grad_add(a, b, grad):
    return grad, np.sum(grad, axis=0, keepdims=True)

def grad_sub(a, b, grad):
    return grad_add(a, -b, grad)

def grad_mul(a, b, grad):
    return grad * b, grad * a

def grad_truediv(a, b, grad):
    return grad / b, -grad * a / (b ** 2)

def grad_pow(a, b, grad):
    return grad * (a ** (b - 1)) * b, grad * (b ** a) * np.log(b)

def grad_rpow(a, b, grad):
    return grad * (b ** a) * np.log(b), grad * (a ** (b - 1)) * b

def grad_gt(a, b, grad):
    return (a > b) * grad, grad * (a <= b)

def grad_sum(a, b, grad):
    return np.sum(grad, axis=0, keepdims=True), np.zeros(a.shape)
