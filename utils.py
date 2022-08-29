import numpy as np



def forward_ReLU(input: np.ndarray, weights):
    return input * (input > 0).view(np.ndarray)

def forward_sigmoid(input: np.ndarray, weights):
    return 1 / (1 + np.exp(-input))

def forward_dense(input: np.ndarray, weights):
    return input @ weights

def forward_bias(input: np.ndarray, weights):
    return input + weights



def forward_mse(predictions, actual):
    return np.mean((predictions - actual) ** 2 / 2) 

def backward_mse(predictions, actual):
    return (predictions - actual) / predictions.size 



def grad_matmul(a, b, grad):
    return (grad * np.ones((a.shape[0], b.shape[1]))) @ b.T, a.T @ (grad * np.ones((a.shape[0], b.shape[1])))

def grad_add(a, b, grad):
    return grad, np.sum(grad, axis=0, keepdims=True)

def grad_sub(a, b, grad):
    return grad_add(a, -b, grad)

def grad_mul(a, b, grad):
    return grad * b, grad * a

def grad_true_div(a, b, grad):
    return grad / b, -grad * a / b ** 2

def grad_pow(a, b, grad):
    return grad * b * a ** (b - 1), grad * np.log(a) * a ** b

def grad_gt(a, b, grad):
    return (a > b) * grad, grad * (a <= b)
