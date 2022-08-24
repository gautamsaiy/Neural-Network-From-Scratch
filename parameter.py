import numpy as np

class Parameter(np.ndarray):
    def __new__(cls, data=None, requires_grad=True, seed=0):
        if data is None:
            data = np.empty(0).view(cls)
        elif isinstance(data, tuple):
            sqrtk = np.sqrt(data[0])
            np.random.seed(0)
            data = np.random.uniform(-sqrtk, sqrtk, data).view(cls)
        else:
            data = np.array(data).view(cls)
        data.requires_grad = requires_grad
        data.grad, data.grad_fn = None, None
        if data.requires_grad == True:
            data.grad = 0
            data.grad_fn = []
        return data

    def backward(self):
        self.__backward(self.grad_fn, 1)

    def __backward(self, grad_fn, grad):
        if len(grad_fn) <= 0:
            return
        l = grad_fn.pop()
        if isinstance(l, list):
            self.__backward(l, grad)
        else:
            grad = l.backward(grad)
            self.__backward(grad_fn, grad)

    def __array_finalize__(self, obj):
        if obj is None: return
        self.requires_grad = getattr(obj, 'requires_grad', None)
        self.grad = getattr(obj, 'grad', None)
        self.grad_fn = getattr(obj, 'grad_fn', None)
