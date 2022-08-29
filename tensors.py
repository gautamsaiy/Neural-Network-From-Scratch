import numpy as np
from utils import *

class Tensor(np.ndarray):
    def __new__(cls, data, source=None, requires_grad=True, grad_fn=None):
        assert data is None or isinstance(data, np.ndarray) or isinstance(data, list) or isinstance(data, tuple), "data must be numpy.ndarray, list, tuple or None"
        assert source is None or (isinstance(source, list) and (len(source) == 0 or all([isinstance(s, Tensor) for s in source]))), "source must be None or list of Data"
        assert isinstance(requires_grad, bool), "requires_grad must be bool"
        assert isinstance(requires_grad, bool), "requires_grad must be bool"
        assert grad_fn is None or isinstance(grad_fn, list), "grad_fn must be None or list"

        if data is None:
            data = np.empty(0).view(cls)
        elif isinstance(data, tuple):
            sqrtk = np.sqrt(1/data[0])
            data = np.random.uniform(-sqrtk, sqrtk, data).view(cls)
        else:
            data = np.array(data).view(cls)

        data.source = [] if source is None else source
        data.requires_grad = requires_grad
        data.grad = np.zeros(data.shape) if requires_grad else None
        data.grad_fn = grad_fn

        return data


    def backward(self, gradient=1):
        if self.requires_grad:
            self.grad += gradient
            if len(self.source) > 0:
                source_gradients = self.grad_fn(self.source[0], self.source[1], gradient)
                for s, g in zip(self.source, source_gradients):
                    if isinstance(s, Tensor): s.backward(g)


    def _change_resulting_type(self, other, new_type):
        if self.requires_grad or isinstance(other, Tensor) and other.requires_grad:
            new_type.requires_grad = True
        return new_type


    def _do_func(self, other, func, grad_fn):
        result = self._change_resulting_type(other, func(other))
        result.source += [self, other]
        if result.requires_grad:
            result.grad_fn = grad_fn
            result.grad = np.zeros(result.shape)
        return result


    def __matmul__(self, other):
        assert isinstance(other, Tensor) or isinstance(other, np.ndarray), "other must be Tensor or numpy.ndarray"
        return self._do_func(other, super().__matmul__, grad_matmul)
    

    def __add__(self, other):
        return self._do_func(other, super().__add__, grad_add)


    def __sub__(self, other):
        return self._do_func(other, super().__sub__, grad_sub)


    def __mul__(self, other):
        return self._do_func(other, super().__mul__, grad_mul)


    def __true_div__(self, other):
        return self._do_func(other, super().__true_div__, grad_true_div)


    def __pow__(self, other):
        return self._do_func(other, super().__pow__, grad_pow)


    def __gt__(self, other):
        return self._do_func(other, super().__gt__, grad_gt)


    def __array_finalize__(self, obj):
        if obj is None: return
        self.source = []
        self.requires_grad = False
        self.grad_fn = None
        self.grad = None

    def __repr__(self):
        return 'Tensor({0}, requires_grad={1})'.format(super().__str__(), self.requires_grad)
