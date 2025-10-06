import numpy as np
from numba import njit, prange
from natorch.nn.parameter import Parameter
from natorch.nn.modules.module import Module
from natorch.nn.functional import _maxpool2d_backward_numba, _maxpool2d_forward_numba, _avgpool2d_backward, _avgpool2d_forward

# MAX POOL -------------------------------------------------------------------------------------------------------------------- #

class MaxPool2d(Module):
    
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size

    def forward(self, x):
        self._caches['input'] = x
        return _maxpool2d_forward_numba(x, self.kernel_size, self.stride)

    def backward(self, grad_out):
        return _maxpool2d_backward_numba(self._caches['input'], grad_out, self.kernel_size, self.stride)

# AVG POOL -------------------------------------------------------------------------------------------------------------------- #

class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size

    def forward(self, x):
        self._caches['input'] = x
        return _avgpool2d_forward(x, self.kernel_size, self.stride)

    def backward(self, grad_out):
        if grad_out is None:
            raise ValueError("avgpool2d_backward: received grad_out=None; "
                         "you must pass in an ndarray of shape (N,C,H,W)")
        return _avgpool2d_backward(self._caches['input'], grad_out, self.kernel_size, self.stride)
