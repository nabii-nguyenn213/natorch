import numpy as np
from numba import njit, prange
from natorch.nn.parameter import Parameter
from natorch.nn.modules.module import Module

# MAX POOL -------------------------------------------------------------------------------------------------------------------- #

@njit(parallel=True)
def _maxpool2d_forward_numba(x, kernel_size, stride):
    batch, channels, H, W = x.shape
    K = kernel_size
    S = stride
    H_out = (H - K) // S + 1
    W_out = (W - K) // S + 1
    out = np.empty((batch, channels, H_out, W_out), dtype=x.dtype)
    for b in prange(batch):
        for c in range(channels):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * S
                    w_start = j * S
                    # find max in the window
                    max_val = x[b, c, h_start, w_start]
                    for di in range(K):
                        for dj in range(K):
                            val = x[b, c, h_start + di, w_start + dj]
                            if val > max_val:
                                max_val = val
                    out[b, c, i, j] = max_val
    return out

@njit(parallel=True)
def _maxpool2d_backward_numba(x, grad_out, kernel_size, stride):
    batch, channels, H, W = x.shape
    K = kernel_size
    S = stride
    H_out = grad_out.shape[2]
    W_out = grad_out.shape[3]
    grad_input = np.zeros_like(x)
    for b in prange(batch):
        for c in range(channels):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * S
                    w_start = j * S
                    # identify max location
                    max_val = x[b, c, h_start, w_start]
                    max_i, max_j = 0, 0
                    for di in range(K):
                        for dj in range(K):
                            val = x[b, c, h_start + di, w_start + dj]
                            if val > max_val:
                                max_val = val
                                max_i, max_j = di, dj
                    grad_input[b, c, h_start + max_i, w_start + max_j] = grad_out[b, c, i, j]
    return grad_input

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

@njit(parallel=True)
def _avgpool2d_forward(x, kernel_size, stride):
    batch, channels, H, W = x.shape
    K, S = kernel_size, stride
    H_out = (H - K)//S + 1
    W_out = (W - K)//S + 1
    out = np.empty((batch, channels, H_out, W_out), dtype=x.dtype)
    for b in prange(batch):
        for c in range(channels):
            for i in range(H_out):
                for j in range(W_out):
                    s = 0.0
                    for di in range(K):
                        for dj in range(K):
                            s += x[b, c, i*S+di, j*S+dj]
                    out[b, c, i, j] = s/(K*K)
    return out

@njit(parallel=True)
def _avgpool2d_backward(x, grad_out, kernel_size, stride):
    batch, channels, H, W = x.shape
    K, S = kernel_size, stride
    H_out, W_out = grad_out.shape[2], grad_out.shape[3]
    grad_input = np.zeros_like(x)
    inv = 1.0/(K*K)
    for b in prange(batch):
        for c in range(channels):
            for i in range(H_out):
                for j in range(W_out):
                    g = grad_out[b, c, i, j] * inv
                    for di in range(K):
                        for dj in range(K):
                            grad_input[b, c, i*S+di, j*S+dj] += g
    return grad_input

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
