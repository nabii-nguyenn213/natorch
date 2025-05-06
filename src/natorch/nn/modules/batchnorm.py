from math import gamma
from re import A
from types import NoneType
import numpy as np
from numba import njit, prange
from natorch.nn import init
from natorch.nn.parameter import Parameter
from natorch.nn.modules.module import Module
from natorch.nn.init import kaiming_normal_, kaiming_uniform_
from natorch.nn.init import constants_, ones_, zeros_
from natorch.nn.init import xavier_normal_, xavier_uniform_
from natorch.nn.init import _calculate_gain, _calculate_fans, _check_param


@njit(parallel=True)
def _batchnorm2d_forward(x, gamma, beta, eps):
    batch, channels, H, W = x.shape
    out = np.empty_like(x)
    mean = np.zeros(channels, dtype=x.dtype)
    var  = np.zeros(channels, dtype=x.dtype)
    N = batch * H * W

    # compute mean
    for c in prange(channels):
        s = 0.0
        for b in range(batch):
            for i in range(H):
                for j in range(W):
                    s += x[b, c, i, j]
        mean[c] = s / N

    # compute var
    for c in prange(channels):
        s = 0.0
        m = mean[c]
        for b in range(batch):
            for i in range(H):
                for j in range(W):
                    d = x[b, c, i, j] - m
                    s += d * d
        var[c] = s / N

    # normalize + scale + shift
    x_hat = np.empty_like(x)
    for b in prange(batch):
        for c in range(channels):
            inv = 1.0/np.sqrt(var[c] + eps)
            for i in range(H):
                for j in range(W):
                    x_hat[b, c, i, j] = (x[b, c, i, j] - mean[c]) * inv
                    out[b, c, i, j] = gamma[c] * x_hat[b, c, i, j] + beta[c]

    return out, x_hat, mean, var

@njit(parallel=True)
def _batchnorm2d_backward(grad_out, x_hat, mean, var, gamma, eps):
    batch, channels, H, W = grad_out.shape
    N = batch * H * W
    grad_input = np.zeros_like(grad_out)
    grad_gamma = np.zeros_like(gamma)
    grad_beta  = np.zeros_like(beta)

    # grad_gamma, grad_beta
    for c in prange(channels):
        sg = 0.0; sb = 0.0
        for b in range(batch):
            for i in range(H):
                for j in range(W):
                    go = grad_out[b, c, i, j]
                    sg += go * x_hat[b, c, i, j]
                    sb += go
        grad_gamma[c] = sg
        grad_beta[c]  = sb

    # grad_input
    for c in prange(channels):
        inv = 1.0/np.sqrt(var[c] + eps)
        sum_d = 0.0; sum_dx = 0.0
        for b in range(batch):
            for i in range(H):
                for j in range(W):
                    dh = grad_out[b, c, i, j] * gamma[c]
                    sum_d  += dh
                    sum_dx += dh * x_hat[b, c, i, j]
        for b in range(batch):
            for i in range(H):
                for j in range(W):
                    dh = grad_out[b, c, i, j] * gamma[c]
                    grad_input[b, c, i, j] = (1.0/N) * inv * (
                        N*dh - sum_d - x_hat[b, c, i, j]*sum_dx
                    )
    return grad_input, grad_gamma, grad_beta

class BatchNorm2d(Module):
    def __init__(self, in_channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.in_channels = in_channels
        self.gamma, self.beta = self.initialize_params()
        self._parameters['gamma'] = gamma
        self._parameters['beta'] = beta

    def initialize_params(self):
        gamma = Parameter(shape=(self.in_channels, ), requires_grad=True)
        beta = Parameter(shape=(self.in_channels, ), requires_grad=True)

        gamma = ones_(param = gamma)
        beta = zeros_(param = beta)

        return gamma, beta

    def _update_params(self):
        self._parameters['gamma'] = self.gamma
        self._parameters['beta'] = self.beta


    def forward(self, x):
        if self.gamma is None or self.beta is None:
            self.gamma, self.beta = self.initialize_params()
            self._parameters['gamma'] = gamma
            self._parameters['beta'] = beta

        self.input = x
        out, self.x_hat, self.mean, self.var = _batchnorm2d_forward(
            x, self.gamma.data, self.beta.data, self.eps
        )
        return out

    def backward(self, grad_out):
        grad_input, gg, gb = _batchnorm2d_backward(
            grad_out, self.x_hat, self.mean, self.var,
            self.gamma, self.eps
        )
        self.gamma.grad[:] = gg
        self.beta.grad[:]  = gb
        return grad_input
