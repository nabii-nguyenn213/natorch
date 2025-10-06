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
from natorch.nn.functional import _batchnorm2d_backward, _batchnorm2d_forward

class BatchNorm2d(Module):
    def __init__(self, in_channels : int, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.in_channels = in_channels
        self.gamma, self.beta = self.initialize_params()
        self._parameters['gamma'] = self.gamma
        self._parameters['beta'] = self.beta

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
            self.gamma.data, self.beta.data, self.eps
        )
        self.gamma.grad[:] = gg
        self.beta.grad[:]  = gb
        return grad_input
