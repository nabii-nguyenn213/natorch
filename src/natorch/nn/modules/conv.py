import numpy as np
from numba import njit, prange
from natorch.nn import init
from natorch.nn.parameter import Parameter
from natorch.nn.modules.module import Module
from natorch.nn.init import kaiming_normal_, kaiming_uniform_
from natorch.nn.init import constants_, ones_, zeros_
from natorch.nn.init import xavier_normal_, xavier_uniform_, random_
from natorch.nn.init import _calculate_gain, _calculate_fans, _check_param
from natorch.nn.functional import _conv2d_backward_numba, _conv2d_forward_numba, _conv_transpose2d_backward_numba, _conv_transpose2d_forward_numba

class Conv2d(Module):
    
    def __init__(self, in_channels : int, out_channels : int, kernel_size : int, stride : int = 1, padding : int = 0, initialization =None):
        super().__init__(initialization=initialization)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.nonlinearity = 'linear'
        self.weights, self.bias = self.initialize_params()
        self._parameters['weights'] = self.weights
        self._parameters['bias'] = self.bias

    def initialize_params(self):
        weights = Parameter(shape=(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size), requires_grad=True)
        bias = Parameter(shape = (self.out_channels, ), requires_grad=True)
        
        if self.initialization is None or self.initialization == 'random':     
            weights = random_(param=weights)
        elif self.initialization == 'xavier_normal':
            gain = _calculate_gain(nonlinearity=self.nonlinearity)
            weights = xavier_normal_(param=weights, gain=gain)
        elif self.initialization == 'xavier_uniform':
            gain = _calculate_gain(nonlinearity=self.nonlinearity)
            weights = xavier_uniform_(param=weights, gain=gain)
        elif self.initialization == 'kaiming_normal':
            weights = kaiming_normal_(param=weights, mode='fan_in', nonlinearity=self.nonlinearity, negative_slope=0.1)
        elif self.initialization == 'kaiming_uniform':
            weights = kaiming_uniform_(param=weights, mode='fan_in', nonlinearity=self.nonlinearity, negative_slope=0.1)
        else: 
            raise ValueError(f"Unknow initialization {self.initialization}")
        bias = zeros_(param=bias)

        return weights, bias

    def _update_params(self):
        self._parameters['weights'] = self.weights
        self._parameters['bias'] = self.bias

    def forward(self, x):
        '''
        Parameter : 
            x shape       = (Batch, In_channels, Height, Width)
            weights shape = (Out_channels, In channels, Kernel_size, Kernel_size)
            bias shape    = (Out_channels, )
        Return : 
            Output shape  = (Batch, Out_channels, Height_out, Width_out)
        '''
        if not hasattr(self, 'weights') or not hasattr(self, 'bias'):
            self.weights, self.bias = self.initialize_params()
            self._parameters['weights'] = self.weights
            self._parameters['bias'] = self.bias
        self._caches['input'] = x
        return _conv2d_forward_numba(x=x, weight=self.weights.data, bias=self.bias.data, stride = self.stride, padding=self.padding, kernel_size=self.kernel_size, 
                                     in_channels=self.in_channels, out_channels=self.out_channels)

    def backward(self, grad_out):
        grad_input, grad_weight, grad_bias = _conv2d_backward_numba(
            self._caches['input'],
            self.weights.data,
            grad_out,
            self.stride,
            self.padding,
            self.kernel_size,
            self.in_channels,
            self.out_channels
        )
        self.weights.grad = grad_weight
        self.bias.grad    = grad_bias
        return grad_input

class ConvTranspose2d(Module):
    def __init__(
        self, in_channels, out_channels,
        kernel_size, stride=1, padding=0,
        output_padding=0, initialization=None
    ):
        super().__init__(initialization=initialization)
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.kernel_size    = kernel_size
        self.stride         = stride
        self.padding        = padding
        self.output_padding = output_padding
        self.nonlinearity   = 'linear'

        self.weights, self.bias = self.initialize_params()
        self._parameters['weights'] = self.weights
        self._parameters['bias']    = self.bias

    def initialize_params(self):
        weights = Parameter(shape=(self.in_channels, self.out_channels, self.kernel_size, self.kernel_size), requires_grad=True)
        bias = Parameter(shape = (self.out_channels, ), requires_grad=True)
        
        if self.initialization is None or self.initialization == 'random':     
            weights = random_(param=weights)
        elif self.initialization == 'xavier_normal':
            gain = _calculate_gain(nonlinearity=self.nonlinearity)
            weights = xavier_normal_(param=weights, gain=gain)
        elif self.initialization == 'xavier_uniform':
            gain = _calculate_gain(nonlinearity=self.nonlinearity)
            weights = xavier_uniform_(param=weights, gain=gain)
        elif self.initialization == 'kaiming_normal':
            weights = kaiming_normal_(param=weights, mode='fan_in', nonlinearity=self.nonlinearity, negative_slope=0.1)
        elif self.initialization == 'kaiming_uniform':
            weights = kaiming_uniform_(param=weights, mode='fan_in', nonlinearity=self.nonlinearity, negative_slope=0.1)
        else: 
            raise ValueError(f"Unknow initialization {self.initialization}")
        bias = zeros_(param=bias)

        return weights, bias

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._caches['input'] = x
        return _conv_transpose2d_forward_numba(
            x,
            self.weights.data,
            self.bias.data,
            self.stride,
            self.padding,
            self.output_padding
        )

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        x = self._caches['input']
        grad_input, gw, gb = _conv_transpose2d_backward_numba(
            x,
            self.weights.data,
            grad_out,
            self.stride,
            self.padding,
            self.output_padding
        )
        self.weights.grad = gw
        self.bias.grad    = gb
        return grad_input
