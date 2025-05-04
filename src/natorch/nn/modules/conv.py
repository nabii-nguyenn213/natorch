import numpy as np
from numba import njit, prange
from natorch.nn import init
from natorch.nn.parameter import Parameter
from natorch.nn.modules.module import Module
from natorch.nn.init import kaiming_normal_, kaiming_uniform_
from natorch.nn.init import constants_, ones_, zeros_
from natorch.nn.init import xavier_normal_, xavier_uniform_
from natorch.nn.init import _calculate_gain, _calculate_fans, _check_param

class Conv2d(Module):
    
    def __init__(self, in_channels : int, out_channels : int, kernel_size : int, stride : int = 1, padding : int = 0, initialization =None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.nonlinearity = 'linear'

    def initialize_params(self):
        weights = Parameter(shape=(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        bias = Parameter(shape = (self.out_channels, ))
        
        if self.initialization is None or self.initialization == 'random':     
            pass
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

    def forward(self, x):
        '''
        Parameter : 
            x shape       = (Batch, In_channels, Height, Width)
            weights shape = (Out_channels, In channels, Kernel_size, Kernel_size)
            bias shape    = (Out_channels, )
        Return : 
            Output shape  = (Batch, Out_channels, Height_out, Width_out)
        '''
        pass

    def backward(self, grad_out):
        pass
