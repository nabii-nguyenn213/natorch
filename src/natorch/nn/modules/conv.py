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
def _conv2d_forward_numba(x, weight, bias, stride, padding, kernel_size, in_channels, out_channels):
    batch, _, H, W = x.shape
    K = kernel_size
    S = stride

    # pad input
    H_p = H + 2*padding
    W_p = W + 2*padding
    x_padded = np.zeros((batch, in_channels, H_p, W_p), dtype=x.dtype)
    for b in prange(batch):
        for c in range(in_channels):
            for i in range(H):
                for j in range(W):
                    x_padded[b, c, i+padding, j+padding] = x[b, c, i, j]

    H_out = (H_p - K)//S + 1
    W_out = (W_p - K)//S + 1

    out = np.zeros((batch, out_channels, H_out, W_out), dtype=x.dtype)

    for b in prange(batch):
        for oc in range(out_channels):
            for ic in range(in_channels):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * S
                        w_start = j * S
                        # element‐wise multiply and sum
                        s = 0.0
                        for di in range(K):
                            for dj in range(K):
                                s += weight[oc, ic, di, dj] * x_padded[b, ic, h_start+di, w_start+dj]
                        out[b, oc, i, j] += s
            for i in range(H_out):
                for j in range(W_out):
                    out[b, oc, i, j] += bias[oc]

    return out

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
        weights = Parameter(shape=(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size), requires_grad=True)
        bias = Parameter(shape = (self.out_channels, ), requires_grad=True)
        
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
        if not hasattr(self, 'weights') or not hasattr(self, 'bias'):
            self.weights, self.bias = self.initialize_params()

        return _conv2d_forward_numba(x=x, weight=self.weights.data, bias=self.bias.data, stride = self.stride, padding=self.padding, kernel_size=self.kernel_size, 
                                     in_channels=self.in_channels, out_channels=self.out_channels)

    def forward_numpy(self, x):

        if not hasattr(self, 'weights') or not hasattr(self, 'bias'):
            self.weights, self.bias = self.initialize_params()

        batch, _, H, W = x.shape
        K, S, P = self.kernel_size, self.stride, self.padding

        # Pad input
        x_padded = np.pad(x,
                          ((0,0), (0,0), (P,P), (P,P)),
                          mode='constant', constant_values=0)

        # Output spatial size
        H_out = (H + 2*P - K) // S + 1
        W_out = (W + 2*P - K) // S + 1

        # Allocate output
        out = np.zeros((batch, self.out_channels, H_out, W_out))

        # Perform convolution
        for b in range(batch):
            for oc in range(self.out_channels):
                for ic in range(self.in_channels):
                    for i in range(H_out):
                        for j in range(W_out):
                            h_start = i * S
                            w_start = j * S
                            window = x_padded[b, ic,
                                              h_start:h_start+K,
                                              w_start:w_start+K]
                            out[b, oc, i, j] += np.sum(
                                window * self.weights.data[oc, ic])
                out[b, oc] += self.bias.data[oc]

        return out

    def backward(self, grad_out):
        pass
