import numpy as np
from natorch.nn.parameter import Parameter
from natorch.nn.modules.module import Module
from natorch.nn.init import kaiming_normal_, kaiming_uniform_
from natorch.nn.init import constants_, ones_, zeros_, random_
from natorch.nn.init import xavier_normal_, xavier_uniform_
from natorch.nn.init import _calculate_gain, _calculate_fans, _check_param

class Dense(Module) : 

    def __init__(self, dim_in, dim_out, initialization = None):
        super().__init__(dim_in, dim_out, initialization)
        self.nonlinearity = 'linear'
        self._check_dim()
        self.weights, self.bias = self.initialize_params()
        self._parameters['weights'] = self.weights
        self._parameters['bias'] = self.bias

    def _check_dim(self):
        if not isinstance(self.dim_in, int) or not isinstance(self.dim_out, int):
            raise TypeError(f"Dense's Dimension must be integer.")
        if self.dim_in <= 0 or self.dim_out <=0:
            raise ValueError(f"Dense's Dimension must be positive integer.")

    def initialize_params(self):
        weights = Parameter(shape=(self.dim_in, self.dim_out), requires_grad=True)
        bias = Parameter(shape=(self.dim_out, ), requires_grad=True)

        # Initialization
        if self.initialization is None or self.initialization == 'random': 
            weighst = random_(param=weights)
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
            x shape       = (D_i, )       where D_i is the number of input features
            weights shape = (D_i, D_o)    where D_o is the number of output features
            bias shape    = (D_o, )
        ---
        Return : 
            Linear_Combination = x @ weights + bias
                               = (D_i, ) @ (D_i, D_o) + (D_o, ) 
                               = (D_o, )
        '''
        if not hasattr(self, 'weights') or not hasattr(self, 'bias'):
            self.weights, self.bias = self.initialize_params()
            self._parameters['weights'] = self.weights
            self._parameters['bias'] = self.bias

        self._caches['input'] = x
        linear_comb = x @ self.weights.data + self.bias.data
        return linear_comb

    def backward(self, grad_out):
        x = self._caches['input']    
        
        grad_w = x.T @ grad_out       
        self.weights.grad += grad_w

        self.bias.grad += np.sum(grad_out, axis=0)
        grad_input = grad_out @ self.weights.data.T

        return grad_input
