from natorch.nn import init
import numpy as np
from typing import Any, Iterator
from natorch.nn.parameter import Parameter
from natorch.nn.init import xavier_normal_, xavier_uniform_, kaiming_normal_, kaiming_uniform_, ones_, zeros_, constants_

class Module: 

    def __init__(self, dim_in = None, dim_out = None, initialization = None):
        self.dim_in = dim_in
        self.dim_out = dim_out
        
        self._caches = {}      # store forward caches, backward_caches
        self._parameters = {}  # 'name' : value, while name in [weights, bias]
        
        self.initialization = self._initialization_check(initialization=initialization)
    
    def _initialization_check(self, initialization):
        _valid = ['xavier_normal', 'xavier_uniform', 'kaiming_normal', 'kaiming_uniform', None, 'random']
        if initialization in _valid:
            return initialization
        raise ValueError(f"Unknow initialization {initialization}")

    def initialize_params(self):
        raise NotImplementedError("Each Module must implement initialize_params()")

    def parameters(self):
        return self._parameters

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Each Module must implement forward()")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def backward(self, *args, **kwargs):
        raise NotImplementedError("Each Module must implement backward()")
