import numpy as np
from natorch.nn.modules import Module

class Flatten(Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        self._caches['input_shape'] = x.shape

        batch = x.shape[0]

        if batch == 1:
            return x.squeeze()
        return x.reshape(batch, -1)

    def backward(self, grad_out):
        return grad_out.reshape(self._caches['input_shape'])   

class UnFlatten(Module):
    
    pass
