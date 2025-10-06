import numpy as np
from natorch.nn.modules.module import Module

class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x, training=True):
        if not training:
            return x
        mask = (np.random.rand(*x.shape) > self.p).astype(x.dtype)
        self.mask = mask
        return x * mask * (1.0/(1.0 - self.p))

    def backward(self, grad_out):
        return grad_out * self.mask * (1.0/(1.0 - self.p))
