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

    def __init__(self, *target_shape: int):
        super().__init__()
        if not target_shape:
            raise ValueError("UnFlatten requires at least one dimension")
        self.shape = tuple(target_shape)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            return x.reshape(self.shape)
        batch = x.shape[0]
        return x.reshape((batch,) + self.shape)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if grad_out.ndim == len(self.shape):
            return grad_out.reshape(-1)
        batch = grad_out.shape[0]
        return grad_out.reshape(batch, -1)
