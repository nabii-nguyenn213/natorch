import numpy as np
from natorch.nn.modules.module import Module

class ReLU(Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        mask = x > 0
        y = np.where(mask, x, 0)
        self._caches['mask'] = mask
        return y

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        mask = self._caches['mask']
        return grad_out * mask

class Sigmoid(Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = 1.0 / (1.0 + np.exp(-x))
        self._caches['output'] = y
        return y

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        y = self._caches['output']
        return grad_out * y * (1 - y)

class Tanh(Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.tanh(x)
        self._caches['output'] = y
        return y

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        y = self._caches['output']
        return grad_out * (1 - y * y)

class LeakyReLU(Module):

    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._caches['input'] = x
        return np.where(x >= 0, x, self.negative_slope * x)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        x = self._caches['input']
        grad_input = np.where(x >= 0, grad_out, self.negative_slope * grad_out)
        return grad_input

class Softmax(Module):

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: np.ndarray) -> np.ndarray:
        shift = x - np.max(x, axis=self.dim, keepdims=True)
        exps = np.exp(shift)
        y = exps / np.sum(exps, axis=self.dim, keepdims=True)
        self._caches['output'] = y
        return y

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        y = self._caches['output']
        dot = np.sum(grad_out * y, axis=self.dim, keepdims=True)
        return y * (grad_out - dot)
