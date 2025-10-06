import numpy as np
from natorch.nn.modules import Module
from natorch.nn.functional import mse_loss, binary_cross_entropy, cross_entropy_loss, cross_entropy_gradient

class MSELoss(Module):

    def __init__(self):
        super().__init__()

    def forward(self, input: np.ndarray, target: np.ndarray) -> float:
        self._caches['input'], self._caches['target'] = input, target
        return mse_loss(input, target)

    def backward(self, grad_out: float = 1.0) -> np.ndarray:
        inp, tgt = self._caches['input'], self._caches['target']
        # derivative of mean((inp - tgt)^2) is 2*(inp - tgt)/N
        grad = 2 * (inp - tgt) / inp.size
        return grad * grad_out

class BCELoss(Module):

    def __init__(self):
        super().__init__()

    def forward(self, input: np.ndarray, target: np.ndarray) -> float:
        self._caches['input'], self._caches['target'] = input, target
        return binary_cross_entropy(input, target)

    def backward(self, grad_out: float = 1.0) -> np.ndarray:
        inp, tgt = self._caches['input'], self._caches['target']
        eps = 1e-12
        inp = np.clip(inp, eps, 1 - eps)
        grad = (-(tgt / inp) + (1 - tgt) / (1 - inp)) / inp.size
        return grad * grad_out

class CrossEntropyLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: np.ndarray, target: np.ndarray) -> float:
        self._caches['input'] = input
        self._caches['target'] = target
        return cross_entropy_loss(input, target)

    def backward(self, grad_out: float = 1.0) -> np.ndarray:
        input = self._caches['input']
        target = self._caches['target']
        grad = cross_entropy_gradient(input, target)
        return grad * grad_out
