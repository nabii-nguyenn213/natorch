from typing import Tuple
import numpy as np

class Parameter:

    def __init__(self, shape : Tuple[int, ...], dtype : np.dtype = np.float32, requires_grad : bool = False):
        self.data = np.empty(shape, dtype=dtype)
        self.requires_grad = requires_grad
        self.dtype = dtype
        self.grad = np.zeros_like(data) if requires_grad else None

    def zero_grad(self):
        if self.requires_grad and self.grad is not None:
            self.grad.fill(0)

    def __repr__(self):
        base = f"Parameter(shape={self.data.shape}, dtype={self.data.dtype}"
        if self.requires_grad:
            base += ", requires_grad=True"
        base += ")"
        return base
