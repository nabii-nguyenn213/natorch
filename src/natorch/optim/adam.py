import numpy as np
from natorch.nn.parameter import Parameter
from typing import Iterable, Union, Dict

class Adam: 
    
    def __init__(
        self,
        parameters: Iterable[Union[Parameter, Dict[str, Parameter]]],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8
    ):
        # Flatten parameters if passed as list of dicts
        flat_params = []
        for item in parameters:
            if isinstance(item, dict):
                for p in item.values():
                    if isinstance(p, Parameter):
                        flat_params.append(p)
                    else:
                        raise TypeError(f"Expected Parameter in dict, got {type(p)}")
            elif isinstance(item, Parameter):
                flat_params.append(item)
            else:
                raise TypeError(f"Unsupported parameter type: {type(item)}")

        self.parameters = flat_params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        # initialize first and second moment buffers
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]
        self.t = 0

    def zero_grad(self) -> None:
        """Set gradients of all parameters to zero."""
        for p in self.parameters:
            if hasattr(p, 'grad') and p.grad is not None:
                p.grad.fill(0)

    def step(self) -> None:
        """
        Perform a single optimization step (parameter update).
        """
        self.t += 1
        for i, p in enumerate(self.parameters):
            g = p.grad
            if g is None:
                continue
            # update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            # update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)
            # compute bias-corrected estimates
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            # update parameters
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
