import numpy as np
from typing import Tuple

def relu(x : np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def sigmoid(x : np.ndarray) -> np.ndarray:
    return 1.0/(1+np.exp(-x))

def stable_sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    neg = ~pos
    exp_x = np.exp(x[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out

def tanh(x : np.ndarray) -> np.ndarray:
    return np.tanh(x)

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def leaky_relu(x : np.ndarray) -> np.ndarray:
    alpha = 0.01
    return np.where(x > 0, x, x * alpha)

def elu(x: np.ndarray) -> np.ndarray:
    alpha = 0.01
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def batch_norm(x : np.ndarray, gamma : np.ndarray, beta : np.ndarray, eps : float = 1e-5, momentum : float = .9, running_mean : np.ndarray = None,
               running_var : np.ndarray = None, training : bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    axes = tuple([i for i in range(x.ndim) if i not in (0, x.ndim-1)])
    if running_mean is None:
        running_mean = np.zeros(x.shape[-1], dtype=x.dtype)
        running_var  = np.ones(x.shape[-1],  dtype=x.dtype)
    if training:
        batch_mean = np.mean(x, axis=axes, keepdims=False)
        batch_var  = np.var(x,  axis=axes, keepdims=False)
        # update running stats
        running_mean = momentum * running_mean + (1 - momentum) * batch_mean
        running_var  = momentum * running_var  + (1 - momentum) * batch_var
        mean, var = batch_mean, batch_var
    else:
        mean, var = running_mean, running_var
    x_norm = (x - mean.reshape((1,)* (x.ndim-1) + (-1,))) / np.sqrt(var + eps).reshape((1,)* (x.ndim-1) + (-1,))
    out = gamma.reshape((1,)* (x.ndim-1) + (-1,)) * x_norm + beta.reshape((1,)* (x.ndim-1) + (-1,))
    return out, running_mean, running_var

def max_pool1d():
    pass

def max_pool2d():
    pass

def conv1d():
    pass

def conv2d():
    pass

def conv_transpose1d():
    pass

def conv_transpose2d():
    pass

def drop_out():
    pass

def linear():
    pass



