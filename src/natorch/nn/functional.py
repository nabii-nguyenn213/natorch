import numpy as np
from typing import Tuple, Optional

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

def max_pool2d(x: np.ndarray, kernel_size: Tuple[int, int], stride: Optional[Tuple[int, int]] = None, padding: Tuple[int, int] = (0, 0)) -> np.ndarray:
    """Naive 2D max pooling."""
    N, C, H, W = x.shape
    kH, kW = kernel_size
    if stride is None:
        sH, sW = kernel_size
    else:
        sH, sW = stride
    pH, pW = padding
    # pad input
    x_p = np.pad(x, ((0,0),(0,0),(pH,pH),(pW,pW)), mode='constant', constant_values=-np.inf)
    H_out = (H + 2*pH - kH) // sH + 1
    W_out = (W + 2*pW - kW) // sW + 1
    out = np.empty((N, C, H_out, W_out), dtype=x.dtype)
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h0 = i * sH
                    w0 = j * sW
                    window = x_p[n, c, h0:h0+kH, w0:w0+kW]
                    out[n, c, i, j] = np.max(window)
    return out


def conv2d(input: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray] = None, stride: Tuple[int, int] = (1, 1), padding: Tuple[int, int] = (0, 0)) -> np.ndarray:
    """Naive 2D convolution."""
    N, C_in, H_in, W_in = input.shape
    C_out, _, K_h, K_w = weight.shape
    sH, sW = stride
    pH, pW = padding
    H_out = (H_in + 2*pH - K_h) // sH + 1
    W_out = (W_in + 2*pW - K_w) // sW + 1
    x_p = np.pad(input, ((0,0),(0,0),(pH,pH),(pW,pW)), mode='constant')
    out = np.zeros((N, C_out, H_out, W_out), dtype=input.dtype)
    for n in range(N):
        for oc in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    region = x_p[n, :, i*sH:i*sH+K_h, j*sW:j*sW+K_w]
                    out[n, oc, i, j] = np.sum(region * weight[oc])
            if bias is not None:
                out[n, oc] += bias[oc]
    return out


def conv_transpose2d(input: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray] = None, stride: Tuple[int, int] = (1, 1), padding: Tuple[int, int] = (0, 0), output_padding: Tuple[int, int] = (0, 0)) -> np.ndarray:
    """Naive 2D transposed convolution (deconvolution)."""
    N, C_in, H_in, W_in = input.shape
    C_out, _, K_h, K_w = weight.shape
    sH, sW = stride
    pH, pW = padding
    opH, opW = output_padding
    H_out = (H_in - 1) * sH - 2*pH + K_h + opH
    W_out = (W_in - 1) * sW - 2*pW + K_w + opW
    out = np.zeros((N, C_out, H_out, W_out), dtype=input.dtype)
    for n in range(N):
        for c in range(C_in):
            for i in range(H_in):
                for j in range(W_in):
                    val = input[n, c, i, j]
                    h0 = i * sH - pH
                    w0 = j * sW - pW
                    for ki in range(K_h):
                        for kj in range(K_w):
                            oi = h0 + ki
                            oj = w0 + kj
                            if 0 <= oi < H_out and 0 <= oj < W_out:
                                out[n, :, oi, oj] += val * weight[:, c, K_h-1-ki, K_w-1-kj]
        if bias is not None:
            out[n] += bias.reshape(-1, 1, 1)
    return out


def dropout(x: np.ndarray, p: float = 0.5, training: bool = True) -> np.ndarray:
    """Dropout: randomly zeroes elements with probability p during training."""
    if not training or p == 0.0:
        return x
    mask = (np.random.rand(*x.shape) > p).astype(x.dtype)
    return x * mask / (1.0 - p)

def mse_loss(input_: np.ndarray, target: np.ndarray) -> float:
    return np.mean((input_ - target) ** 2)

def binary_cross_entropy(input_: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    input_ = np.clip(input_, eps, 1 - eps)
    return -np.mean(target * np.log(input_) + (1 - target) * np.log(1 - input_))

def cross_entropy_loss(input_: np.ndarray, target: np.ndarray):
    if input_.ndim == 1:
        input_ = input_[np.newaxis, :]
    N, C = input_.shape

    x_max    = np.max(input_, axis=1, keepdims=True)
    logits   = input_ - x_max
    exp_logits = np.exp(logits)
    sum_exp    = np.sum(exp_logits, axis=1, keepdims=True)
    log_probs  = logits - np.log(sum_exp)       

    if target.ndim == 1:
        losses = -log_probs[np.arange(N), target]
    else:
        losses = -np.sum(target * log_probs, axis=1)

    return np.mean(losses)

def cross_entropy_gradient(input_: np.ndarray, target: np.ndarray) -> np.ndarray:
    squeeze = False
    if input_.ndim == 1:
        input_ = input_[np.newaxis, :]
        squeeze = True

    N, C = input_.shape

    x_max     = np.max(input_, axis=1, keepdims=True)
    logits    = input_ - x_max
    exp_logits = np.exp(logits)
    sum_exp    = np.sum(exp_logits, axis=1, keepdims=True)
    probs      = exp_logits / sum_exp       # (N, C)

    if target.ndim == 1:
        grad = probs.copy()
        grad[np.arange(N), target] -= 1
    else:
        grad = probs - target

    grad /= N

    if squeeze:
        return grad[0]  
    return grad
