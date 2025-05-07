import numpy as np
from numba import njit, prange
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

# --------------------------------------------------------------- #

@njit(parallel=True)
def _conv2d_forward_numba(x, weight, bias, stride, padding, kernel_size, in_channels, out_channels):
    batch, _, H, W = x.shape
    K = kernel_size
    S = stride

    # pad input
    H_p = H + 2*padding
    W_p = W + 2*padding
    x_padded = np.zeros((batch, in_channels, H_p, W_p), dtype=x.dtype)
    for b in prange(batch):
        for c in range(in_channels):
            for i in range(H):
                for j in range(W):
                    x_padded[b, c, i+padding, j+padding] = x[b, c, i, j]

    H_out = (H_p - K)//S + 1
    W_out = (W_p - K)//S + 1

    out = np.zeros((batch, out_channels, H_out, W_out), dtype=x.dtype)

    for b in prange(batch):
        for oc in range(out_channels):
            for ic in range(in_channels):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * S
                        w_start = j * S
                        # element‐wise multiply and sum
                        s = 0.0
                        for di in range(K):
                            for dj in range(K):
                                s += weight[oc, ic, di, dj] * x_padded[b, ic, h_start+di, w_start+dj]
                        out[b, oc, i, j] += s
            for i in range(H_out):
                for j in range(W_out):
                    out[b, oc, i, j] += bias[oc]

    return out

@njit(parallel=True)
def _conv2d_backward_numba(x, weight, grad_out,
                           stride, padding,
                           kernel_size, in_channels, out_channels):
    B, C_in, H, W = x.shape
    C_out, _, K, _ = weight.shape
    S, P = stride, padding

    # pad input
    H_p = H + 2*P
    W_p = W + 2*P
    x_padded = np.zeros((B, C_in, H_p, W_p), dtype=x.dtype)
    for b in prange(B):
        for c in range(C_in):
            for i in range(H):
                for j in range(W):
                    x_padded[b, c, i+P, j+P] = x[b, c, i, j]

    # output spatial
    H_out = (H_p - K)//S + 1
    W_out = (W_p - K)//S + 1

    # grads
    grad_input_padded = np.zeros_like(x_padded)
    grad_weight       = np.zeros_like(weight)
    grad_bias         = np.zeros((C_out,), dtype=x.dtype)

    # accumulate
    for b in prange(B):
        for oc in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    g = grad_out[b, oc, i, j]
                    grad_bias[oc] += g
                    h0 = i * S
                    w0 = j * S
                    # grad wrt weight and input
                    for ic in range(C_in):
                        for di in range(K):
                            for dj in range(K):
                                x_val = x_padded[b, ic, h0+di, w0+dj]
                                grad_weight[oc, ic, di, dj] += x_val * g
                                grad_input_padded[b, ic, h0+di, w0+dj] += weight[oc, ic, di, dj] * g

    # unpad input gradient
    grad_input = grad_input_padded[:, :,
                                   P:P+H,
                                   P:P+W]
    return grad_input, grad_weight, grad_bias

# --------------------------------------------------------------------------- #

@njit(parallel=True)
def _batchnorm2d_forward(x, gamma, beta, eps):
    batch, channels, H, W = x.shape
    out = np.empty_like(x)
    mean = np.zeros(channels, dtype=x.dtype)
    var  = np.zeros(channels, dtype=x.dtype)
    N = batch * H * W

    # compute mean
    for c in prange(channels):
        s = 0.0
        for b in range(batch):
            for i in range(H):
                for j in range(W):
                    s += x[b, c, i, j]
        mean[c] = s / N

    # compute var
    for c in prange(channels):
        s = 0.0
        m = mean[c]
        for b in range(batch):
            for i in range(H):
                for j in range(W):
                    d = x[b, c, i, j] - m
                    s += d * d
        var[c] = s / N

    # normalize + scale + shift
    x_hat = np.empty_like(x)
    for b in prange(batch):
        for c in range(channels):
            inv = 1.0/np.sqrt(var[c] + eps)
            for i in range(H):
                for j in range(W):
                    x_hat[b, c, i, j] = (x[b, c, i, j] - mean[c]) * inv
                    out[b, c, i, j] = gamma[c] * x_hat[b, c, i, j] + beta[c]

    return out, x_hat, mean, var

@njit(parallel=True)
def _batchnorm2d_backward(grad_out, x_hat, mean, var, gamma, beta, eps):
    batch, channels, H, W = grad_out.shape
    N = batch * H * W
    grad_input = np.zeros_like(grad_out)
    grad_gamma = np.zeros_like(gamma)
    grad_beta  = np.zeros_like(beta)

    # grad_gamma, grad_beta
    for c in prange(channels):
        sg = 0.0; sb = 0.0
        for b in range(batch):
            for i in range(H):
                for j in range(W):
                    go = grad_out[b, c, i, j]
                    sg += go * x_hat[b, c, i, j]
                    sb += go
        grad_gamma[c] = sg
        grad_beta[c]  = sb

    # grad_input
    for c in prange(channels):
        inv = 1.0/np.sqrt(var[c] + eps)
        sum_d = 0.0; sum_dx = 0.0
        for b in range(batch):
            for i in range(H):
                for j in range(W):
                    dh = grad_out[b, c, i, j] * gamma[c]
                    sum_d  += dh
                    sum_dx += dh * x_hat[b, c, i, j]
        for b in range(batch):
            for i in range(H):
                for j in range(W):
                    dh = grad_out[b, c, i, j] * gamma[c]
                    grad_input[b, c, i, j] = (1.0/N) * inv * (
                        N*dh - sum_d - x_hat[b, c, i, j]*sum_dx
                    )
    return grad_input, grad_gamma, grad_beta

# --------------------------------------------------------------------------- #

@njit(parallel=True)
def _conv_transpose2d_forward_numba(
    x,             # (N, C_in, H_in, W_in)
    weight,        # (C_in, C_out, K, K)
    bias,          # (C_out,)
    stride,        # int
    padding,       # int
    output_padding # int
):
    N, C_in, H_in, W_in = x.shape
    _, C_out, K, _      = weight.shape
    S = stride
    P = padding
    O = output_padding

    # compute output spatial dims
    H_out = (H_in - 1) * S - 2*P + K + O
    W_out = (W_in - 1) * S - 2*P + K + O

    out = np.zeros((N, C_out, H_out, W_out), dtype=x.dtype)

    # scatter each input pixel into the upsampled output
    for n in prange(N):
        for ci in range(C_in):
            for i in range(H_in):
                for j in range(W_in):
                    val = x[n, ci, i, j]
                    # top-left corner of this input patch
                    h0 = i * S - P
                    w0 = j * S - P
                    # slide the K×K kernel
                    for co in range(C_out):
                        for di in range(K):
                            for dj in range(K):
                                oi = h0 + di
                                oj = w0 + dj
                                if 0 <= oi < H_out and 0 <= oj < W_out:
                                    # note the kernel is flipped for transpose
                                    out[n, co, oi, oj] += val * weight[ci, co, K-1-di, K-1-dj]
        # add bias term across the spatial map
        for co in range(C_out):
            out[n, co, :, :] += bias[co]

    return out

@njit(parallel=True)
def _conv_transpose2d_backward_numba(
    x,             # (N, C_in, H_in, W_in)
    weight,        # (C_in, C_out, K, K)
    grad_out,      # (N, C_out, H_out, W_out)
    stride,
    padding,
    output_padding
):
    N, C_in, H_in, W_in = x.shape
    _, C_out, K, _      = weight.shape
    S = stride
    P = padding

    # grad_out spatial dims
    H_out = grad_out.shape[2]
    W_out = grad_out.shape[3]

    grad_input  = np.zeros_like(x)
    grad_weight = np.zeros_like(weight)
    grad_bias   = np.zeros((C_out,), dtype=x.dtype)

    for n in prange(N):
        # 1) bias gradient
        for co in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    grad_bias[co] += grad_out[n, co, i, j]

        # 2) weight gradient—loop over kernel indices
        for ci in range(C_in):
            for co in range(C_out):
                for di in range(K):
                    for dj in range(K):
                        acc_w = 0.0
                        # sum contributions from all input positions
                        for i in range(H_in):
                            for j in range(W_in):
                                oi = i * S - P + di
                                oj = j * S - P + dj
                                if 0 <= oi < H_out and 0 <= oj < W_out:
                                    acc_w += grad_out[n, co, oi, oj] * x[n, ci, i, j]
                        grad_weight[ci, co, di, dj] += acc_w

        # 3) grad_input
        for ci in range(C_in):
            for i in range(H_in):
                for j in range(W_in):
                    acc_x = 0.0
                    h0 = i * S - P
                    w0 = j * S - P
                    for di in range(K):
                        for dj in range(K):
                            oi = h0 + di
                            oj = w0 + dj
                            if 0 <= oi < H_out and 0 <= oj < W_out:
                                for co in range(C_out):
                                    # note: weight is indexed flipped in transpose
                                    acc_x += weight[ci, co, K-1-di, K-1-dj] * grad_out[n, co, oi, oj]
                    grad_input[n, ci, i, j] = acc_x

    return grad_input, grad_weight, grad_bias

# --------------------------------------------------------------------------- #

@njit(parallel=True)
def _maxpool2d_forward_numba(x, kernel_size, stride):
    batch, channels, H, W = x.shape
    K = kernel_size
    S = stride
    H_out = (H - K) // S + 1
    W_out = (W - K) // S + 1
    out = np.empty((batch, channels, H_out, W_out), dtype=x.dtype)
    for b in prange(batch):
        for c in range(channels):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * S
                    w_start = j * S
                    # find max in the window
                    max_val = x[b, c, h_start, w_start]
                    for di in range(K):
                        for dj in range(K):
                            val = x[b, c, h_start + di, w_start + dj]
                            if val > max_val:
                                max_val = val
                    out[b, c, i, j] = max_val
    return out

@njit(parallel=True)
def _maxpool2d_backward_numba(x, grad_out, kernel_size, stride):
    batch, channels, H, W = x.shape
    K = kernel_size
    S = stride
    H_out = grad_out.shape[2]
    W_out = grad_out.shape[3]
    grad_input = np.zeros_like(x)
    for b in prange(batch):
        for c in range(channels):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * S
                    w_start = j * S
                    # identify max location
                    max_val = x[b, c, h_start, w_start]
                    max_i, max_j = 0, 0
                    for di in range(K):
                        for dj in range(K):
                            val = x[b, c, h_start + di, w_start + dj]
                            if val > max_val:
                                max_val = val
                                max_i, max_j = di, dj
                    grad_input[b, c, h_start + max_i, w_start + max_j] = grad_out[b, c, i, j]
    return grad_input

# --------------------------------------------------------------------------- #

@njit(parallel=True)
def _avgpool2d_forward(x, kernel_size, stride):
    batch, channels, H, W = x.shape
    K, S = kernel_size, stride
    H_out = (H - K)//S + 1
    W_out = (W - K)//S + 1
    out = np.empty((batch, channels, H_out, W_out), dtype=x.dtype)
    for b in prange(batch):
        for c in range(channels):
            for i in range(H_out):
                for j in range(W_out):
                    s = 0.0
                    for di in range(K):
                        for dj in range(K):
                            s += x[b, c, i*S+di, j*S+dj]
                    out[b, c, i, j] = s/(K*K)
    return out

@njit(parallel=True)
def _avgpool2d_backward(x, grad_out, kernel_size, stride):
    batch, channels, H, W = x.shape
    K, S = kernel_size, stride
    H_out, W_out = grad_out.shape[2], grad_out.shape[3]
    grad_input = np.zeros_like(x)
    inv = 1.0/(K*K)
    for b in prange(batch):
        for c in range(channels):
            for i in range(H_out):
                for j in range(W_out):
                    g = grad_out[b, c, i, j] * inv
                    for di in range(K):
                        for dj in range(K):
                            grad_input[b, c, i*S+di, j*S+dj] += g
    return grad_input
