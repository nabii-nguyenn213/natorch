from typing import Tuple
from natorch.nn import parameter
import numpy as np
import math

def zeros_(param : Parameter) -> Parameter:
    _check_param(param)
    param.data[...] = 0.0
    return param

def ones_(param : Parameter) -> Parameter:
    _check_param(param)
    param.data[...] = 1.0
    return param

def constants_(param : Parameter, val : float) -> Parameter:
    _check_param(param)
    param.data[...] = val
    return param

def xavier_uniform_(param : Parameter, gain : float = 1.0) -> Parameter:
    _check_param(param)
    shape = param.data.shape

    fan_in, fan_out = _calculate_fans(shape)
    alpha = gain * math.sqrt(6.0 / (fan_in + fan_out))
    
    param.data[...] = np.random.uniform(-alpha, alpha, size=shape)
    return param
    
def xavier_normal_(param : Parameter, gain : float = 1.0) -> Parameter:
    _check_param(param)
    shape = param.data.shape

    fan_in, fan_out = _calculate_fans(shape)
    std = gain * math.sqrt(2.0, (fan_in + fan_out))

    param.data[...] = np.random.normal(0,0, std, size=shape)
    return param

def kaiming_uniform_(param : Parameter, mode :str = 'fan_in', nonlinearity : str = 'relu') -> Parameter:
    _check_param(param)
    mode = mode.lower()
    if mode not in ['fan_in', 'fan_out']:
        raise ValueError(f"mode must be 'fan_in' or 'fan_out', got {mode}")
    shape = param.data.shape

    fan_in, fan_out = _calculate_fans(shape)
    fan = fan_in if mode == 'fan_in' else fan_out

    gain = _calculate_gain(nonlinearity, negative_slope)
    bound = gain * math.sqrt(3.0 / fan)

    param.data[...] = np.random.uniform(-bound, bound, size=shape).astype(param.dtype)
    return param

def kaiming_normal_(param : Parameter, mode : str = 'fan_in', nonlinearity : str = 'relu', negative_slope : float = 0.0) -> Parameter:
    _check_param(param)
    mode = mode.lower()
    if mode not in ['fan_in', 'fan_out']:
        raise ValueError(f"mode must be 'fan_in' or 'fan_out', got {mode!r}")
    shape = param.data.shape

    fan_in, fan_out = _calculate_fans(shape)
    fan = fan_in if mode == 'fan_in' else fan_out

    gain = _calculate_gain(nonlinearity, negative_slope)
    std = gain * math.sqrt(2.0 / fan)

    param.data[...] = np.random.normal(0.0, std, size=shape).astype(param.dtype)
    return param


def _calculate_gain(nonlinearity : str, negative_slope : float = 0.1):
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv_transpose1d', 'conv_transpose2d']

    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0/3.0
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        return math.sqrt(2.0/(1+negative_slope**2))
    elif nonlinearity == 'selu':
        return 3.0/4.0
    raise ValueError(f"Unsupported nonlinearity '{nonlinearity}'")

def _calculate_fans(shape : Tuple[int, ...]) -> Tuple[int, int]:
    
    if len(shape) == 2:
        return shape[0], shape[1]

    num_out_feature_maps, num_in_feature_maps = shape[0], shape[1]
    receptive_field_size = int(np.prod(shape[2:])) if len(shape) > 2 else 1

    fan_in = num_in_feature_maps * receptive_field_size
    fan_out = num_out_feature_maps * receptive_field_size

    return fan_in, fan_out

def _check_param(param: Parameter) -> None:
    if not isinstance(param, Parameter):
        raise TypeError(f"Expected Parameter, got {type(param)}")
