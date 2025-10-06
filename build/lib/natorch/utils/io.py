import os
import json
import shutil
import numpy as np
from typing import List, Dict, Union
from natorch.nn.parameter import Parameter

DEFAULT_CHECKPOINT_DIR = os.path.expanduser('../../../../../checkpoints/lenet5')

def save_model(params: List[Dict[str, Union[Parameter, np.ndarray]]],save_dir: str = None,file_name: str = None) -> None:
    if save_dir is None:
        save_dir = DEFAULT_CHECKPOINT_DIR
    if not file_name:
        raise ValueError("file_name cannot be empty")
    os.makedirs(save_dir, exist_ok=True)

    if not file_name.endswith('.npz'):
        file_name += '.npz'
    path = os.path.join(save_dir, file_name)

    flat: Dict[str, np.ndarray] = {}
    for idx, layer in enumerate(params):
        for name, arr in layer.items():
            # extract raw array
            if isinstance(arr, Parameter):
                data = arr.data
            elif isinstance(arr, np.ndarray):
                data = arr
            else:
                raise TypeError(f"Layer {idx} param '{name}' has unsupported type {type(arr)}")
            flat[f'layer{idx}.{name}'] = data

    np.savez(path, **flat)


def load_params(save_dir: str = None,file_name: str = None) -> List[Dict[str, np.ndarray]]:
    if save_dir is None:
        save_dir = DEFAULT_CHECKPOINT_DIR
    if not file_name:
        raise ValueError("file_name cannot be empty")
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    path = os.path.join(save_dir, file_name)

    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such file: {path!r}")

    archive = np.load(path, allow_pickle=True)
    layers: Dict[int, Dict[str, np.ndarray]] = {}

    for key in archive.files:
        prefix, name = key.split('.', 1)
        idx = int(prefix.replace('layer', ''))
        layers.setdefault(idx, {})[name] = np.asarray(archive[key])

    return [layers[i] for i in sorted(layers.keys())]


def load_model(model,save_dir: str = None,file_name: str = None) -> None:
    loaded = load_params(save_dir=save_dir, file_name=file_name)

    # pick out only the layers that actually have weights
    layers_with_params = [
        layer for layer in model.net._layers
        if (hasattr(layer, 'weights') and isinstance(layer.weights, Parameter)
           and hasattr(layer, 'bias') and isinstance(layer.bias, Parameter)) or 
        (hasattr(layer, 'gamma') and isinstance(layer.gamma, Parameter)
           and hasattr(layer, 'beta') and isinstance(layer.beta, Parameter))
    ]

    if len(loaded) != len(layers_with_params):
        raise ValueError(
            f"Loaded {len(loaded)} layers but model has {len(layers_with_params)} parameterized layers"
        )

    for layer, params in zip(layers_with_params, loaded):
        if hasattr(layer, 'gamma') and hasattr(layer, 'beta'):
            layer.gamma.data = params['gamma']
            layer.beta.data = params['beta']
            continue
        layer.weights.data = params['weights']
        layer.bias.data    = params['bias']
        # zero gradients
        layer.weights.grad = np.zeros_like(layer.weights.data)
        layer.bias.grad    = np.zeros_like(layer.bias.data)
        # if your layer needs to update any internal buffers, call it
        if hasattr(layer, '_update_params'):
            layer._update_params()


def clear_checkpoint(save_dir: str = None) -> None:
    if save_dir is None:
        save_dir = DEFAULT_CHECKPOINT_DIR
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)


def save_hyper_params(file_path: str = None,**kwargs) -> None:
    if file_path is None:
        file_path = os.path.join(DEFAULT_CHECKPOINT_DIR, 'hyper_params.json')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w') as f:
        json.dump(kwargs, f, indent=4)
