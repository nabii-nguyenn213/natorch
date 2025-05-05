import numpy as np
from typing import Tuple, Any

def one_hot(x: np.ndarray, num_classes: int) -> np.ndarray:
    flat = x.ravel().astype(int)
    eye = np.eye(num_classes, dtype=x.dtype)
    oh = eye[flat]                              
    return oh.reshape(*x.shape, num_classes)

def train_test_split( *arrays: np.ndarray, test_size: float = 0.25, shuffle: bool = True, random_seed: int = None) -> Tuple[Any, ...]:
    
    if not arrays:
        raise ValueError("At least one array must be provided")

    n_samples = arrays[0].shape[0]
    for arr in arrays:
        if arr.shape[0] != n_samples:
            raise ValueError("All input arrays must have the same first dimension length")

    if random_seed is not None:
        np.random.seed(random_seed)

    if isinstance(test_size, float):
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size must be between 0.0 and 1.0 when a float")
        n_test = int(n_samples * test_size)
    elif isinstance(test_size, int):
        if not 0 < test_size < n_samples:
            raise ValueError("test_size must be between 1 and the number of samples when an int")
        n_test = test_size
    else:
        raise ValueError("test_size must be float or int")

    if shuffle:
        indices = np.random.permutation(n_samples)
    else:
        indices = np.arange(n_samples)

    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    split_arrays = []
    for arr in arrays:
        if hasattr(arr, 'iloc'):
            subset_train = arr.iloc[train_idx]
            subset_test  = arr.iloc[test_idx]
        else:
            subset_train = arr[train_idx]
            subset_test  = arr[test_idx]
        split_arrays.append(subset_train)
        split_arrays.append(subset_test)

    return tuple(split_arrays)
