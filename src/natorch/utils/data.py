import numpy as np

def one_hot(x: np.ndarray, num_classes: int) -> np.ndarray:
    flat = x.ravel().astype(int)
    eye = np.eye(num_classes, dtype=x.dtype)
    oh = eye[flat]                              
    return oh.reshape(*x.shape, num_classes)
