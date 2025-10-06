import numpy as np

def accuracy(outputs : np.ndarray, targets : np.ndarray) -> float:

    if outputs.ndim > 1:
        preds = np.argmax(outputs, axis=1)
    else:
        preds = (outputs >= 0.5).astype(int)

    if targets.ndim > 1:
        true = np.argmax(targets, axis=1)
    else:
        true = targets

    return float(np.mean(preds == true))
