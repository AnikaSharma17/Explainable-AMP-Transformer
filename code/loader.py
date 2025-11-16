import numpy as np

def dict_to_array(pc6_dict, label_dict):
    """Convert pc6_dict and label_dict to arrays.

    pc6_dict: dict mapping id -> np.array shape (200,6)
    label_dict: dict mapping id -> label (0/1)

    Returns:
        X: np.ndarray shape (N,200,6)
        y: np.ndarray shape (N,)
    """
    X = np.array(list(pc6_dict.values()))
    y = np.array([label_dict[k] for k in pc6_dict.keys()])
    return X, y
