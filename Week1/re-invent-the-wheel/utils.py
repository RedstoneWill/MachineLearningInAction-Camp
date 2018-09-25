import numpy as np


def check_X_y(X, y):
    """Input validation on an array or list."""

    if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
        if X.ndim == 1:
            raise Exception("Input only accepts 2-D array.")
        else:
            return X, y
    if isinstance(X, list) and isinstance(y, list):
        return np.array(X), np.array(y)
    else:
        raise Exception('Please convert input data type to array or list.')


def check_X(X):
    """Input validation on an array or list."""

    if isinstance(X, np.ndarray):
        if X.ndim == 1:
            raise Exception("Input only accepts 2-D array.")
        else:
            return X


def create_dataset():
    """Create a simple dataset for testing.  """

    X = np.array([
        [1.0, 1.1],
        [1.0, 1.0],
        [0, 0],
        [0, 0.1],
    ])
    y = np.array(['0', '0', '1', '1'])

    return X, y
