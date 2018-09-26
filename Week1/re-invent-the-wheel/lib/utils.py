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
    """Create a simple dataset for classification testing."""

    X = np.array([
        [1.0, 1.1],
        [1.0, 1.0],
        [0, 0],
        [0, 0.1],
    ])
    y = np.array(['0', '0', '1', '1'])

    return X, y


def load_data(fname, delimiter='\t'):
    """
    Load data from a text file.

    No headers.
    Each row in the text file must have the same number of values.

    In production, this function should be replaced by numpy.loadtxt or \
        pd.read_csv

    Parameters
    ----------
    fname: str or path-like object (absolute or relative to the current \
        working dir)

    Returns
    -------
    result: numpy array.
    X : feature array with the shape (n_samples, n_features)
    y : label array with the shape (n_samples,)
        Features and labels.
    """

    with open(fname, 'r') as fin:
        raw_data = fin.readlines()
        n_rows = len(raw_data)
        n_cols = len(raw_data[0].split(delimiter))

        X = np.empty((n_rows, n_cols - 1))
        y = []
        for i, row in enumerate(raw_data):
            X[i, :] = row.split(delimiter)[:-1]
            y.append(row.split(delimiter)[-1].strip())

        y = np.array(y)

    return X, y


def split_train_test(X, y, ratio=0.2):
    """
    Create new training and test data sets based on the provided ratio

    Parameters:
    ----------
    X : array.
        Feature array with the shape (n_samples, n_features)
    y : array.
        Label array with the shape (n_samples,)
    ratio: float
        The ratio of n_training_samples and n_test_samples.

    Returns
    -------
    X_train : array.
        Training feature array.
    y_train : array.
        Training labels.
    X_test : array.
        Test feature array.
    y_test : array.
        Test labels.
    """

    shuffled_indices = np.random.permutation(len(X))
    n_test_samples = int(len(X) * ratio)
    test_indices = shuffled_indices[:n_test_samples]
    train_indices = shuffled_indices[n_test_samples:]
    X_train = X[train_indices, :]
    y_train = y[train_indices]
    X_test = X[test_indices, :]
    y_test = y[test_indices]

    return X_train, y_train, X_test, y_test


def get_error_rate(y_label, y_predict):
    """
    Get the error rate to test the effectiveness of a binaryclassification \
        model.

    Parameter:
    ----------
    y_label: array
        The true labels in a test data set.
    y_predict: array
        The predicted classes from a trained model.

    Returns:
    -------
    error_rate: float
        n_errors / n_samples.
    """

    n_samples = len(y_label)
    n_errors = n_samples - np.sum(y_label == y_predict)
    error_rate = n_errors / len(y_label)
    return error_rate
