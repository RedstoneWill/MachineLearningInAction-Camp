"""
Nearest Neighbor Classification
The structure is inspired by scikit-learn python library.
"""

# Author: Yang Dai <daiy@mit.edu>


import numpy as np
from collections import Counter

import utils


class KNeighborsClassifier:
    """Classifier implementing the k-nearest neighbors vote.

    Euclidean distance is chosen as the metric.

    Parameters
    ----------
    n_neighbors : int, optional (default = 3)
        Number of neighbors used by default for kneighbors' queries.
    """

    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """
        Fit KNN model.

        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]
            Training set.

        y : 1-d numpy array (vector) of shape [n_samples].
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """

        self.X, self.y = utils.check_X_y(X, y)
        return self

    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]
            Training set.

        Returns
        -------
        y_predict : 1-d numpy array (vector) of shape [n_samples].
            Class labels for each data sample.
        """

        X = utils.check_X(X)
        n_samples = self.X.shape[0]

        y_predict = []
        for instance in X:
            distance = np.linalg.norm(
                self.X - np.tile(instance, (n_samples, 1)), axis=1,
            )

            # Get the indices of k highest numbers.
            k_nearest_indx = distance.argsort()[:self.n_neighbors]

            # Get the label with the majority votes.
            nearest_vectors = self.y[k_nearest_indx]
            c = Counter(nearest_vectors)
            y_predict.append(c.most_common(1)[0][0])

        y_predict = np.array(y_predict)
        return y_predict
