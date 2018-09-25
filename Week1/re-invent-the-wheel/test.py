import numpy as np
import matplotlib.pyplot as plt

from knn import KNeighborsClassifier
from utils import create_dataset


X, y = create_dataset()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.get_cmap('jet'))
plt.show()

X_test = np.array([
    [1.0, 1.1],
    [0.2, 0.1],
])

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

y_predict = neigh.predict(X_test)
print(y_predict)
