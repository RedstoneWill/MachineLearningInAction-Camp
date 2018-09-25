import numpy as np
from knn import KNeighborsClassifier
from utils import create_dataset


X, y = create_dataset()
X_test = np.array([1.0, 1.1])

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

y_predict = neigh.predict(X_test)
print(y_predict)
