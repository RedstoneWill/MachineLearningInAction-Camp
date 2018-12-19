"""Predict if you like this person"""

import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

from lib.knn import KNeighborsClassifier
from lib.utils import load_data


def print_header():
    print('--------------------------------------------------')
    print('              PREDICT YOUR FAVORITE')
    print('--------------------------------------------------')
    print()


def get_input_from_user():
    miles = input("Frequent flier miles earned per year: ")
    time_on_game = input("Percentage of time spent playing video game: ")
    icecream = input("Liters of ice cream consumed per year: ")

    try:
        X = np.array(
            [miles, time_on_game, icecream], dtype=float
        ).reshape(1, -1)

        return X

    except ValueError:
        print("Please input a number.")


def predict(data):
    X, y = load_data('datingTestSet.txt')

    scale = StandardScaler()
    X = scale.fit_transform(X)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)

    result = model.predict(scale.transform(data))
    result = encoder.inverse_transform(result)

    return result


if __name__ == '__main__':
    print_header()
    data = get_input_from_user()
    result = predict(data)

    print("Based on your info, your result is: {}".format(result))
