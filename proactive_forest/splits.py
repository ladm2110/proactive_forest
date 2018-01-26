import numpy as np


def compute_split_info(args):
    split_criterion, X, y, feature_id, split_value = args

    if isinstance(X[0, feature_id], str):
        _, _, y_left, y_right = split_categorical_data(X, y, feature_id, split_value)
    else:
        _, _, y_left, y_right = split_numerical_data(X, y, feature_id, split_value)

    n_left, n_right = len(y_left), len(y_right)
    if n_left == 0 or n_right == 0:
        return None, np.min([n_left, n_right])

    gain = compute_split_gain(split_criterion, y, y_left, y_right)
    return gain, np.min([n_left, n_right])


def split_categorical_data(X, y, feature_id, value):
    mask = X[:, feature_id] == value
    return X[mask], X[~mask], y[mask], y[~mask]


def split_numerical_data(X, y, feature_id, value):
    mask = X[:, feature_id] <= value
    return X[mask], X[~mask], y[mask], y[~mask]


def compute_split_gain(split_criterion, y, y_left, y_right):
    splits = [y_left, y_right]
    return split_criterion.impurity_gain(y) - sum([split_criterion.impurity_gain(split) * float(len(split)) / len(y) for split in splits])

