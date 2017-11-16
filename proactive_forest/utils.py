import numpy as np


def compute_split_info(args):
    split_criterion, X, y, feature_id, split_value = args

    if X[:, feature_id].dtype.type == np.object_:
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
    return split_criterion(y) - sum([split_criterion(split) * float(len(split)) / len(y) for split in splits])


class Sampler:
    def __init__(self, n_instances):
        self.n_instances = n_instances
        self.training_ids = None
        self.out_of_bag_ids = None

    def get_training_sample(self):
        if self.training_ids is None:
            self.training_ids = np.random.choice(self.n_instances, replace=True, size=self.n_instances)
        return self.training_ids

    def get_out_of_bag(self):
        if self.training_ids is not None:
            if self.out_of_bag_ids is None:
                self.out_of_bag_ids = self._calculate_out_of_bag_ids
            return self.out_of_bag_ids
        else:
            raise Exception('Training sample should be generated first')

    def _calculate_out_of_bag_ids(self):
        # TODO
        pass

