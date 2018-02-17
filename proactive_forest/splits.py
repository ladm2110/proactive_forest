from abc import abstractmethod, ABC
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


class Split:
    def __init__(self, feature_id, feature_weight, value, gain):
        self.feature_id = feature_id
        self.feature_weight = feature_weight
        self.value = value
        self.gain = gain


class SplitChooser(ABC):
    @abstractmethod
    def get_split(self):
        pass


class BestSplitChooser(SplitChooser):
    def get_split(self, splits):
        best_split = None
        if len(splits) > 0:
            best_split = splits[0]
            for i in range(len(splits)):
                if splits[i].gain > best_split.gain:
                    best_split = splits[i]
        return best_split


class RandomSplitChooser(SplitChooser):
    def get_split(self, splits):
        split = None
        if len(splits) > 0:
            choice = np.random.randint(low=0, high=len(splits))
            split = splits[choice]
        return split


class KBestRandomSplitChooser(SplitChooser):
    def get_split(self, splits):
        split = None
        if len(splits) > 0:
            split_gains = [-split.gain for split in splits]
            sorted_args = np.argsort(split_gains)
            choice = np.random.randint(low=0, high=np.math.floor(np.math.sqrt(len(splits))))
            split = splits[sorted_args[choice]]
        return split


class WeightedBestSplitChooser(SplitChooser):
    def get_split(self, splits):
        best_split = None
        if len(splits) > 0:
            best_split = splits[0]
            for i in range(len(splits)):
                if splits[i].gain * splits[i].feature_weight > best_split.gain * best_split.feature_weight:
                    best_split = splits[i]
        return best_split


def resolve_split_selection(split_criterion):
    if split_criterion == 'best':
        return BestSplitChooser()
    elif split_criterion == 'rand':
        return RandomSplitChooser()
    elif split_criterion == 'krand':
        return KBestRandomSplitChooser()
    elif split_criterion == 'wbest':
        return WeightedBestSplitChooser()
    else:
        raise ValueError("%s is not a recognizable split chooser."
                         % split_criterion)
