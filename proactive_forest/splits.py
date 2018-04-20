from abc import abstractmethod, ABC
import numpy as np
import proactive_forest.utils as utils


def compute_split_values(x):
    """
    Returns all possible cut points in a feature. For numerical data the max is not considered.

    :param x: Numpy Array
    :return:
        An array containing the cut points for the given array.
    """
    if utils.categorical_data(x):
        return np.unique(x)
    else:
        uniques = np.unique(x)
        return np.array([(uniques[i]+uniques[i+1])/2 for i in range(len(uniques)-1)])


def compute_split_info(args):
    """
    Computes the gain measure for splitting the data with feature_id at split_value.

    :param args:
    :return:
        A tuple in the form (gain, min_size_split) containing the gain info for the split.
    """
    split_criterion, X, y, feature_id, split_value = args

    _, _, y_left, y_right = split_data(X, y, feature_id, split_value)

    n_left, n_right = len(y_left), len(y_right)
    if n_left == 0 or n_right == 0:
        return None

    gain = compute_split_gain(split_criterion, y, y_left, y_right)
    return gain, np.min([n_left, n_right])


def split_data(X, y, feature_id, value):
    """
    Splits the data, no matter if it is categorical or numerical.

    :param X: Ndarray containing the training set.
    :param y: Array containing the target values.
    :param feature_id: The selected feature to split the training set.
    :param value: The value for which the feature is going to be split.
    :return:
        A tuple in the form (X_left, X_right, y_left, y_right)
    """
    is_categorical = utils.categorical_data(X[:, feature_id])
    if is_categorical:
        return split_categorical_data(X, y, feature_id, value)
    else:
        return split_numerical_data(X, y, feature_id, value)


def split_categorical_data(X, y, feature_id, value):
    """
    Splits categorical data in the form
        - Left branch: Value
        - Right branch: Not Value

    :param X:
    :param y:
    :param feature_id:
    :param value:
    :return:
        A tuple in the form (X_left, X_right, y_left, y_right)
    """
    mask = X[:, feature_id] == value
    return X[mask], X[~mask], y[mask], y[~mask]


def split_numerical_data(X, y, feature_id, value):
    """
    Splits categorical data in the form
        - Left branch: <= Value
        - Right branch: > Value

    :param X:
    :param y:
    :param feature_id:
    :param value:
    :return:
        A tuple in the form (X_left, X_right, y_left, y_right)
    """
    mask = X[:, feature_id] <= value
    return X[mask], X[~mask], y[mask], y[~mask]


def compute_split_gain(split_criterion, y, y_left, y_right):
    """
    Computes the information gain measure.

    :param split_criterion:
    :param y:
    :param y_left:
    :param y_right:
    :return:
    """
    return split_criterion.impurity_gain(y) - split_criterion.impurity_gain(y_left) * len(y_left) / len(y) \
                                            - split_criterion.impurity_gain(y_right) * len(y_right) / len(y)


class Split:
    def __init__(self, feature_id, value, gain):
        self.feature_id = feature_id
        self.value = value
        self.gain = gain


class SplitChooser(ABC):
    @abstractmethod
    def get_split(self, splits):
        pass


class BestSplitChooser(SplitChooser):
    def get_split(self, splits):
        """

        :param splits:
        :return:
        """
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


def resolve_split_selection(split_criterion):
    if split_criterion == 'best':
        return BestSplitChooser()
    elif split_criterion == 'rand':
        return RandomSplitChooser()
    else:
        raise ValueError("%s is not a recognizable split chooser."
                         % split_criterion)
