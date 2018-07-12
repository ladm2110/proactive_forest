from abc import abstractmethod, ABC
import numpy as np
import proactive_forest.utils as utils


def compute_split_values(x):
    """
    Returns all possible cut points in a feature. For numerical data the max is not considered.

    :param x: <numpy array> Feature values
    :return: <numpy array>
    """
    if utils.categorical_data(x):
        return np.unique(x)
    else:
        uniques = np.unique(x)
        return np.array([(uniques[i]+uniques[i+1])/2 for i in range(len(uniques)-1)])


def compute_split_info(split_criterion, X, y, feature_id, split_value, n_leaf_min):
    """
    Computes the gain measure for splitting the data with feature_id at split_value.

    :param split_criterion: <SplitCriterion> The selected split criterion
    :param X: <numpy ndarray> Array containing the training set
    :param y: <numpy array> Array containing the target values
    :param feature_id: <int> The selected feature to split the training set
    :param split_value: <float> The value for which the feature is going to be split
    :param n_leaf_min: <int> Minimum number of instances in a leaf
    :return: <tuple or None>
    """
    y_left, y_right = split_target(X, y, feature_id, split_value)

    n_left, n_right = len(y_left), len(y_right)
    n_min = np.min([n_left, n_right])
    if n_min == 0 or n_min < n_leaf_min:
        return None

    gain = compute_split_gain(split_criterion, y, y_left, y_right)

    return gain, feature_id, split_value


def split_target(X, y, feature_id, value):
    """
    Splits the data, no matter if it is categorical or numerical.

    :param X: <numpy ndarray> Array containing the training set
    :param y: <numpy array> Array containing the target values
    :param feature_id: <int> The selected feature to split the training set
    :param value: <float> The value for which the feature is going to be split
    :return: <tuple> (X_left, X_right, y_left, y_right)
    """
    is_categorical = utils.categorical_data(X[:, feature_id])
    if is_categorical:
        return split_categorical_target(X, y, feature_id, value)
    else:
        return split_numerical_target(X, y, feature_id, value)


def split_categorical_data(X, y, feature_id, value):
    """
    Splits categorical data in the form
        - Left branch: Value
        - Right branch: Not Value

    :param X: <numpy ndarray> Array containing the training set
    :param y: <numpy array> Array containing the target values
    :param feature_id: <int> The selected feature to split the training set
    :param value: <float> The value for which the feature is going to be split
    :return: <tuple> (X_left, X_right, y_left, y_right)
    """
    mask = X[:, feature_id] == value
    return X[mask], X[~mask], y[mask], y[~mask]


def split_categorical_target(X, y, feature_id, value):
    """
    Splits categorical target in the form
        - Left branch: Value
        - Right branch: Not Value

    :param X: <numpy ndarray> Array containing the training set
    :param y: <numpy array> Array containing the target values
    :param feature_id: <int> The selected feature to split the training set
    :param value: <float> The value for which the feature is going to be split
    :return: <tuple> (X_left, X_right, y_left, y_right)
    """
    mask = X[:, feature_id] == value
    return y[mask], y[~mask]


def split_numerical_data(X, y, feature_id, value):
    """
    Splits numerical data in the form
        - Left branch: <= Value
        - Right branch: > Value

    :param X: <numpy ndarray> Array containing the training set
    :param y: <numpy array> Array containing the target values
    :param feature_id: <int> The selected feature to split the training set
    :param value: <float> The value for which the feature is going to be split
    :return: <tuple> (X_left, X_right, y_left, y_right)
    """
    mask = X[:, feature_id] <= value
    return X[mask], X[~mask], y[mask], y[~mask]


def split_numerical_target(X, y, feature_id, value):
    """
    Splits numerical target in the form
        - Left branch: <= Value
        - Right branch: > Value

    :param X: <numpy ndarray> Array containing the training set
    :param y: <numpy array> Array containing the target values
    :param feature_id: <int> The selected feature to split the training set
    :param value: <float> The value for which the feature is going to be split
    :return: <tuple> (X_left, X_right, y_left, y_right)
    """
    mask = X[:, feature_id] <= value
    return y[mask], y[~mask]


def compute_split_gain(split_criterion, y, y_left, y_right):
    """
    Computes the information gain measure.

    :param split_criterion: <SplitCriterion> The criterion used to measure the impurity gain
    :param y: <numpy array> Target features
    :param y_left: <numpy array> Target features of the left branch
    :param y_right: <numpy array> Target features of the right branch
    :return: <float>
    """
    return split_criterion.impurity(y) - split_criterion.impurity(y_left) * len(y_left) / len(y) \
                                       - split_criterion.impurity(y_right) * len(y_right) / len(y)


class Split:
    def __init__(self, feature_id, value, gain):
        """
        Constructs a tree split.

        :param feature_id: <int> Feature to be split
        :param value: <float> Cut point for the feature
        :param gain: <float> Impurity gain for the split
        """
        self.feature_id = feature_id
        self.value = value
        self.gain = gain


class SplitChooser(ABC):
    @abstractmethod
    def get_split(self, splits):
        pass


class BestSplitChooser(SplitChooser):
    @property
    def name(self):
        return 'best'

    def get_split(self, splits):
        """
        Selects the split with the highest impurity gain.

        :param splits: <list> All splits to consider
        :return: <Split>
        """
        best_split = None
        if len(splits) > 0:
            best_split = splits[0]
            for i in range(len(splits)):
                if splits[i].gain > best_split.gain:
                    best_split = splits[i]
        return best_split


class RandomSplitChooser(SplitChooser):
    @property
    def name(self):
        return 'rand'

    def get_split(self, splits):
        """
        Selects a random split from the candidates.

        :param splits: <list> All splits to consider
        :return: <Split>
        """
        split = None
        if len(splits) > 0:
            choice = np.random.randint(low=0, high=len(splits))
            split = splits[choice]
        return split


def resolve_split_selection(split_criterion):
    """
    Returns the class instance of the selected criterion.

    :param split_criterion: <string> Name of the criterion
    :return: <SplitChooser>
    """
    if split_criterion == 'best':
        return BestSplitChooser()
    elif split_criterion == 'rand':
        return RandomSplitChooser()
    else:
        raise ValueError("%s is not a recognizable split chooser."
                         % split_criterion)
