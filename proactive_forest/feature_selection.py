from abc import ABC, abstractmethod
import numpy as np


class FeatureSelection(ABC):
    @abstractmethod
    def get_features(self, n_features, prob):
        pass


class AllFeatureSelection(FeatureSelection):
    @property
    def name(self):
        return 'all'

    def get_features(self, n_features, prob=None):
        """
        Returns all features as candidates features in a list.

        :param n_features: <int> Amount of features to consider
        :param prob: <list> Probabilities of those features
        :return: <list>
        """
        return list(range(n_features))


class LogFeatureSelection(FeatureSelection):
    @property
    def name(self):
        return 'log'

    def get_features(self, n_features, prob=None):
        """
        Returns log(n_features)+1 candidate features in a list.

        :param n_features: <int> Amount of features to consider
        :param prob: <list> Probabilities of those features
        :return: <list>
        """
        sample_size = int(np.math.floor(np.math.log2(n_features)) + 1)
        population = list(range(n_features))
        selected = np.random.choice(population, replace=False, size=sample_size, p=prob)
        return selected


class ProbFeatureSelection(FeatureSelection):
    @property
    def name(self):
        return 'prob'

    def get_features(self, n_features, prob=None):
        """
        Returns the candidate features in a list according to its probabilities.
        The amount of features is not fixed. It is random.

        :param n_features: <int> Amount of features to consider
        :param prob: <list> Probabilities of those features
        :return: <list>
        """
        sample_size = n_features
        population = list(range(n_features))
        selected = np.random.choice(population, replace=True, size=sample_size, p=prob)
        return np.unique(selected)


def resolve_feature_selection(name):
    """
    Returns the class instance of the selected criterion.

    :param name: <string> Name of the criterion
    :return: <FeatureSelection>
    """
    if name == 'all':
        return AllFeatureSelection()
    elif name == 'log':
        return LogFeatureSelection()
    elif name == 'prob':
        return ProbFeatureSelection()
    else:
        raise ValueError('Unknown feature selection criterion {}'.format(name))
