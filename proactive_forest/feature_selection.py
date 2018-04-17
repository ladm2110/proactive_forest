from abc import ABC, abstractmethod
import numpy as np


class FeatureSelection(ABC):
    @abstractmethod
    def get_features(self, n_features, prob):
        pass


class AllFeatureSelection(FeatureSelection):
    def get_features(self, n_features, prob=None):
        """

        :param n_features:
        :param prob:
        :return:
        """
        return list(range(n_features))


class LogFeatureSelection(FeatureSelection):
    def get_features(self, n_features, prob=None):
        """

        :param n_features:
        :param prob:
        :return:
        """
        sample_size = np.math.floor(np.math.log2(n_features)) + 1
        population = list(range(n_features))
        selected = np.random.choice(population, replace=False, size=sample_size, p=prob)
        return selected


class ProbFeatureSelection(FeatureSelection):
    def get_features(self, n_features, prob=None):
        """

        :param n_features:
        :param prob:
        :return:
        """
        sample_size = n_features
        population = list(range(n_features))
        selected = np.random.choice(population, replace=True, size=sample_size, p=prob)
        return np.unique(selected)


def resolve_feature_selection(name):
    if name == 'all':
        return AllFeatureSelection()
    elif name == 'log':
        return LogFeatureSelection()
    elif name == 'prob':
        return ProbFeatureSelection()
    else:
        raise ValueError('Unknown feature selection criterion {}'.format(name))

