import numpy as np
from abc import ABC, abstractmethod


class SplitCriterion(ABC):
    @abstractmethod
    def impurity(self, x):
        pass


class GiniCriterion(SplitCriterion):
    @property
    def name(self):
        return 'gini'

    def impurity(self, x):
        """
        Calculates the Gini metric.

        :param x: <numpy array> Target values
        :return: <float>
        """
        if len(x) == 0:
            return 0.0
        counts = np.bincount(x)
        prob = counts / float(len(x))
        return 1.0 - np.sum(prob * prob)


class EntropyCriterion(SplitCriterion):
    @property
    def name(self):
        return 'entropy'

    def impurity(self, x):
        """
        Calculates the Entropy metric.

        :param x: <numpy array> Target values
        :return: <float>
        """
        if len(x) == 0:
            return 0.0
        counts = np.bincount(x)
        prob = counts / float(len(x))
        return -np.sum(p * np.log2(p) for p in prob if p != 0)


def resolve_split_criterion(name):
    """
    Returns the class instance of the selected criterion.

    :param name: <string> Name of the criterion
    :return: <SplitCriterion>
    """
    if name == 'gini':
        return GiniCriterion()
    elif name == 'entropy':
        return EntropyCriterion()
    else:
        raise ValueError('Unknown criterion {}'.format(name))
