import numpy as np
from abc import ABC, abstractmethod


GINI = 'gini'
ENTROPY = 'entropy'


def resolve_split_criterion(criterion_name):
    if criterion_name == GINI:
        return GiniCriterion()
    elif criterion_name == ENTROPY:
        return EntropyCriterion()
    else:
        raise ValueError('Unknown criterion {}'.format(criterion_name))


class SplitCriterion(ABC):

    @abstractmethod
    def impurity_gain(self, x):
        pass


class GiniCriterion(SplitCriterion):
    def impurity_gain(self, x):
        if len(x) == 0:
            return 0.0
        counts = np.bincount(x)
        prob = counts / float(len(x))
        return 1.0 - np.sum(prob * prob)


class EntropyCriterion(SplitCriterion):
    def impurity_gain(self, x):
        if len(x) == 0:
            return 0.0
        counts = np.bincount(x)
        prob = counts / float(len(x))
        return -np.sum(p * np.log2(p) for p in prob if p != 0)
