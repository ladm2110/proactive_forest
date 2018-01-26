import numpy.random as rand
import math
from pandas import unique


class FeatureSelection:

    @staticmethod
    def resolve_feature_selection(self, selection):
        if self.feature_selection == 'all':
            return all_features
        elif self.feature_selection == 'log':
            return log2_features
        elif self.feature_selection == 'log_prob':
            return prob_log2_features
        elif self.feature_selection == 'prob':
            return prob_features


def all_features(n_features, prob):
    return list(range(n_features))


def log2_features(n_features, prob):
    sample_size = math.floor(math.log2(n_features)) + 1
    population = list(range(n_features))
    selected = rand.choice(population, replace=False, size=sample_size)
    return selected


def prob_log2_features(n_features, prob):
    sample_size = math.floor(math.log2(n_features)) + 1
    population = list(range(n_features))
    selected = rand.choice(population, replace=False, size=sample_size, p=prob)
    return selected


def prob_features(n_features, prob):
    sample_size = n_features
    population = list(range(n_features))
    selected = rand.choice(population, replace=True, size=sample_size, p=prob)
    return unique(selected)
