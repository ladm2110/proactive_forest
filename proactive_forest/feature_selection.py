import numpy.random as rand
import math
from pandas import unique


def all_features(n_features, _):
    return list(range(n_features))


def log_features(n_features, _):
    sample_size = math.floor(math.log2(n_features)) + 1
    population = list(range(n_features))
    selected = rand.choice(population, replace=False, size=sample_size)
    return selected


def prob_features(n_features, prob):
    sample_size = n_features
    population = list(range(n_features))
    selected = rand.choice(population, replace=True, size=sample_size, p=prob)
    return unique(selected)
