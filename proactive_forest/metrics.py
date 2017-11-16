import numpy as np


class SplitCriterion:
    GINI = 'gini'
    ENTROPY = 'entropy'

    @staticmethod
    def resolve_split_criterion(criterion_name):
        if criterion_name == SplitCriterion.GINI:
            return gini_index
        elif criterion_name == SplitCriterion.ENTROPY:
            return entropy
        else:
            raise ValueError('Unknown criterion {}'.format(criterion_name))


def gini_index(x):
    if len(x) == 0:
        return 0.0
    counts = np.bincount(x)
    p = counts / float(len(x))
    return 1.0 - np.sum(p*p)


def entropy(x):
    if len(x) == 0:
        return 0.0
    counts = np.bincount(x)

    aux = list()
    for c in counts.tolist():
        if c != 0:
            aux.append(c)
    counts = np.array(aux)

    p = counts / float(len(x))
    p.dumps()
    return -np.sum(p * np.log(p))
