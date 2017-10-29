import numpy as np


def list_to_discrete_dist(x):
    """
    Takes a list of positive ints and turns it into a discrete distribution
    E.g.: list_to_discrete_dist([0,0,1,1,2]) => ([0, 1, 2], [0.4, 0.4, 0.2])
    """
    counts = np.bincount(x)
    unique_indices = np.nonzero(counts)[0]
    return unique_indices, counts[unique_indices] / float(x.size)


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
