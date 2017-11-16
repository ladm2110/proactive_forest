import numpy as np


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
