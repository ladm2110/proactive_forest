import numpy as np


def all_instances_same_class(x):
    """
    Returns True only if all belongs to the same class, False otherwise.

    >> all_instances_same_class(np.array([0, 0, 0, 0]))
    True

    >> all_instances_same_class(np.array([1, 0, 2, 0]))
    False

    :param x: A Numpy array or list with the class values.
    :return:
        A bool value
    """
    return len(np.unique(x)) == 1


def categorical_data(x):
    """
    Returns True only if all objects are categorical, False otherwise.

    :param x:
    :return:
    """
    return x.dtype == object or x.dtype == str


def bin_count(x, length):
    """
    Counts the number of times a value appears in an array.

    :param x: A Numpy array containing the values to count.
    :param length: The length of the returned array.
    :return:
        An array
    """
    results = np.zeros(length)
    for i in x:
        results[i] += 1
    return results.tolist()
