import numpy as np


def all_instances_same_class(x):
    """
    Returns True only if all belongs to the same class, False otherwise.

    >> all_instances_same_class(np.array([0, 0, 0, 0]))
    True

    >> all_instances_same_class(np.array([1, 0, 2, 0]))
    False

    :param x: <numpy array> An array with the class values
    :return: <bool>
    """
    return len(np.unique(x)) == 1


def categorical_data(x):
    """
    Returns True only if all objects are categorical, False otherwise.

    :param x: <numpy array> An array with the feature values
    :return: <bool>
    """
    return isinstance(x[0], str)


def bin_count(x, length):
    """
    Counts the number of times a value appears in an array.

    :param x: <numpy array> An array containing the values to count
    :param length: <int> The length of the returned array
    :return: <list>
    """
    results = np.zeros(length, dtype=int)
    for i in x:
        results[i] += 1
    return results.tolist()


def count_classes(x):
    """
    Counts the number of classes in an array.

    :param x: <numpy array> An array containing the classes
    :return: <int>
    """
    return len(np.unique(x))


def check_positive_array(x):
    array = np.array(x)
    return all(array > 0)


def check_array_sum_one(x):
    array = np.array(x)
    return sum(array) == 1
