from proactive_forest import metrics
import numpy as np


def compute_split_info(args):
    split_criterion, X, y, feature_id, split_value = args
    # None is passed for categorical values
    if split_value is not None:
        splits = [y_split for _, y_split in split(X, y, feature_id, split_value)]
    else:
        splits = [y_split for _, _, y_split in split_categorical(X, y, feature_id)]
    if len(splits) == 1:
        return None
    gain = compute_split_gain(split_criterion, y, splits)
    return gain, np.min([len(s) for s in splits])


def split(X, y, feature_id, value):
    mask = X[:, feature_id] <= value
    return [(X[mask], y[mask]), (X[~mask], y[~mask])]


def split_categorical(X, y, feature_id):
    splits = []
    for key in set(X[:, feature_id]):
        ind_k = np.where(X[:, feature_id] == key)
        splits.append((key, X[ind_k], y[ind_k]))
    return splits


def compute_split_gain(split_criterion, y, splits):
    return split_criterion(y) - \
        sum([split_criterion(split) * float(len(split)) / len(y) for split in splits])


class SplitCriterion:
    GINI = 'gini'
    ENTROPY = 'entropy'

    @staticmethod
    def resolve_split_criterion(criterion_name):
        if criterion_name == SplitCriterion.GINI:
            return metrics.gini_index
        elif criterion_name == SplitCriterion.ENTROPY:
            return metrics.entropy
        else:
            raise ValueError('Unknown criterion {}'.format(criterion_name))
