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


class Sampler:
    def __init__(self, n_instances):
        self.n_instances = n_instances
        self.training_ids = None
        self.out_of_bag_ids = None

    def get_training_sample(self):
        if self.training_ids is None:
            self.training_ids = np.random.choice(self.n_instances, replace=True, size=self.n_instances)
        return self.training_ids

    def get_out_of_bag(self):
        if self.training_ids is not None:
            if self.out_of_bag_ids is None:
                self.out_of_bag_ids = self._calculate_out_of_bag_ids
            return self.out_of_bag_ids
        else:
            raise Exception('Training sample should be generated first')

    def _calculate_out_of_bag_ids(self):
        # TODO
        pass

