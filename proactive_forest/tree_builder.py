import numpy as np
import scipy.stats
from random import choices
import pandas as pd

from proactive_forest.tree import BinaryDecisionTree, BinaryDecisionTreeSplit, DecisionTree, DecisionLeaf, DecisionFork, \
    DecisionForkCategorical, DecisionForkNumerical

from multiprocessing import Pool

from proactive_forest.utils import compute_split_info, compute_split_gain, SplitCriterion, split, \
    split_categorical


class TreeSplit(BinaryDecisionTreeSplit):
    def __init__(self, feature_id, value, gain):
        super(TreeSplit, self).__init__(feature_id, value)
        self.gain = gain


class Builder:
    def __init__(self,
                 max_depth=10,
                 min_samples_split=2,
                 min_samples_leaf=5,
                 max_n_splits=None,
                 criterion='gini',
                 feature_selection='all',
                 feature_prob=None,
                 min_gain_split=0.05,
                 categorical=[],
                 n_jobs=1):
        self.max_depth = max_depth
        self.max_n_splits = max_n_splits
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain_split = min_gain_split

        assert feature_selection in ['all', 'rand', 'prob']
        self.feature_selection = feature_selection
        self.feature_prob = feature_prob

        assert criterion in [SplitCriterion.GINI, SplitCriterion.ENTROPY]
        self.split_criterion = SplitCriterion.resolve_split_criterion(criterion)

        self.categorical = categorical

        self.n_jobs = n_jobs
        self.pool = None

    def build_tree(self, X, y):
        n_samples, n_features = X.shape
        tree = DecisionTree(n_features=n_features)
        if self.n_jobs > 1:
            self.pool = Pool(self.n_jobs)

        tree.last_node_id = tree.root()
        self._build_tree_recursive(tree, tree.last_node_id, X, y, depth=1)
        # self._prune_tree(tree, X, y)
        self.pool = None
        return tree

    def _build_tree_recursive(self, tree, cur_node, X, y, depth):
        n_samples, n_features = X.shape
        leaf_reached = False

        # Evaluates if all instances belong to the same class
        if self._all_same_class(y):
            leaf_reached = True

        # Evaluates the min_samples_leaf stopping criteria
        # it should be min_samples_split
        if n_samples < self.min_samples_split:
            leaf_reached = True

        # Evaluates the depth stopping criteria
        if self.max_depth is not None and depth >= self.max_depth:
            leaf_reached = True

        best_split = None
        if not leaf_reached:
            best_split = self._find_best_split(X, y)
            if best_split is None or best_split.gain < self.min_gain_split:
                leaf_reached = True

        if leaf_reached:
            result = scipy.stats.mode(y)
            probability = result.count[0]/len(y)
            tree.nodes.append(DecisionLeaf(result=result.mode[0], prob=probability, n_samples=len(y), depth=depth))

        else:
            if best_split.feature_id in self.categorical:
                tree.nodes.append(DecisionForkCategorical(n_samples=len(y), depth=depth,
                                                          feature_id=best_split.feature_id, gain=best_split.gain))
                for key, X_new, y_new in split_categorical(X, y, best_split.feature_id):
                    tree.last_node_id += 1
                    node_to_split = tree.last_node_id
                    tree.nodes[cur_node].add(key, self._build_tree_recursive(tree, node_to_split, X_new, y_new, depth=depth + 1))
            else:
                tree.nodes.append(DecisionForkNumerical(n_samples=len(y), depth=depth, feature_id=best_split.feature_id,
                                               value=best_split.value, gain=best_split.gain))
                for X_new, y_new in split(X, y, best_split.feature_id, best_split.value):
                    tree.last_node_id += 1
                    node_to_split = tree.last_node_id
                    tree.nodes[cur_node].add(self._build_tree_recursive(tree, node_to_split, X_new, y_new, depth=depth+1))

        return cur_node

    def _prune_tree(self, tree, X, y):
        # TODO: add tree pruning
        pass

    def _all_same_class(self, y):
        return scipy.stats.mode(y).count[0] == len(y)

    def _compute_split_values(self, X, y, feature_id):
        x = X[:, feature_id]
        if self.max_n_splits is None:
            split_values = list(set(x))
            return split_values
        else:
            d = max(1, int(100.0 / (self.max_n_splits + 1)))
            split_values = [np.percentile(x, p) for p in range(d, 100, d)]
            split_values = list(set(split_values))
            return split_values

    def _find_best_split(self, X, y):

        n_samples, n_features = X.shape
        args = []

        # Select features to consider
        features = self._select_features(self.feature_selection, n_features)

        for feature_id in features:
            if feature_id in self.categorical:
                args.append([self.split_criterion, X, y, feature_id, None])
            else:
                for split_value in self._compute_split_values(X, y, feature_id):
                    args.append([self.split_criterion, X, y, feature_id, split_value])
        if self.pool is not None:
            split_infos = self.pool.map(compute_split_info, args)
        else:
            split_infos = map(compute_split_info, args)

        splits = []
        for arg, split_info in zip(args, split_infos):
            if split_info is not None:
                _, _, _, feature_id, split_value = arg
                gain, n_min = split_info
                if n_min < self.min_samples_leaf:
                    continue
                if gain is not None and gain > 0:
                    split = TreeSplit(feature_id, value=split_value, gain=gain)
                    splits.append(split)
            else:
                continue

        # Find best split
        best_split = None
        for split in splits:
            if best_split is None or split.gain > best_split.gain:
                    best_split = split
        return best_split

    def _select_features(self, criterion='all', n_features=1):
        if criterion == 'all':
            return list(range(n_features))
        elif criterion == 'rand':
            population = list(range(n_features))
            weights = [1/n_features for _ in range(n_features)]
            selected = choices(population, weights, k=n_features)
            return pd.unique(selected)
        elif criterion == 'prob':
            population = list(range(n_features))
            weights = self.feature_prob
            selected = choices(population, weights, k=n_features)
            return pd.unique(selected)


