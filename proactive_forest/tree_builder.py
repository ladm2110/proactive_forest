import numpy as np
import scipy.stats
import logging
from random import choices
import pandas as pd

from proactive_forest.metrics import list_to_discrete_dist
from proactive_forest.tree import BinaryDecisionTree, BinaryDecisionTreeSplit

from multiprocessing import Pool

from proactive_forest.utils import split_dataset, compute_split_info, compute_split_gain, SplitCriterion


class TreeSplit(BinaryDecisionTreeSplit):
    def __init__(self, feature_id, value, gain):
        super(TreeSplit, self).__init__(feature_id, value)
        self.gain = gain


class TreeBuilder:
    """
    TreeBuilder implements an algorithm (currently, without pruning phase)
    for building decision trees.
    """

    logger = logging.getLogger("TreeBuilder")
    debug = False

    def __init__(self,
                 max_depth=10,
                 min_samples_leaf=5,
                 max_n_splits=None,
                 criterion='gini',
                 feature_selection='all',
                 feature_probs=None,
                 n_jobs=1):
        """
        Initializes proactive_forest new tree builder and validates the parameters.

        Args:
            problem (string):
                Can be either 'classification' or 'regression'

            max_depth (int): default is 10
                A stopping criteria, the maximum depth of the tree

            min_samples_per_leaf (int): default is 5
                A stopping criteria, stop building the subtree if
                the number of samples in it is less than `min_samples_per_leaf`

            max_n_splits (int): default is 1
                Number of splitting values to consider when choosing the best split

            criterion (string):
                A criterion used for estimating quality of a split.
                It can be either 'gini' or 'entropy'.

            n_jobs (int): the size of process pool to use when building the tree.
                When None, use the number of cores in the system
        """

        self.max_depth = max_depth
        self.max_n_splits = max_n_splits
        self.min_samples_leaf = min_samples_leaf

        assert feature_selection in ['all', 'rand', 'prob']
        self.feature_selection = feature_selection
        self.feature_probs = feature_probs

        assert criterion in [SplitCriterion.GINI, SplitCriterion.ENTROPY]
        self.split_criterion = SplitCriterion.resolve_split_criterion(criterion)

        self.n_jobs = n_jobs
        self.pool = None

    def build_tree(self, X, y):
        """
        Args:
            X: object-features matrix
            y: target vector

        Returns:
            A `BinaryDecisionTree` fitted to the dataset.

            The actual structure of the tree depends both on dataset and the parameters
            passed to the `TreeBuilder` constructor.

        """
        n_samples, n_features = X.shape
        tree = BinaryDecisionTree(n_features=n_features)
        if self.n_jobs > 1:
            self.pool = Pool(self.n_jobs)

        leaf_to_split = tree.root()
        self._build_tree_recursive(tree, leaf_to_split, X, y)
        self._prune_tree(tree, X, y)
        if TreeBuilder.debug:
            TreeBuilder.logger.debug(tree)
        self.pool = None
        return tree

    def _build_tree_recursive(self, tree, cur_node, X, y):

        n_samples, n_features = X.shape

        # Evaluates if there is no instance
        if n_samples == 0:
            return

        leaf_reached = False

        # Evaluates the min_samples_leaf stopping criteria
        if n_samples <= self.min_samples_leaf:
            leaf_reached = True

        # Evaluates the depth stopping criteria
        depth = tree.depth(cur_node)
        if self.max_depth is not None and depth >= self.max_depth:
            leaf_reached = True

        best_split = None
        if not leaf_reached:
            if TreeBuilder.debug:
                TreeBuilder.logger.debug('Split at {}, n = {}'.format(cur_node, n_samples))

            best_split = self._find_best_split(X, y)
            if best_split is None:
                leaf_reached = True

        tree._leaf_n_samples[cur_node] = len(y)
        if leaf_reached:
            tree._leaf_values[cur_node] = scipy.stats.mode(y).mode[0]
        else:
            tree.split_node(cur_node, best_split)

            left_child = tree.left_child(cur_node)
            right_child = tree.right_child(cur_node)
            X_left, X_right, y_left, y_right = split_dataset(
                    X, y, best_split.feature_id, best_split.value)
            self._build_tree_recursive(tree, left_child, X_left, y_left)
            self._build_tree_recursive(tree, right_child, X_right, y_right)

    def _prune_tree(self, tree, X, y):
        # TODO: add tree pruning
        pass

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
            for split_value in self._compute_split_values(X, y, feature_id):
                args.append([self.split_criterion, X, y, feature_id, split_value])
        if self.pool is not None:
            split_infos = self.pool.map(compute_split_info, args)
        else:
            split_infos = map(compute_split_info, args)

        splits = []
        for arg, split_info in zip(args, split_infos):
            _, _, _, feature_id, split_value = arg
            gain, n_left, n_right = split_info
            if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                continue
            if gain is not None and gain > 0:
                split = TreeSplit(feature_id, value=split_value, gain=gain)
                splits.append(split)

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
            print(pd.unique(selected))
            return pd.unique(selected)
        elif criterion == 'prob':
            population = list(range(n_features))
            weights = self.feature_probs
            selected = choices(population, weights, k=n_features)
            print(pd.unique(selected))
            return pd.unique(selected)

