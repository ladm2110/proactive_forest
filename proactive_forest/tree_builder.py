import numpy as np
from proactive_forest.feature_selection import resolve_feature_selection
from proactive_forest.tree import DecisionTree, DecisionLeaf, DecisionForkCategorical, DecisionForkNumerical
from proactive_forest.splits import compute_split_info, split_categorical_data, split_numerical_data, \
    resolve_split_selection, Split, compute_split_values
from proactive_forest.metrics import resolve_split_criterion
import proactive_forest.utils as utils


class TreeBuilder:
    def __init__(self,
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 criterion='gini',
                 max_features='all',
                 feature_prob=None,
                 min_gain_split=0,
                 split='best'):
        """
        Creates a Decision Tree Builder.

        :param max_depth: <int> or <None> Defines the maximum depth of the tree
        :param min_samples_split: <int> Minimum number of instances to consider creating a split
        :param min_samples_leaf: <int> Minimum number of instances to place in a leaf
        :param criterion: <string> "gini" for using Gini Criterion or "entropy" for Entropy Criterion
        :param max_features: <string> "all" for using all features as candidates for a split,
                            "log" for using log(n)+1 or "prob" to let the probabilities decide
        :param feature_prob: <list> Feature probabilities
        :param min_gain_split: <float> Minimum split gain value to consider splitting a node
        :param split: <string> "best" for selecting the best possible split or "rand" for selecting
                            a random split
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain_split = min_gain_split
        self.max_features = max_features
        self.feature_prob = feature_prob
        self.split_criterion = resolve_split_criterion(criterion)
        self.split = split

        self.n_classes = None

    def build_tree(self, X, y, n_classes):
        """
        Constructs a decision tree using the (X, y) as training set.

        :param X: <numpy ndarray> An array containing the feature vectors
        :param y: <numpy array> An array containing the target features
        :param n_classes: <int> Number of classes
        :return: <DecisionTree>
        """
        n_samples, n_features = X.shape
        self.n_classes = n_classes

        tree = DecisionTree(n_features=n_features)

        tree.last_node_id = tree.root()
        self._build_tree_recursive(tree, tree.last_node_id, X, y, depth=1)
        return tree

    def _build_tree_recursive(self, tree, cur_node, X, y, depth):
        """
        Algorithm to build the decision tree in a recursive manner.

        :param tree: <DecisionTree> The decision tree to be constructed
        :param cur_node: <int> Node id to be processed
        :param X: <numpy ndarray> An array containing the feature vectors
        :param y: <numpy array> An array containing the target features
        :param depth: <int> Current depth of the tree
        :return: <int>
        """
        n_samples, n_features = X.shape
        leaf_reached = False

        # Evaluates if all instances belong to the same class
        if utils.all_instances_same_class(y):
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
            best_split = self._find_split(X, y)
            if best_split is None or best_split.gain < self.min_gain_split:
                leaf_reached = True

        if leaf_reached:

            samples = utils.bin_count(y, length=self.n_classes)
            result = np.argmax(samples)
            new_leaf = DecisionLeaf(samples=samples, depth=depth, result=result)
            tree.nodes.append(new_leaf)

        else:

            is_categorical = utils.categorical_data(X[:, best_split.feature_id])
            samples = utils.bin_count(y, length=self.n_classes)

            if is_categorical:

                new_fork = DecisionForkCategorical(samples=samples, depth=depth,
                                                   feature_id=best_split.feature_id, value=best_split.value,
                                                   gain=best_split.gain)
                X_left, X_right, y_left, y_right = split_categorical_data(X, y, best_split.feature_id, best_split.value)

            else:

                new_fork = DecisionForkNumerical(samples=samples, depth=depth,
                                                 feature_id=best_split.feature_id, value=best_split.value,
                                                 gain=best_split.gain)
                X_left, X_right, y_left, y_right = split_numerical_data(X, y, best_split.feature_id, best_split.value)

            tree.nodes.append(new_fork)
            tree.last_node_id += 1
            node_to_split = tree.last_node_id
            new_branch = self._build_tree_recursive(tree, node_to_split, X_left, y_left, depth=depth+1)
            tree.nodes[cur_node].left_branch = new_branch

            tree.last_node_id += 1
            node_to_split = tree.last_node_id
            new_branch = self._build_tree_recursive(tree, node_to_split, X_right, y_right, depth=depth+1)
            tree.nodes[cur_node].right_branch = new_branch

        return cur_node

    def _find_split(self, X, y):
        """
        Computes all possible split and selects the split according to the criterion.

        :param X: <numpy ndarray> An array containing the feature vectors
        :param y: <numpy array> An array containing the target features
        :return: <Split>
        """
        n_samples, n_features = X.shape
        args = []

        # Select features to consider
        selection = resolve_feature_selection(self.max_features)
        features = selection.get_features(n_features, self.feature_prob)

        # Get candidate splits
        for feature_id in features:
            for split_value in compute_split_values(X[:, feature_id]):
                args.append([self.split_criterion, X, y, feature_id, split_value])

        # Compute splits info
        splits_info = map(compute_split_info, args)

        splits = []
        for arg, split_info in zip(args, splits_info):
            if split_info is not None:
                _, _, _, feature_id, split_value = arg
                gain, n_min = split_info
                if n_min < self.min_samples_leaf:
                    continue
                if gain is not None and gain > 0:
                    split = Split(feature_id, value=split_value, gain=gain)
                    splits.append(split)
            else:
                continue

        selected_split = resolve_split_selection(self.split).get_split(splits)

        return selected_split



