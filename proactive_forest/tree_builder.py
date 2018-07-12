import numpy as np
import proactive_forest.utils as utils
from proactive_forest.tree import DecisionTree, DecisionLeaf, DecisionForkCategorical, DecisionForkNumerical
from proactive_forest.splits import compute_split_info, split_categorical_data, split_numerical_data, Split, \
    compute_split_values


class TreeBuilder:
    def __init__(self,
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 split_criterion=None,
                 feature_selection=None,
                 feature_prob=None,
                 min_gain_split=0,
                 split_chooser=None):
        """
        Creates a Decision Tree Builder.

        :param max_depth: <int> or <None> Defines the maximum depth of the tree
        :param min_samples_split: <int> Minimum number of instances to consider creating a split
        :param min_samples_leaf: <int> Minimum number of instances to place in a leaf
        :param split_criterion: <SplitCriterion> The split criterion, it can be GiniCriterion
                            or EntropyCriterion
        :param feature_selection: <FeatureSelection> The feature selection criterion, it can be
                            AllFeatureSelection, LogFeatureSelection or ProbFeatureSelection
        :param feature_prob: <list> Feature probabilities
        :param min_gain_split: <float> Minimum split gain value to consider splitting a node
        :param split_chooser: <SplitChooser> The split chooser criterion, it can be BestSplitChooser
                            or RandomSplitChooser
        """
        self._n_classes = None
        self._max_depth = None
        self._split_criterion = None
        self._split_chooser = None
        self._feature_selection = None
        self._min_samples_split = None
        self._min_samples_leaf = None
        self._min_gain_split = None
        self._feature_prob = None

        if max_depth is None or max_depth > 0:
            self._max_depth = max_depth
        else:
            raise(ValueError("The depth of the tree must be greater than 0."))

        if split_criterion is not None:
            self._split_criterion = split_criterion
        else:
            raise (ValueError("The split criterion can not be None."))

        if split_chooser is not None:
            self._split_chooser = split_chooser
        else:
            raise (ValueError("The split chooser can not be None."))

        if feature_selection is not None:
            self._feature_selection = feature_selection
        else:
            raise (ValueError("The feature selection can not be None."))

        if min_samples_split is not None and min_samples_split > 1:
            self._min_samples_split = min_samples_split
        else:
            raise(ValueError("The min_samples_split must be greater than 1."))

        if min_samples_leaf is not None and min_samples_leaf > 0:
            self._min_samples_leaf = min_samples_leaf
        else:
            raise(ValueError("The min_samples_leaf must be greater than 0."))

        if min_gain_split is not None and min_gain_split >= 0:
            self._min_gain_split = min_gain_split
        else:
            raise(ValueError("The min_gain_split must be greater or equal than 0."))

        if feature_prob is not None:
            self._feature_prob = feature_prob

    @property
    def max_depth(self):
        return self._max_depth

    @max_depth.setter
    def max_depth(self, max_depth):
        self._max_depth = max_depth

    @property
    def min_samples_leaf(self):
        return self._min_samples_leaf

    @min_samples_leaf.setter
    def min_samples_leaf(self, min_samples_leaf):
        self._min_samples_leaf = min_samples_leaf

    @property
    def min_samples_split(self):
        return self._min_samples_split

    @min_samples_split.setter
    def min_samples_split(self, min_samples_split):
        self._min_samples_split = min_samples_split

    @property
    def min_gain_split(self):
        return self._min_gain_split

    @min_gain_split.setter
    def min_gain_split(self, min_gain_split):
        self._min_gain_split = min_gain_split

    @property
    def split_chooser(self):
        return self._split_chooser

    @split_chooser.setter
    def split_chooser(self, split_chooser):
        self._split_chooser = split_chooser

    @property
    def split_criterion(self):
        return self._split_criterion

    @split_criterion.setter
    def split_criterion(self, split_criterion):
        self._split_criterion = split_criterion

    @property
    def feature_selection(self):
        return self._feature_selection

    @feature_selection.setter
    def feature_selection(self, feature_selection):
        self._feature_selection = feature_selection

    @property
    def feature_prob(self):
        return self._feature_prob

    @feature_prob.setter
    def feature_prob(self, feature_prob):
        self._feature_prob = feature_prob

    def build_tree(self, X, y, n_classes):
        """
        Constructs a decision tree using the (X, y) as training set.

        :param X: <numpy ndarray> An array containing the feature vectors
        :param y: <numpy array> An array containing the target features
        :param n_classes: <int> Number of classes
        :return: <DecisionTree>
        """
        n_samples, n_features = X.shape

        if n_classes > 0:
            self._n_classes = n_classes
        else:
            raise ValueError("The number of classes must be greater than 0.")

        if self._feature_prob is None:
            initial_prob = 1 / n_features
            self._feature_prob = [initial_prob for _ in range(n_features)]
        else:
            if len(self._feature_prob) != n_features:
                raise ValueError('The number of features does not match the given probabilities list.')

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

        # Evaluates the min_samples_split stopping criteria
        if n_samples < self._min_samples_split:
            leaf_reached = True

        # Evaluates the depth stopping criteria
        if self._max_depth is not None and depth >= self._max_depth:
            leaf_reached = True

        best_split = None
        if not leaf_reached:
            best_split = self._find_split(X, y, n_features)
            if best_split is None or best_split.gain < self._min_gain_split:
                leaf_reached = True

        if leaf_reached:
            samples = utils.bin_count(y, length=self._n_classes)
            result = np.argmax(samples)
            new_leaf = DecisionLeaf(samples=samples, depth=depth, result=result)
            tree.nodes.append(new_leaf)

        else:
            is_categorical = utils.categorical_data(X[:, best_split.feature_id])
            samples = utils.bin_count(y, length=self._n_classes)

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

    def _find_split(self, X, y, n_features):
        """
        Computes all possible split and selects the split according to the criterion.

        :param X: <numpy ndarray> An array containing the feature vectors
        :param y: <numpy array> An array containing the target features
        :param n_features: <int> Amount of features
        :return: <Split>
        """
        splits_info = []

        # Select features to consider
        features = self._feature_selection.get_features(n_features, self._feature_prob)

        # Get candidate splits
        for feature_id in features:
            for split_value in compute_split_values(X[:, feature_id]):
                splits_info.append(
                    compute_split_info(self._split_criterion, X, y, feature_id, split_value, self._min_samples_leaf))

        splits = []
        for split_info in splits_info:
            if split_info is not None:
                gain, feature_id, split_value = split_info
                split = Split(feature_id, value=split_value, gain=gain)
                splits.append(split)
            else:
                continue

        selected_split = self._split_chooser.get_split(splits)
        return selected_split
