from abc import ABC, abstractmethod
import numpy as np


class DecisionTree:
    def __init__(self, n_features):
        self._n_features = n_features
        self._nodes = []
        self._last_node_id = None
        self._weight = 1

    @property
    def n_features(self):
        return self._n_features

    @n_features.setter
    def n_features(self, n_features):
        self._n_features = n_features

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, nodes):
        self._nodes = nodes

    @property
    def last_node_id(self):
        return self._last_node_id

    @last_node_id.setter
    def last_node_id(self, last_node_id):
        self._last_node_id = last_node_id

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, weight):
        self._weight = weight

    @staticmethod
    def root():
        """
        Return the position of the root node.
        """
        return 0

    def predict(self, x):
        """

        Predicts for a given array x the class to which it belongs.

        Example:
            >> p = tree.predict(numpy.array([0.2, 1, 4.5]))
            >> p
            1

        :param x: <numpy array> Feature vector
        :return: <int>
        """
        current_node = self.root()
        leaf_found = False
        prediction = None
        while not leaf_found:
            if isinstance(self._nodes[current_node], DecisionLeaf):
                leaf_found = True
                prediction = self._nodes[current_node].result
            else:
                current_node = self._nodes[current_node].result_branch(x)
        return prediction

    def predict_proba(self, x):
        """
        Predicts for a given array x the class probability estimates
        using frequency-based Laplace correction.

        Example:
             >> p = tree.predict_proba(numpy.array([0.2, 1, 4.5]))
             >> p
             [0.23, 0.77]

        :param x: <numpy array> Feature vector
        :return: <list>
        """
        current_node = self.root()
        leaf_found = False
        class_proba = None
        while not leaf_found:
            if isinstance(self._nodes[current_node], DecisionLeaf):
                leaf_found = True
                class_proba = [n + 1 for n in self._nodes[current_node].samples] / \
                              (np.sum(self._nodes[current_node].samples) + len(self._nodes[current_node].samples))
            else:
                current_node = self._nodes[current_node].result_branch(x)
        return class_proba.tolist()

    def feature_importances(self):
        """
        Calculates the feature importances according to Breiman 2001.

        :return: <numpy array>
        """
        importances = np.zeros(self._n_features)
        for node in self._nodes:
            if isinstance(node, DecisionFork):
                importances[node.feature_id] += node.gain * np.sum(node.samples) / np.sum(
                    self._nodes[self.root()].samples)

        normalizer = np.sum(importances)
        if normalizer > 0:
            # Avoid dividing by 0
            importances /= normalizer

        return importances

    def total_nodes(self):
        """
        Returns the amount of nodes in the decision tree.

        :return: <int>
        """
        return len(self._nodes)

    def total_splits(self):
        """
        Returns the amount of splits done in the decision tree.

        :return: <int>
        """
        count = 0
        for node in self._nodes:
            if isinstance(node, DecisionFork):
                count += 1
        return count

    def total_leaves(self):
        """
        Returns the amount of leaves in the decision tree.

        :return: <int>
        """
        count = 0
        for node in self._nodes:
            if isinstance(node, DecisionLeaf):
                count += 1
        return count


class DecisionNode(ABC):
    def __init__(self, samples, depth):
        """
        Creates a decision node for the tree.

        :param samples: <list> Distribution of instances per class
        :param depth: <int> Depth in the tree
        """
        self._samples = samples
        self._depth = depth
        super().__init__()

    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, samples):
        self._samples = samples

    @property
    def depth(self):
        return self._depth

    @depth.setter
    def depth(self, depth):
        self._depth = depth


class DecisionFork(DecisionNode):
    def __init__(self, samples, depth, feature_id, gain, value):
        """
        Creates a decision fork for the tree.

        :param samples: <list> Distribution of instances per class
        :param depth: <int> Depth in the tree
        :param feature_id: <int> Split feature
        :param gain: <float> Impurity gain of the split
        :param value: <float> Cut point of the feature
        """
        self._feature_id = feature_id
        self._gain = gain
        self._left_branch = None
        self._right_branch = None
        self._value = value
        super().__init__(samples, depth)

    @property
    def feature_id(self):
        return self._feature_id

    @feature_id.setter
    def feature_id(self, feature_id):
        self._feature_id = feature_id

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, gain):
        self._gain = gain

    @property
    def left_branch(self):
        return self._left_branch

    @left_branch.setter
    def left_branch(self, left_branch):
        self._left_branch = left_branch

    @property
    def right_branch(self):
        return self._right_branch

    @right_branch.setter
    def right_branch(self, right_branch):
        self._right_branch = right_branch

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    @abstractmethod
    def result_branch(self, x):
        pass


class DecisionForkNumerical(DecisionFork):
    def result_branch(self, x):
        """
        Evaluates the feature vector x and return the id of the next node in the path.

        :param x: <numpy array> Feature vector
        :return: <int>
        """
        if x[self.feature_id] <= self.value:
            return self.left_branch
        else:
            return self.right_branch


class DecisionForkCategorical(DecisionFork):
    def result_branch(self, x):
        """
        Evaluates the feature vector x and return the id of the next node in the path.

        :param x: <numpy array> Feature vector
        :return: <int>
        """
        if x[self.feature_id] == self.value:
            return self.left_branch
        else:
            return self.right_branch


class DecisionLeaf(DecisionNode):
    def __init__(self, samples, depth, result):
        """
        Creates a decision leaf for the tree.

        :param samples: <list> Distribution of instances per class
        :param depth: <int> Depth in the tree
        :param result: <int> Class of the leaf
        """
        super().__init__(samples, depth)
        self._result = result

    @property
    def result(self):
        return self._result

    @result.setter
    def result(self, result):
        self._result = result
