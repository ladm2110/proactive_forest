from abc import ABC, abstractmethod
import numpy as np


class DecisionTree:
    def __init__(self, n_features):
        self.n_features = n_features
        self.nodes = []
        self.last_node_id = None
        self.weight = 1

    @staticmethod
    def root():
        """Return the position of the root node."""
        return 0

    def predict(self, x):
        """
        Predicts for a given array x the class to which it belongs.

        Params:
            x - Numpy array

        Returns:
            An integer with the class prediction.

        Example:
            >> p = tree.predict(numpy.array([0.2, 1, 4.5]))
            >> p
            1
        """
        current_node = self.root()
        leaf_found = False
        while not leaf_found:
            if isinstance(self.nodes[current_node], DecisionLeaf):
                leaf_found = True
                prediction = self.nodes[current_node].result
            else:
                current_node = self.nodes[current_node].result_branch(x)
        return prediction

    def predict_proba(self, x):
        """
        Predicts for a given array x the class probability estimates
        using frequency-based Laplace correction.

        Params:
            x - Numpy array

        Returns:
            A list with the class probability estimates.

        Example:
             >> p = tree.predict_proba(numpy.array([0.2, 1, 4.5]))
             >> p
             [0.23, 0.77]
        """
        current_node = self.root()
        leaf_found = False
        while not leaf_found:
            if isinstance(self.nodes[current_node], DecisionLeaf):
                leaf_found = True
                class_proba = [n + 1 for n in self.nodes[current_node].samples] / \
                              (np.sum(self.nodes[current_node].samples) + len(self.nodes[current_node].samples))
            else:
                current_node = self.nodes[current_node].result_branch(x)
        return class_proba.tolist()

    def feature_levels(self):
        return [(n.feature_id, n.depth) for n in self.nodes if isinstance(n, DecisionFork)]

    def feature_importances(self):
        importances = np.zeros(self.n_features)
        for node in self.nodes:
            if isinstance(node, DecisionFork):
                importances[node.feature_id] += node.gain * np.sum(node.samples) / np.sum(
                    self.nodes[self.root()].samples)

        normalizer = np.sum(importances)
        if normalizer > 0:
            # Avoid dividing by 0
            importances /= normalizer

        return importances

    def rank_features_by_importances(self):
        importances = self.feature_importances()
        """
        Numpy argsort uses ascending order, when you negate an array the lowest elements become
        the highest elements and vice-versa.
        """
        sort_arg = np.argsort(-importances)
        return zip(sort_arg, range(1, self.n_features + 1))

    def rank_features_by_importances_inverse(self):
        importances = self.feature_importances()
        sort_arg = np.argsort(importances)
        return zip(sort_arg, range(1, self.n_features + 1))


class DecisionNode(ABC):
    def __init__(self, samples, depth):
        self.samples = samples
        self.depth = depth
        super().__init__()


class DecisionFork(DecisionNode):
    def __init__(self, samples, depth, feature_id, gain, value):
        self.feature_id = feature_id
        self.gain = gain
        self.left_branch = None
        self.right_branch = None
        self.value = value
        super().__init__(samples, depth)

    @abstractmethod
    def result_branch(self, x):
        pass


class DecisionForkNumerical(DecisionFork):
    def __init__(self, samples, depth, feature_id, gain, value):
        super().__init__(samples, depth, feature_id, gain, value)

    def result_branch(self, x):
        if x[self.feature_id] <= self.value:
            return self.left_branch
        else:
            return self.right_branch


class DecisionForkCategorical(DecisionFork):
    def __init__(self, samples, depth, feature_id, gain, value):
        super().__init__(samples, depth, feature_id, gain, value)

    def result_branch(self, x):
        if x[self.feature_id] == self.value:
            return self.left_branch
        else:
            return self.right_branch


class DecisionLeaf(DecisionNode):
    def __init__(self, samples, depth, result):
        super().__init__(samples, depth)
        self.result = result
