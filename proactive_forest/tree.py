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
            if isinstance(self.nodes[current_node], DecisionLeaf):
                leaf_found = True
                class_proba = [n + 1 for n in self.nodes[current_node].samples] / \
                              (np.sum(self.nodes[current_node].samples) + len(self.nodes[current_node].samples))
            else:
                current_node = self.nodes[current_node].result_branch(x)
        return class_proba.tolist()

    def feature_importances(self):
        """
        Calculates the feature importances according to Breiman 2001.

        :return: <numpy array>
        """
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

    def features(self):
        return np.unique([node.feature_id for node in self.nodes if isinstance(node, DecisionFork)])

    def total_nodes(self):
        """
        Returns the amount of nodes in the decision tree.

        :return: <int>
        """
        return len(self.nodes)

    def total_splits(self):
        """
        Returns the amount of splits done in the decision tree.

        :return: <int>
        """
        count = 0
        for node in self.nodes:
            if isinstance(node, DecisionFork):
                count += 1
        return count

    def total_leaves(self):
        """
        Returns the amount of leaves in the decision tree.

        :return: <int>
        """
        count = 0
        for node in self.nodes:
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
        self.samples = samples
        self.depth = depth
        super().__init__()


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
        self.result = result
