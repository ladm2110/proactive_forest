from abc import ABC, abstractmethod
import numpy as np


class DecisionTree:
    def __init__(self, n_features):
        self.n_features = n_features
        self.nodes = []
        self.last_node_id = None

    @staticmethod
    def root():
        return 0

    def predict(self, x):
        current_node = self.root()
        leaf_found = False
        while not leaf_found:
            if isinstance(self.nodes[current_node], DecisionLeaf):
                leaf_found = True
                prediction = self.nodes[current_node].result
            else:
                current_node = self.nodes[current_node].result_branch(x)
        return prediction

    def predict_proba(self, X):
        pass

    def features_and_levels(self):
        return [(n.feature_id, n.depth) for n in self.nodes if isinstance(n, DecisionFork)]

    def feature_importances(self):
        importances = np.zeros(self.n_features)
        for node in self.nodes:
            if isinstance(node, DecisionFork):
                importances[node.feature_id] += node.gain * node.n_samples / self.nodes[self.root()].n_samples

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
        return zip(sort_arg, range(1, self.n_features+1))


class DecisionNode(ABC):
    def __init__(self, n_samples, depth):
        self.n_samples = n_samples
        self.depth = depth
        super().__init__()


class DecisionFork(DecisionNode):
    def __init__(self,  n_samples, depth, feature_id, gain, value):
        self.feature_id = feature_id
        self.gain = gain
        self.left_branch = None
        self.right_branch = None
        self.value = value
        super().__init__(n_samples, depth)

    @abstractmethod
    def result_branch(self, x):
        pass


class DecisionForkNumerical(DecisionFork):
    def __init__(self, n_samples, depth, feature_id, gain, value):
        super().__init__(n_samples, depth, feature_id, gain, value)

    def result_branch(self, x):
        if x[self.feature_id] <= self.value:
            return self.left_branch
        else:
            return self.right_branch


class DecisionForkCategorical(DecisionFork):
    def __init__(self, n_samples, depth, feature_id, gain, value):
        super().__init__(n_samples, depth, feature_id, gain, value)

    def result_branch(self, x):
        if x[self.feature_id] == self.value:
            return self.left_branch
        else:
            return self.right_branch


class DecisionLeaf(DecisionNode):
    def __init__(self, n_samples, depth, result, prob):
        super().__init__(n_samples, depth)
        self.result = result
        self.prob = prob
