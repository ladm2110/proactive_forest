from abc import ABC, abstractmethod
import numpy as np


class DecisionTree:

    MISSING = '?'

    def __init__(self, n_features):
        self.n_features = n_features
        self.nodes = []
        self.last_node_id = None

    @staticmethod
    def root():
        return 0

    def predict(self, X):

        def predict_one(x):
            current_node = self.root()
            leaf_found = False
            while not leaf_found:
                if isinstance(self.nodes[current_node], DecisionLeaf):
                    leaf_found = True
                    prediction = self.nodes[current_node].result
                else:
                    current_node = self.nodes[current_node].result_branch(x)
            return prediction

        sample_size, features_count = X.shape
        assert features_count == self.n_features
        result = np.zeros(sample_size)
        for i in range(sample_size):
            x = X[i]
            result[i] = predict_one(x)
        return result

    def features_and_levels(self):
        return [(n.feature_id, n.depth) for n in self.nodes if isinstance(n, DecisionFork)]

    def features_and_importances(self):

        def prob_reaching_node(node):
            return self.nodes[node].n_samples / self.nodes[self.root()].n_samples

        def weighted_gain(node):
            # TODO: decisionleafs do not have a gain property
            return self.nodes[node].gain * prob_reaching_node(node_id)

        importances = np.zeros(self.n_features)
        for node_id in range(self.last_node_id):
            if isinstance(self.nodes[node_id], DecisionFork):
                importances[self.nodes[node_id].feature_id] += weighted_gain(node_id) \
                                                               - weighted_gain(self.nodes[node_id].left_branch) \
                                                               - weighted_gain(self.nodes[node_id].right_branch)

        normalizer = np.sum(importances)
        if normalizer > 0:
            # Avoid dividing by 0
            importances /= normalizer

        return zip(range(self.n_features), importances)


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
