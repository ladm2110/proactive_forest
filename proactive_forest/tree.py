import numpy as np


class DecisionTree:

    MISSING = '?'

    def __init__(self, n_features):
        self.n_features = n_features
        self.nodes = []
        self.last_node_id = None

    def root(self):
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

    def predict_one(self, x):
        predictions = self.predict_recur(x)
        best_result, best_n = None, 0
        for result, value in predictions:
            if value > best_n:
                best_result = result
        return best_result

    def predict_recur(self, x, current_node=0):
        p = []
        if isinstance(self.nodes[current_node], DecisionLeaf):
            return [(self.nodes[current_node].result, self.nodes[current_node].prob * self.nodes[current_node].n_samples)]
        for node in self.nodes[current_node].result_branch(x):
            p.extend(self.predict_recur(x, current_node=node))
        return p

    def features_and_levels(self):
        return [(n.feature_id, n.depth) for n in self.nodes if isinstance(n, DecisionFork)]

    def features_and_importances(self):

        def prob_reaching_node(node):
            return self.nodes[node].n_samples / self.nodes[self.root()].n_samples

        def weighted_gain(node):
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


class DecisionNode:
    def __init__(self, n_samples, depth):
        self.n_samples = n_samples
        self.depth = depth


class DecisionFork(DecisionNode):
    def __init__(self,  n_samples, depth, feature_id, gain):
        super(DecisionFork, self).__init__(n_samples, depth)
        self.feature_id = feature_id
        self.gain = gain

    def add(self):
        pass

    def result_branch(self):
        pass


class DecisionForkNumerical(DecisionFork):
    def __init__(self, n_samples, depth, feature_id, gain, value):
        super(DecisionForkNumerical, self).__init__(n_samples, depth, feature_id, gain)
        self.value = value
        self.left_branch = None
        self.right_branch = None

    def result_branch(self, x):
        if x[self.feature_id] != DecisionTree.MISSING:
            if x[self.feature_id] <= self.value:
                return self.left_branch
            else:
                return self.right_branch
        else:
            return [self.left_branch, self.right_branch]


class DecisionForkCategorical(DecisionFork):
    def __init__(self, n_samples, depth, feature_id, gain, value):
        super(DecisionForkCategorical, self).__init__(n_samples, depth, feature_id, gain)
        self.value = value
        self.left_branch = None
        self.right_branch = None

    def result_branch(self, x):
        if x[self.feature_id] != DecisionTree.MISSING:
            if x[self.feature_id] == self.value:
                return self.left_branch
            else:
                return self.right_branch
        else:
            return [self.left_branch, self.right_branch]


class DecisionLeaf(DecisionNode):
    def __init__(self, n_samples, prob, depth, result):
        super(DecisionLeaf, self).__init__(n_samples, depth)
        self.result = result
        self.prob = prob
