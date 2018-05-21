from abc import ABC, abstractmethod
import numpy as np


class ProbabilityLedger(ABC):
    def __init__(self, probabilities, n_features, alpha):
        """
        Creates a probability ledger.

        :param probabilities: <list> Feature probabilities
        :param n_features: <int> Amount of features
        :param alpha: <float> Diversity rate
        """
        if probabilities is None:

            if n_features is not None:
                initial_p = 1 / n_features
                self._probabilities = np.array([initial_p for _ in range(n_features)])
            else:
                raise Exception('Cannot initialize ledger without the number of features.')
        else:

            if len(probabilities) == n_features:
                self._probabilities = np.array(probabilities)
            else:
                raise Exception('Number of features must be equal to length of list of probabilities.')

        self._n_features = n_features
        self._alpha = alpha
        super().__init__()

    @abstractmethod
    def update_probabilities(self, new_tree, rate):
        pass

    def _normalize(self):
        """
        Normalizes the probabilities.
        """
        total = np.sum(self._probabilities)
        self._probabilities /= total

    @property
    def probabilities(self):
        return self._probabilities.tolist()

    @probabilities.setter
    def probabilities(self, probabilities):
        self._probabilities = probabilities

    @property
    def n_features(self):
        return self._n_features

    @n_features.setter
    def n_features(self, n_features):
        self._n_features = n_features

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha


class FIProbabilityLedger(ProbabilityLedger):
    def __init__(self, probabilities, n_features, alpha=0.1):
        """
        Creates a probabilities ledger which updates the probabilities according
        to the feature importances.

        param probabilities: <list> Feature probabilities
        :param n_features: <int> Amount of features
        :param alpha: <float> Diversity rate
        """
        self._feature_importances = np.zeros(n_features)
        self._n_trees = 0
        super().__init__(probabilities, n_features, alpha)

    def update_probabilities(self, new_tree, rate):
        """
        Updates the probabilities given a new tree.

        :param new_tree: <DecisionTree> New tree in the forest
        :param rate: <float> Rate of construction of the forest
        """
        self._feature_importances += new_tree.feature_importances()
        self._n_trees += 1
        self._probabilities = self._probabilities * (1 - (self._feature_importances / self._n_trees) *
                                                     self._alpha * rate)
        self._normalize()
