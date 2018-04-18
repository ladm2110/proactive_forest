from abc import ABC, abstractmethod
import numpy as np


class ProbabilityLedger(ABC):
    def __init__(self, probabilities, n_features, alpha=0.1):
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
        total = np.sum(self._probabilities)
        self._probabilities /= total

    @property
    def probabilities(self):
        return self._probabilities.tolist()

    @property
    def n_features(self):
        return self._n_features


class FIProbabilityLedger(ProbabilityLedger):
    def __init__(self, probabilities, n_features, alpha=0.1):
        self._feature_importances = np.zeros(n_features)
        self._n_trees = 0
        super().__init__(probabilities, n_features, alpha)

    def update_probabilities(self, new_tree, rate):
        self._feature_importances += new_tree.feature_importances()
        self._n_trees += 1
        self._probabilities = self._probabilities * (1 - (self._feature_importances / self._n_trees) *
                                                     self._alpha * rate)
        self._normalize()
