from abc import ABC, abstractmethod
import numpy as np


class ProbabilityLedger(ABC):
    def __init__(self, probabilities, n_features, alpha=5):
        if probabilities is None:

            if n_features is not None:
                initial_p = 1 / n_features
                self.probabilities = [initial_p for _ in range(n_features)]
            else:
                raise Exception('Cannot initialize ledger without the number of features.')
        else:

            if len(probabilities) == n_features:
                self.probabilities = probabilities
            else:
                raise Exception('Number of features must be equal to length of list of probabilities.')

        self.n_features = n_features
        self.alpha = alpha
        super().__init__()

    @abstractmethod
    def update_probabilities(self, features_and_levels):
        pass

    def _normalize(self):
        total = np.sum(self.probabilities)
        for i in range(self.n_features):
            self.probabilities[i] /= total

    def _fill_remainder(self):
        remainder = self._calculate_remainder()
        split_remainder = remainder / self.n_features

        for i in range(self.n_features):
            self.probabilities[i] += split_remainder

    def _calculate_remainder(self):
        remainder = 1
        for p in self.probabilities:
            remainder -= p

        return remainder


class ModerateLedger(ProbabilityLedger):
    def __init__(self, probabilities, n_features, alpha=5):
        super().__init__(probabilities, n_features, alpha)

    def update_probabilities(self, features_and_levels):
        for feature, level in features_and_levels:
            old_prob = self.probabilities[feature]
            score = self.alpha * level
            new_prob = old_prob * (1 - 1 / score)
            self.probabilities[feature] = new_prob

        self._normalize()


class AggressiveLedger(ProbabilityLedger):
    def __init__(self, probabilities, n_features, alpha=5):
        super().__init__(probabilities, n_features, alpha)

    def update_probabilities(self, features_and_levels):
        for feature, level in features_and_levels:
            old_prob = self.probabilities[feature]
            score = self.alpha + level
            new_prob = old_prob * (1 - 1 / score)
            self.probabilities[feature] = new_prob

        self._normalize()
