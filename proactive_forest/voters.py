import numpy as np
from abc import ABC, abstractmethod


class Voter(ABC):
    def __init__(self, predictors, n_classes):
        self.predictors = predictors
        self.votes = np.zeros(n_classes)

    @abstractmethod
    def predict(self, x):
        pass


class MajorityVoter(Voter):
    def predict(self, x):
        for model in self.predictors:
            result = model.predict(x)
            self.votes[result] += 1
        final_prediction = np.argmax(self.votes)
        return final_prediction


class WeightedVoter(Voter):
    def predict(self, x):
        pass
