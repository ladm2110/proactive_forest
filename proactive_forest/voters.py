import scipy.stats
import numpy as np
from abc import ABC, abstractmethod


class Voter(ABC):
    def __init__(self, predictors, n_classes):
        self.predictors = predictors
        self.n_classes = n_classes

    @abstractmethod
    def predict(self, x):
        pass


class MajorityVoter(Voter):
    def predict(self, x):
        results = np.zeros(self.n_classes)
        for model in self.predictors:
            prediction = model.predict(x)
            results[prediction] += 1
        final_prediction = np.argmax(results)
        return final_prediction


class WeightedVoter(Voter):
    def predict(self, X):
        pass
