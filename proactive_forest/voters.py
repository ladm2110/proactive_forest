import numpy as np
from abc import ABC, abstractmethod


class WeightingVoter(ABC):
    def __init__(self, predictors, n_classes):
        """
        Creates a decision fusion model for the forest.

        :param predictors: <list> List containing all the trees
        :param n_classes: <int> Amount of classes
        """
        self._predictors = predictors
        self._n_classes = n_classes

    @property
    def predictors(self):
        return self._predictors

    @predictors.setter
    def predictors(self, predictors):
        self._predictors = predictors

    @property
    def n_classes(self):
        return self._n_classes

    @n_classes.setter
    def n_classes(self, n_classes):
        self._n_classes = n_classes

    @abstractmethod
    def predict(self, x):
        pass

    def predict_proba(self, x):
        """
        Predicts for a given array x the class probability estimates.

        :param x: <numpy array> Feature vector
        :return: <list>
        """
        results = np.zeros(self._n_classes)
        for model in self._predictors:
            pred_proba = model.predict_proba(x)
            results += pred_proba
        final_pred_proba = results / len(self._predictors)
        return final_pred_proba.tolist()


class MajorityVoter(WeightingVoter):
    def predict(self, x):
        """
        Predicts for a given array x the class to which it belongs
        using majority voting.

        :param x: <numpy array> Feature vector
        :return: <int>
        """
        results = np.zeros(self._n_classes)
        for model in self._predictors:
            prediction = model.predict(x)
            results[prediction] += 1
        final_prediction = np.argmax(results)
        return final_prediction


class PerformanceWeightingVoter(WeightingVoter):
    def predict(self, x):
        """
        Predicts for a given array x the class to which it belongs
        using performance weighting voting.

        :param x: <numpy array> Feature vector
        :return: <int>
        """
        # Normalizing the predictors weights
        weights = [model.weight for model in self._predictors]
        weights /= np.sum(weights)

        results = np.zeros(self._n_classes)
        for model, w in zip(self._predictors, weights):
            prediction = model.predict(x)
            results[prediction] += w
        final_prediction = np.argmax(results)
        return final_prediction


class DistributionSummationVoter(WeightingVoter):
    def predict(self, x):
        """
        Predicts for a given array x the class to which it belongs
        using distribution summation voting.

        :param x: <numpy array> Feature vector
        :return: <int>
        """
        results = np.zeros(self._n_classes)
        for model in self._predictors:
            pred_proba = model.predict_proba(x)
            results += pred_proba
        final_prediction = np.argmax(results)
        return final_prediction
