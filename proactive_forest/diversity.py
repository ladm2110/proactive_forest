from abc import ABC, abstractmethod
import numpy as np


class DiversityMeasure(ABC):
    @abstractmethod
    def get_measure(self, predictors, X, y):
        pass


class PercentageCorrectDiversity(DiversityMeasure):
    def get_measure(self, predictors, X, y):
        """
        Measures the Percentage of Correct Diversity.

        :param predictors: <list> Decision Trees
        :param X: <numpy ndaray> Feature vectors
        :param y: <numpy array> Target feature
        :return: <float>
        """
        tally = 0
        n_instances = X.shape[0]
        for i in range(n_instances):
            instance, target = X[i], y[i]
            n_corrects = 0
            for p in predictors:
                prediction = p.predict(instance)
                if prediction == target:
                    n_corrects += 1
            if 0.1 * len(predictors) <= n_corrects <= 0.9 * len(predictors):
                tally += 1
        diversity = tally / n_instances
        return diversity


class QStatisticDiversity(DiversityMeasure):
    def get_measure(self, predictors, X, y):
        """
        Measures the QStatistic Diversity.

        :param predictors: <list> Decision Trees
        :param X: <numpy ndaray> Feature vectors
        :param y: <numpy array> Target feature
        :return: <float>
        """
        n_instances = X.shape[0]
        n_predictors = len(predictors)
        q_total = 0
        for i in range(0, n_predictors-1):
            for j in range(i+1, n_predictors):
                n = np.zeros((2, 2))
                for k in range(n_instances):
                    i_pred = predictors[i].predict(X[k])
                    j_pred = predictors[j].predict(X[k])
                    true_y = y[k]
                    if i_pred == true_y:
                        if j_pred == true_y:
                            n[1][1] += 1
                        else:
                            n[1][0] += 1
                    else:
                        if j_pred == true_y:
                            n[0][1] += 1
                        else:
                            n[0][0] += 1

                # Adding a one value to the variables which are zeros
                for k in range(2):
                    for l in range(2):
                        if n[k][l] == 0:
                            n[k][l] += 1
                same = n[1][1] * n[0][0]
                diff = n[1][0] * n[0][1]
                q_ij = (same - diff) / (same + diff)
                q_total += q_ij

        q_av = 2 * q_total / (n_predictors * (n_predictors - 1))
        return q_av

