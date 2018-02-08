from abc import ABC, abstractmethod

class DiversityMeasure(ABC):
    @abstractmethod
    def get_measure(self, predictors, X, y):
        pass


class PercentageCorrectDiversity(DiversityMeasure):
    def get_measure(self, predictors, X, y):
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
        n_instances = X.shape[0]
        n_predictors = len(predictors)
        q_total = 0
        for i in range(0, n_predictors-1):
            for j in range(i+1, n_predictors):
                n11 = 0
                n10 = 0
                n01 = 0
                n00 = 0
                for k in range(n_instances):
                    i_pred = predictors[i].predict(X[k])
                    j_pred = predictors[j].predict(X[k])
                    true_y = y[k]
                    if i_pred == true_y:
                        if j_pred == true_y:
                            n11 += 1
                        else:
                            n10 += 1
                    else:
                        if j_pred == true_y:
                            n01 += 1
                        else:
                            n00 += 1

                if n11 == 0:
                    n11 += 1
                if n01 == 0:
                    n01 += 1
                if n10 == 0:
                    n10 += 1
                if n00 == 0:
                    n00 += 1
                same = n11 * n00
                diff = n10 * n01
                q_ij = (same - diff) / (same + diff)
                q_total += q_ij

        q_av = 2 * q_total / (n_predictors * (n_predictors - 1))
        return q_av
