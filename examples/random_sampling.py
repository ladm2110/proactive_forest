
from random import choices
import pandas as pd
from proactive_forest.estimator import DecisionTreeClassifier

if __name__ == '__main__':

    population = range(4)
    weights = [0.4, 0.4, 0.18, 0.02]
    selected = choices(population, weights, k=4)
    uniques = pd.unique(selected)
    print(selected)
    print(uniques)
