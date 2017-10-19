from proactive_forest.estimator import DecisionTreeClassifier, DecisionForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import time
import pandas as pd
import numpy as np

if __name__ == '__main__':

    dataset = pd.read_csv('../data/iris.csv')
    dataset = dataset.drop('Id', axis=1)

    y = dataset['Species']
    X = dataset.drop('Species', axis=1)

    encoder = LabelEncoder()
    y = pd.Series(encoder.fit_transform(y))

    model = DecisionForestClassifier(n_estimators=100, criterion='gini', feature_selection='prob', feature_prob=[0.4, 0.4, 0.1, 0.1])

    t0 = time.time()

    # model.fit(X, y)
    # p = model.predict(np.array([1, 1, 1, 1]).reshape(1, -1))
    # print(encoder.inverse_transform(p.astype(int)))

    print(cross_val_score(model, X, y, scoring='accuracy', cv=10).mean())

    print(time.time() - t0)
