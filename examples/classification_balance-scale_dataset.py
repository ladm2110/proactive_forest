from proactive_forest.estimator import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import time
import pandas as pd

if __name__ == '__main__':

    dataset = pd.read_csv('../data/balance-scale.csv', header=None)

    y = dataset[4]
    X = dataset.drop(4, axis=1)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    model = DecisionTreeClassifier()

    t0 = time.time()

    print(cross_val_score(model, X, y, scoring='accuracy', cv=10, n_jobs=-1).mean())

    print(time.time() - t0)
