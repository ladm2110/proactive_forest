from proactive_forest.estimator import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from pandas import read_csv
from sklearn.model_selection import cross_val_score
import time

if __name__ == '__main__':

    dataset = read_csv('../data/arrhythmia.csv', header=None, na_values='?')
    dataset.dropna(inplace=True)

    print(dataset.info())

    y = dataset[279]
    X = dataset.drop(279, axis=1)

    model = DecisionTreeClassifier()

    t0 = time.time()

    print(cross_val_score(model, X, y, scoring='accuracy', cv=5).mean())

    print(time.time() - t0)
