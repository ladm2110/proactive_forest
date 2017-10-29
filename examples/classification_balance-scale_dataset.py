from proactive_forest.estimator import DecisionTreeClassifier, DecisionForestClassifier, ProactiveForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import time
import pandas as pd

if __name__ == '__main__':

    dataset = pd.read_csv('../data/balance-scale.csv', header=None)

    dataset = dataset.sample(frac=1).reset_index(drop=True)

    y = dataset[4]
    X = dataset.drop(4, axis=1)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    model = DecisionForestClassifier(n_estimators=100, criterion='gini', feature_selection='all',
                                     feature_prob=[0.25, 0.25, 0.25, 0.25])
    model_r = DecisionForestClassifier(n_estimators=100, criterion='gini', feature_selection='rand',
                                       feature_prob=[0.25, 0.25, 0.25, 0.25])
    model_p = ProactiveForestClassifier(n_estimators=100, criterion='gini', feature_selection='prob',
                                        feature_prob=[0.25, 0.25, 0.25, 0.25])

    print('Decision Forest:')
    t = time.time()
    print(cross_val_score(model, X, y, scoring='accuracy', cv=10).mean())
    print(time.time() - t)

    print('Decision Forest (Random):')
    tr = time.time()
    print(cross_val_score(model_r, X, y, scoring='accuracy', cv=10).mean())
    print(time.time() - tr)

    print('Proactive Forest (Random):')
    tp = time.time()
    print(cross_val_score(model_p, X, y, scoring='accuracy', cv=10).mean())
    print(time.time() - tp)
