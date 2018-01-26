import pandas as pd
import time

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from proactive_forest.estimator import DecisionForestClassifier, ProactiveForestClassifier

if __name__ == '__main__':
    dataset = pd.read_csv('../data/tae.csv')

    dataset = dataset.sample(frac=1).reset_index(drop=True)

    y = dataset['class']
    X = dataset.drop('class', axis=1)

    encoder = LabelEncoder()
    y = pd.Series(encoder.fit_transform(y))

    rf_wb = DecisionForestClassifier(n_estimators=100, criterion='gini', feature_selection='log', bootstrap=False)
    rf_b = DecisionForestClassifier(n_estimators=100, criterion='gini', feature_selection='log', bootstrap=True)
    pf_rf_wb = ProactiveForestClassifier(n_estimators=100, criterion='gini', feature_selection='log_prob',
                                         bootstrap=False)
    pf_rf_b = ProactiveForestClassifier(n_estimators=100, criterion='gini', feature_selection='log_prob',
                                        bootstrap=True)
    pf_wb = ProactiveForestClassifier(n_estimators=100, criterion='gini', feature_selection='prob', bootstrap=False)
    pf_b = ProactiveForestClassifier(n_estimators=100, criterion='gini', feature_selection='prob', bootstrap=True)

    print('Random Forest No Bagging:')
    t1 = time.time()
    print(cross_val_score(rf_wb, X, y, scoring='accuracy', cv=10).mean())
    print(time.time() - t1)

    print('Random Forest + Bagging:')
    t2 = time.time()
    print(cross_val_score(rf_b, X, y, scoring='accuracy', cv=10).mean())
    print(time.time() - t2)

    print('Proactive Forest (RF) No Bagging:')
    t3 = time.time()
    print(cross_val_score(pf_rf_wb, X, y, scoring='accuracy', cv=10).mean())
    print(time.time() - t3)

    print('Proactive Forest (RF) + Bagging:')
    t4 = time.time()
    print(cross_val_score(pf_rf_b, X, y, scoring='accuracy', cv=10).mean())
    print(time.time() - t4)

    print('Proactive Forest No Bagging:')
    t5 = time.time()
    print(cross_val_score(pf_wb, X, y, scoring='accuracy', cv=10).mean())
    print(time.time() - t5)

    print('Proactive Forest + Bagging:')
    t6 = time.time()
    print(cross_val_score(pf_b, X, y, scoring='accuracy', cv=10).mean())
    print(time.time() - t6)