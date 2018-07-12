from sklearn.model_selection import cross_val_score, KFold
from examples import load_data
from proactive_forest.estimator import DecisionForestClassifier, ProactiveForestClassifier
import pandas as pd
import numpy as np

if __name__ == '__main__':

    X, y = load_data.load_iris()

    pf = ProactiveForestClassifier(alpha=0.1, bootstrap=False)
    rf = DecisionForestClassifier(feature_selection='log', split_chooser='best', bootstrap=False)
    """
    pf_scores = cross_val_score(pf, X, y, cv=5)
    print('Processed: Proactive Forest')
    rf_scores = cross_val_score(rf, X, y, cv=5)
    print('Processed: Random Forest')
    """
    pf_scores = []
    rf_scores = []
    
    skf = KFold(n_splits=5, random_state=4)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        pf.fit(X_train, y_train)
        rf.fit(X_train, y_train)
        pf_scores.append(pf.score(X_test, y_test))
        rf_scores.append(rf.score(X_test, y_test))

    cross_val = pd.DataFrame()
    cross_val['PF'] = pf_scores
    cross_val['RF'] = rf_scores
    print(cross_val)

    print('Proactive Forest Score:', np.mean(pf_scores))
    print('Random Forest Score:', np.mean(rf_scores))
