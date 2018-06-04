from sklearn.model_selection import cross_val_score
from examples import load_data
from proactive_forest.estimator import DecisionForestClassifier, ProactiveForestClassifier
import pandas as pd


if __name__ == '__main__':

    X, y = load_data.load_liver_disorder()

    pf = ProactiveForestClassifier(alpha=0.1, bootstrap=False)
    rf = DecisionForestClassifier(feature_selection='log', split_chooser='best', bootstrap=False)

    pf_scores = cross_val_score(pf, X, y, cv=3)
    print('Processed: Proactive Forest')
    rf_scores = cross_val_score(rf, X, y, cv=3)
    print('Processed: Random Forest')

    cross_val = pd.DataFrame()
    cross_val['PF'] = pf_scores
    cross_val['RF'] = rf_scores
    print(cross_val)

    print('Proactive Forest Score:', pf_scores.mean())
    print('Random Forest Score:', rf_scores.mean())
