from sklearn.model_selection import cross_val_score
from examples import load_data
from proactive_forest.estimator import DecisionForestClassifier, ProactiveForestClassifier


if __name__ == '__main__':

    X, y = load_data.load_car()

    rf_b = DecisionForestClassifier(n_estimators=100, criterion='gini', max_features='log', bootstrap=True)
    pf_b = ProactiveForestClassifier(n_estimators=100, criterion='gini', max_features='log', bootstrap=True, alpha=0.1)

    rf = cross_val_score(rf_b, X, y, cv=10)
    pf = cross_val_score(pf_b, X, y, cv=10)

    print(rf.mean())
    print(pf.mean())
