from sklearn.model_selection import train_test_split
from examples import load_data
from proactive_forest.estimator import DecisionForestClassifier, ProactiveForestClassifier


if __name__ == '__main__':

    X, y = load_data.load_kr_vs_kp()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4)

    pf = ProactiveForestClassifier(alpha=0.1, bootstrap=True)
    rf = DecisionForestClassifier(split_chooser='best', feature_selection='log', bootstrap=True)

    pf.fit(X_train, y_train)
    print('Processed: Proactive Forest')
    rf.fit(X_train, y_train)
    print('Processed: Random Forest')

    pf_diversity = pf.diversity_measure(X_test, y_test)
    rf_diversity = rf.diversity_measure(X_test, y_test)

    print('Proactive Forest Diversity: ', pf_diversity)
    print('Random Forest Diversity: ', rf_diversity)
