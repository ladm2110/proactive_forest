from sklearn.model_selection import train_test_split
from examples import load_data
from proactive_forest.estimator import DecisionForestClassifier, ProactiveForestClassifier
import pandas as pd


if __name__ == '__main__':

    X, y = load_data.load_credit()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4)

    pf = ProactiveForestClassifier(alpha=0.1, bootstrap=True)

    pf.fit(X_train, y_train)
    print('Processed: Proactive Forest')

    pf_predictions = pf.predict(X_test)
    print('Predictions made.')

    pf_proba = [max(p) for p in pf.predict_proba(X_test)]
    print('Probabilities obtained.')

    predictions = pd.DataFrame()
    predictions['Correct'] = y_test
    predictions['PF'] = pf_predictions
    predictions['Prob'] = pf_proba
    print(predictions)
