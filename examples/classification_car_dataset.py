from proactive_forest.estimator import DecisionTreeClassifier, DecisionForestClassifier, ProactiveForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import time
import pandas as pd

if __name__ == '__main__':

    dataset = pd.read_csv('../data/car.csv')

    dataset = dataset.sample(frac=1).reset_index(drop=True)

    y = dataset['class']
    X = dataset.drop('class', axis=1)

    encoder = LabelEncoder()
    y = pd.Series(encoder.fit_transform(y))

    n_instances, n_features = X.shape

    model = DecisionForestClassifier(n_estimators=100, criterion='gini', feature_selection='all', categorical=[0,1,2,3,4,5])
    model_r = DecisionForestClassifier(n_estimators=100, criterion='gini', feature_selection='rand', categorical=[0,1,2,3,4,5])
    model_p = ProactiveForestClassifier(n_estimators=100, criterion='gini', feature_selection='prob', categorical=[0,1,2,3,4,5],
                                        feature_prob=[1/n_features for n in range(n_features)])

    print('Decision Forest:')
    t = time.time()
    print(cross_val_score(model, X, y, scoring='accuracy', cv=5).mean())
    print(time.time() - t)

    print('Decision Forest (Random):')
    tr = time.time()
    print(cross_val_score(model_r, X, y, scoring='accuracy', cv=5).mean())
    print(time.time() - tr)

    print('Proactive Forest:')
    tp = time.time()
    print(cross_val_score(model_p, X, y, scoring='accuracy', cv=5).mean())
    print(time.time() - tp)
