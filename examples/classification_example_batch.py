from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from examples import load_batch
from proactive_forest.estimator import DecisionForestClassifier, ProactiveForestClassifier

if __name__ == '__main__':

    data = pd.DataFrame()

    for name, loader in load_batch.get_all():

        data_name = name
        X, y = loader[0], loader[1]

        fc = ProactiveForestClassifier(alpha=0.1)
        #fc = DecisionForestClassifier()

        cross_val = cross_val_score(fc, X, y, cv=10)

        score = cross_val.mean()
        std = cross_val.std()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

        fc.fit(X_train, y_train)

        pcd = fc.diversity_measure(X_test, y_test)

        qstat = fc.diversity_measure(X_test, y_test, diversity='qstat')

        acc = accuracy_score(y_test, fc.predict(X_test))

        fc_weight = fc.trees_mean_weight()

        data[data_name] = pd.Series([score, std, pcd, qstat, acc, fc_weight],
                                    index=['CV Score Mean', 'CV Score Std', 'PCD Div.', 'QStat Div.', 'For. Acc.',
                                           'Trees Mean Weight'])
        print('Done:', name)

    data.T.to_csv("C:/Users/Luis Alberto Denis/Desktop/results/pf_bag_alpha05_rate1.csv", header=True, index=True)
