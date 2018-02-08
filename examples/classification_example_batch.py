from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from examples import load_batch
import pandas as pd
from proactive_forest.estimator import DecisionForestClassifier, ProactiveForestClassifier


if __name__ == '__main__':

    for name, loader in load_batch.get_batch_3():

        data_name = name
        X, y = loader[0], loader[1]

        rf_b = DecisionForestClassifier(n_estimators=100, criterion='gini', max_features='log', bootstrap=True)
        et_b = DecisionForestClassifier(n_estimators=100, criterion='gini', max_features='all', bootstrap=True,
                                        split='rand')
        pf_b = ProactiveForestClassifier(n_estimators=100, criterion='gini', max_features='prob', bootstrap=True,
                                         split='best')

        rf = cross_val_score(rf_b, X, y, cv=10)
        et = cross_val_score(et_b, X, y, cv=10)
        pf = cross_val_score(pf_b, X, y, cv=10)

        data = pd.DataFrame(index=['RF', 'ET', 'PF'])

        data['CV Acc. Mean'] = pd.Series([rf.mean(), et.mean(), pf.mean()], index=['RF', 'ET', 'PF'])
        data['CV Acc. Std'] = pd.Series([rf.std(), et.std(), pf.std()], index=['RF', 'ET', 'PF'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

        rf_b.fit(X_train, y_train)
        et_b.fit(X_train, y_train)
        pf_b.fit(X_train, y_train)

        data['PCD Diversity'] = pd.Series([rf_b.diversity_measure(X_test, y_test),
                                           et_b.diversity_measure(X_test, y_test),
                                           pf_b.diversity_measure(X_test, y_test)], index=['RF', 'ET', 'PF'])

        data['QStat Diversity'] = pd.Series([rf_b.diversity_measure(X_test, y_test, type='qstat'),
                                             et_b.diversity_measure(X_test, y_test, type='qstat'),
                                             pf_b.diversity_measure(X_test, y_test, type='qstat')],
                                            index=['RF', 'ET', 'PF'])

        data['Forest Acc.'] = pd.Series([accuracy_score(y_test, rf_b.predict(X_test)),
                                         accuracy_score(y_test, et_b.predict(X_test)),
                                         accuracy_score(y_test, pf_b.predict(X_test))], index=['RF', 'ET', 'PF'])

        rf_weight = rf_b.trees_mean_weight()
        et_weight = et_b.trees_mean_weight()
        pf_weight = pf_b.trees_mean_weight()

        data['Tree Weight Mean'] = pd.Series([rf_weight, et_weight, pf_weight], index=['RF', 'ET', 'PF'])

        data.to_csv("C:/Users/Luis Alberto Denis/Desktop/results/{}.csv".format(data_name), header=True, index=True)

        pf = ProactiveForestClassifier(n_estimators=100, criterion='gini', max_features='prob', bootstrap=True)

        params = {
            'alpha': list(range(4, 16))
        }

        grid = GridSearchCV(pf, params, scoring='accuracy', n_jobs=4, cv=10)
        grid.fit(X, y)

        data = pd.DataFrame()

        data['Mean Test Score'] = grid.cv_results_['mean_test_score']
        data['Rank Test Score'] = grid.cv_results_['rank_test_score']
        data['Param'] = [dic['alpha'] for dic in grid.cv_results_['params']]

        data.to_csv("C:/Users/Luis Alberto Denis/Desktop/results/grid/{}.csv".format(data_name), header=True, index=False)

