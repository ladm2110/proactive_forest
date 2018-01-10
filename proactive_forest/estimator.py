import scipy.stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import NotFittedError
from proactive_forest.tree_builder import Builder
from proactive_forest.utils import Sampler
from proactive_forest.probabilites import AggressiveLedger

class DecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 max_depth=None,
                 splitter='best',
                 criterion='entropy',
                 min_samples_leaf=5,
                 min_samples_split=10,
                 feature_selection='all',
                 feature_prob=None,
                 min_gain_split=0.01,
                 n_jobs=1):
        """
        Builds a decision tree for a classification problem.
        Args:

        Returns: self
        """
        self._tree = None
        self._n_features = None
        self._n_instances = None
        self._tree_builder = None

        # Tree parameters
        self.max_depth = max_depth
        self.splitter = splitter
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_gain_split = min_gain_split
        self.feature_selection = feature_selection
        self.feature_prob = feature_prob
        self.n_jobs = n_jobs

    def fit(self, X, y):

        X, y = check_X_y(X, y, dtype=None)

        self._n_instances, self._n_features = X.shape

        self._tree_builder = Builder(criterion=self.criterion,
                                     feature_prob=self.feature_prob,
                                     feature_selection=self.feature_selection,
                                     max_depth=self.max_depth,
                                     min_samples_leaf=self.min_samples_leaf,
                                     min_gain_split=self.min_gain_split,
                                     min_samples_split=self.min_samples_split,
                                     n_jobs=self.n_jobs)
        self._tree = self._tree_builder.build_tree(X, y)

        return self

    def predict(self, X, check_input=True):
        if check_input:
            X = self._validate_predict(X, check_input=check_input)
        return self._tree.predict(X)

    def predict_proba(self, X, check_input=True):
        pass

    def _validate_predict(self, X, check_input):
        """Validate X whenever one tries to predict, apply, predict_proba"""
        if self._tree is None:
            raise NotFittedError("Estimator not fitted, "
                                 "call `fit` before exploiting the model.")

        if check_input:
            X = check_array(X, dtype=None)

        n_features = X.shape[1]
        if self._n_features != n_features:
            raise ValueError("Number of features of the model must "
                             " match the input. Model n_features is %s and "
                             " input n_features is %s "
                             % (self._n_features, n_features))

        return X


class DecisionForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 n_estimators=10,
                 bootstrap=True,
                 max_depth=None,
                 splitter='best',
                 criterion='entropy',
                 min_samples_leaf=5,
                 feature_selection='rand',
                 feature_prob=None,
                 min_gain_split=0.01,
                 min_samples_split=10,
                 n_jobs=1):

        self._trees = None
        self._samplers = None
        self._n_features = None
        self._n_instances = None
        self._tree_builder = None

        # Ensemble parameters
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs

        # Tree parameters
        self.max_depth = max_depth
        self.splitter = splitter
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_gain_split = min_gain_split
        self.feature_selection = feature_selection
        self.feature_prob = feature_prob

    def fit(self, X, y=None):
        X, y = check_X_y(X, y, dtype=None)

        self._n_instances, self._n_features = X.shape
        self._trees = []

        if self.feature_prob is None:
            self.feature_prob = [1/self._n_features for _ in range(self._n_features)]

        if self.bootstrap:
            self._samplers = []

        self._tree_builder = Builder(criterion=self.criterion,
                                     feature_prob=self.feature_prob,
                                     feature_selection=self.feature_selection,
                                     max_depth=self.max_depth,
                                     min_samples_leaf=self.min_samples_leaf,
                                     min_gain_split=self.min_gain_split,
                                     min_samples_split=self.min_samples_split,
                                     n_jobs=self.n_jobs)

        if self.bootstrap:
            for _ in range(self.n_estimators):
                sampler = Sampler(self._n_instances)
                ids = sampler.get_training_sample()
                X_new = X[ids]
                y_new = y[ids]
                self._samplers.append(sampler)
                self._trees.append(self._tree_builder.build_tree(X_new, y_new))
        else:
            for _ in range(self.n_estimators):
                self._trees.append(self._tree_builder.build_tree(X, y))

        return self

    def predict(self, X, check_input=True):
        if check_input:
            X = self._validate_predict(X, check_input=check_input)

        predictions = []
        for tree in self._trees:
            predictions.append(tree.predict(X))
        return scipy.stats.mode(predictions).mode[0]

    def _validate_predict(self, X, check_input):
        """Validate X whenever one tries to predict, apply, predict_proba"""
        if self._trees is None:
            raise NotFittedError("Estimator not fitted, "
                                 "call `fit` before exploiting the model.")

        if check_input:
            X = check_array(X, dtype=None)

        n_features = X.shape[1]
        if self._n_features != n_features:
            raise ValueError("Number of features of the model must "
                             " match the input. Model n_features is %s and "
                             " input n_features is %s "
                             % (self._n_features, n_features))

        return X


class ProactiveForestClassifier(DecisionForestClassifier):

    def fit(self, X, y=None):
        # Cleaning input, obtaining ndarrays
        X, y = check_X_y(X, y, dtype=None)

        self._n_instances, self._n_features = X.shape
        self._trees = []

        prob_ledger = AggressiveLedger(probabilities=self.feature_prob, n_features=self._n_features)

        if self.bootstrap:
            self._samplers = []

        self._tree_builder = Builder(criterion=self.criterion,
                                     feature_prob=prob_ledger.probabilities,
                                     feature_selection=self.feature_selection,
                                     max_depth=self.max_depth,
                                     min_samples_leaf=self.min_samples_leaf,
                                     min_gain_split=self.min_gain_split,
                                     min_samples_split=self.min_samples_split,
                                     n_jobs=self.n_jobs)

        if self.bootstrap:
            for _ in range(self.n_estimators):
                sampler = Sampler(self._n_instances)
                ids = sampler.get_training_sample()
                X_new = X[ids]
                y_new = y[ids]
                self._samplers.append(sampler)
                new_tree = self._tree_builder.build_tree(X_new, y_new)

                prob_ledger.update_probabilities(new_tree.features_and_levels())
                self._tree_builder.feature_prob = prob_ledger.probabilities

                self._trees.append(new_tree)

        else:
            for _ in range(self.n_estimators):
                new_tree = self._tree_builder.build_tree(X, y)

                prob_ledger.update_probabilities(new_tree.features_and_levels())
                self._tree_builder.feature_prob = prob_ledger.probabilities

                self._trees.append(new_tree)

        return self

