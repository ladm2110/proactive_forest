import scipy.stats
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import NotFittedError
from proactive_forest.tree_builder import Builder
from proactive_forest.probabilites import ModerateLedger
from proactive_forest.voters import MajorityVoter
from proactive_forest.sets import SimpleSet, BaggingSet


class DecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 max_depth=None,
                 splitter='best',
                 criterion='gini',
                 min_samples_leaf=1,
                 min_samples_split=2,
                 max_features='all',
                 feature_prob=None,
                 min_gain_split=0,
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
        self.max_features = max_features
        self.feature_prob = feature_prob
        self.n_jobs = n_jobs

    def fit(self, X, y):

        X, y = check_X_y(X, y, dtype=None)

        self._n_instances, self._n_features = X.shape

        self._tree_builder = Builder(criterion=self.criterion,
                                     feature_prob=self.feature_prob,
                                     max_features=self.max_features,
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

        sample_size, features_count = X.shape
        result = np.zeros(sample_size)
        for i in range(sample_size):
            x = X[i]
            result[i] = self._tree.predict(x)
        return result

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
                 n_estimators=100,
                 bootstrap=True,
                 max_depth=None,
                 splitter='best',
                 criterion='gini',
                 min_samples_leaf=1,
                 max_features='rand',
                 feature_prob=None,
                 min_gain_split=0,
                 min_samples_split=2,
                 n_jobs=1):

        self._trees = None
        self._samplers = None
        self._n_features = None
        self._n_instances = None
        self._tree_builder = None
        self._n_classes = None

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
        self.max_features = max_features
        self.feature_prob = feature_prob

    def fit(self, X, y=None):
        X, y = check_X_y(X, y, dtype=None)

        self._n_instances, self._n_features = X.shape
        self._n_classes = len(np.unique(y))
        self._trees = []

        if self.feature_prob is None:
            self.feature_prob = [1/self._n_features for _ in range(self._n_features)]

        if self.bootstrap:
            set_generator = BaggingSet(self._n_instances)
        else:
            set_generator = SimpleSet(self._n_instances)

        self._tree_builder = Builder(criterion=self.criterion,
                                     feature_prob=self.feature_prob,
                                     max_features=self.max_features,
                                     max_depth=self.max_depth,
                                     min_samples_leaf=self.min_samples_leaf,
                                     min_gain_split=self.min_gain_split,
                                     min_samples_split=self.min_samples_split,
                                     n_jobs=self.n_jobs)

        for _ in range(self.n_estimators):

            ids = set_generator.training_ids()
            X_new = X[ids]
            y_new = y[ids]

            new_tree = self._tree_builder.build_tree(X_new, y_new)

            self._trees.append(new_tree)
            set_generator.clear()

        return self

    def predict(self, X, check_input=True):
        if check_input:
            X = self._validate_predict(X, check_input=check_input)

        voter = MajorityVoter(self._trees, self._n_classes)

        sample_size, features_count = X.shape
        result = np.zeros(sample_size)
        for i in range(sample_size):
            x = X[i]
            result[i] = voter.predict(x)
        return result

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
        self._n_classes = len(np.unique(y))
        self._trees = []

        prob_ledger = ModerateLedger(probabilities=self.feature_prob, n_features=self._n_features)

        if self.bootstrap:
            set_generator = BaggingSet(self._n_instances)
        else:
            set_generator = SimpleSet(self._n_instances)

        self._tree_builder = Builder(criterion=self.criterion,
                                     feature_prob=prob_ledger.probabilities,
                                     max_features=self.max_features,
                                     max_depth=self.max_depth,
                                     min_samples_leaf=self.min_samples_leaf,
                                     min_gain_split=self.min_gain_split,
                                     min_samples_split=self.min_samples_split,
                                     n_jobs=self.n_jobs)

        for _ in range(self.n_estimators):

            ids = set_generator.training_ids()
            X_new = X[ids]
            y_new = y[ids]

            new_tree = self._tree_builder.build_tree(X_new, y_new)

            prob_ledger.update_probabilities(new_tree.rank_features_by_importances())
            self._tree_builder.feature_prob = prob_ledger.probabilities

            self._trees.append(new_tree)
            set_generator.clear()

        return self

