import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import NotFittedError
from sklearn.metrics import accuracy_score

from proactive_forest.diversity import PercentageCorrectDiversity, QStatisticDiversity
from proactive_forest.tree_builder import TreeBuilder
from proactive_forest.probabilites import ModerateLedger, AggressiveLedger
from proactive_forest.voters import MajorityVoter, PerformanceWeightingVoter
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
                 min_gain_split=0):
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

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=None)

        self._n_instances, self._n_features = X.shape

        self._tree_builder = TreeBuilder(criterion=self.criterion,
                                         feature_prob=self.feature_prob,
                                         max_features=self.max_features,
                                         max_depth=self.max_depth,
                                         min_samples_leaf=self.min_samples_leaf,
                                         min_gain_split=self.min_gain_split,
                                         min_samples_split=self.min_samples_split)
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
        if check_input:
            X = self._validate_predict(X, check_input=check_input)

        sample_size, features_count = X.shape
        result = list(range(sample_size))
        for i in range(sample_size):
            x = X[i]
            result[i] = self._tree.predict_proba(x)
        return result

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
                 split='best',
                 criterion='gini',
                 min_samples_leaf=1,
                 max_features='rand',
                 feature_prob=None,
                 min_gain_split=0,
                 min_samples_split=2):

        self._trees = None
        self._n_features = None
        self._n_instances = None
        self._tree_builder = None
        self._n_classes = None

        # Ensemble parameters
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap

        # Tree parameters
        self.max_depth = max_depth
        self.split = split
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
            self.feature_prob = [1 / self._n_features for _ in range(self._n_features)]

        if self.bootstrap:
            set_generator = BaggingSet(self._n_instances)
        else:
            set_generator = SimpleSet(self._n_instances)

        self._tree_builder = TreeBuilder(criterion=self.criterion,
                                         feature_prob=self.feature_prob,
                                         max_features=self.max_features,
                                         max_depth=self.max_depth,
                                         min_samples_leaf=self.min_samples_leaf,
                                         min_gain_split=self.min_gain_split,
                                         min_samples_split=self.min_samples_split,
                                         split=self.split)

        for _ in range(self.n_estimators):
            ids = set_generator.training_ids()
            X_new = X[ids]
            y_new = y[ids]

            new_tree = self._tree_builder.build_tree(X_new, y_new)

            if self.bootstrap:
                validation_ids = set_generator.oob_ids()
                new_tree.weight = accuracy_score(y[validation_ids], self._predict_on_tree(X[validation_ids], new_tree))

            self._trees.append(new_tree)
            set_generator.clear()

        return self

    def predict(self, X, check_input=True):
        if check_input:
            X = self._validate_predict(X, check_input=check_input)

        # voter = MajorityVoter(self._trees, self._n_classes)
        voter = PerformanceWeightingVoter(self._trees, self._n_classes)

        sample_size, features_count = X.shape
        result = np.zeros(sample_size)
        for i in range(sample_size):
            x = X[i]
            result[i] = voter.predict(x)
        return result

    def _predict_on_tree(self, X, tree, check_input=True):
        if check_input:
            X = self._validate_predict(X, check_input=check_input)

        sample_size, features_count = X.shape
        result = np.zeros(sample_size)
        for i in range(sample_size):
            x = X[i]
            result[i] = tree.predict(x)
        return result

    def predict_proba(self, X, check_input=True):
        if check_input:
            X = self._validate_predict(X, check_input=check_input)

        voter = PerformanceWeightingVoter(self._trees, self._n_classes)

        sample_size, features_count = X.shape
        result = list(range(sample_size))
        for i in range(sample_size):
            x = X[i]
            result[i] = voter.predict_proba(x)
        return result

    def feature_importances(self):
        importances = np.zeros(self._n_features)
        for tree in self._trees:
            importances += tree.feature_importances()
        importances /= len(self._trees)
        return importances

    def trees_mean_weight(self):
        weights = [tree.weight for tree in self._trees]
        mean_weight = np.mean(weights)
        return mean_weight

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

    def diversity_measure(self, X_test, y_test, type='pcd'):
        X, y = check_X_y(X_test, y_test, dtype=None)

        if type == 'pcd':
            metric = PercentageCorrectDiversity()
        elif type == 'qstat':
            metric = QStatisticDiversity()

        forest_diversity = metric.get_measure(self._trees, X, y)

        return forest_diversity


class ProactiveForestClassifier(DecisionForestClassifier):
    def __init__(self,
                 n_estimators=100,
                 bootstrap=True,
                 max_depth=None,
                 split='best',
                 criterion='gini',
                 min_samples_leaf=1,
                 max_features='rand',
                 feature_prob=None,
                 min_gain_split=0,
                 min_samples_split=2,
                 alpha=10):
        self.alpha = alpha
        super().__init__(n_estimators=n_estimators,
                         bootstrap=bootstrap,
                         max_depth=max_depth,
                         split=split,
                         criterion=criterion,
                         min_samples_leaf=min_samples_leaf,
                         max_features=max_features,
                         feature_prob=feature_prob,
                         min_gain_split=min_gain_split,
                         min_samples_split=min_samples_split
                         )

    def fit(self, X, y=None):
        # Cleaning input, obtaining ndarrays
        X, y = check_X_y(X, y, dtype=None)

        self._n_instances, self._n_features = X.shape
        self._n_classes = len(np.unique(y))
        self._trees = []

        prob_ledger = AggressiveLedger(probabilities=self.feature_prob, n_features=self._n_features, alpha=self.alpha)

        if self.bootstrap:
            set_generator = BaggingSet(self._n_instances)
        else:
            set_generator = SimpleSet(self._n_instances)

        self._tree_builder = TreeBuilder(criterion=self.criterion,
                                         feature_prob=prob_ledger.probabilities,
                                         max_features=self.max_features,
                                         max_depth=self.max_depth,
                                         min_samples_leaf=self.min_samples_leaf,
                                         min_gain_split=self.min_gain_split,
                                         min_samples_split=self.min_samples_split,
                                         split=self.split)

        for _ in range(self.n_estimators):
            ids = set_generator.training_ids()
            X_new = X[ids]
            y_new = y[ids]

            new_tree = self._tree_builder.build_tree(X_new, y_new)

            if self.bootstrap:
                validation_ids = set_generator.oob_ids()
                new_tree.weight = accuracy_score(y[validation_ids], self._predict_on_tree(X[validation_ids], new_tree))

            prob_ledger.update_probabilities(new_tree.rank_features_by_importances())
            self._tree_builder.feature_prob = prob_ledger.probabilities

            self._trees.append(new_tree)
            set_generator.clear()

        return self


class ForestClassifier(DecisionForestClassifier):
    def fit(self, X, y=None):
        # Cleaning input, obtaining ndarrays
        X, y = check_X_y(X, y, dtype=None)

        self._n_instances, self._n_features = X.shape
        self._n_classes = len(np.unique(y))
        self._trees = []

        if self.feature_prob is None:
            self.feature_prob = [1 / self._n_features for _ in range(self._n_features)]

        if self.bootstrap:
            set_generator = BaggingSet(self._n_instances)
        else:
            set_generator = SimpleSet(self._n_instances)

        self._tree_builder = TreeBuilder(criterion=self.criterion,
                                         feature_prob=self.feature_prob,
                                         max_features=self.max_features,
                                         max_depth=self.max_depth,
                                         min_samples_leaf=self.min_samples_leaf,
                                         min_gain_split=self.min_gain_split,
                                         min_samples_split=self.min_samples_split,
                                         split=self.split)

        for i in range(1, self.n_estimators+1):

            # print(i, np.array(self._tree_builder.feature_prob))

            ids = set_generator.training_ids()
            X_new = X[ids]
            y_new = y[ids]

            new_tree = self._tree_builder.build_tree(X_new, y_new)

            if self.bootstrap:
                validation_ids = set_generator.oob_ids()
                new_tree.weight = accuracy_score(y[validation_ids], self._predict_on_tree(X[validation_ids], new_tree))

            self._trees.append(new_tree)
            set_generator.clear()

            # Evaluate code

            importances = self.feature_importances()
            old_prob = np.array(self._tree_builder.feature_prob)
            new_prob = old_prob * (1 - importances * i/self.n_estimators)
            normalizer = np.sum(new_prob)
            new_prob /= normalizer

            self._tree_builder.feature_prob = new_prob.tolist()

        return self
