import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_X_y, check_array
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
import proactive_forest.utils as utils
from proactive_forest.diversity import PercentageCorrectDiversity, QStatisticDiversity
from proactive_forest.tree_builder import TreeBuilder
from proactive_forest.voters import PerformanceWeightingVoter
from proactive_forest.sets import SimpleSet, BaggingSet
from proactive_forest.probabilites import FIProbabilityLedger
from proactive_forest.splits import resolve_split_selection
from proactive_forest.metrics import resolve_split_criterion
from proactive_forest.feature_selection import resolve_feature_selection


class DecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 max_depth=None,
                 split_chooser='best',
                 split_criterion='gini',
                 min_samples_leaf=1,
                 min_samples_split=2,
                 feature_selection='all',
                 feature_prob=None,
                 min_gain_split=0):
        """
        Builds a decision tree for a classification problem.

        :param max_depth: <int> or <None> Defines the maximum depth of the tree
        :param split_chooser: <string> The name of the split chooser:
                            "best" for selecting the best possible split
                            "rand" for selecting a random split
        :param split_criterion: <string> The name of the split criterion:
                            "gini" for selecting the Gini criterion
                            "entropy" for selecting the Entropy criterion
        :param min_samples_leaf: <int> Minimum number of instances to place in a leaf
        :param min_samples_split: <int> Minimum number of instances to consider creating a split
        :param feature_selection: <string> The name of the feature selection criteria:
                            "all" for selecting all features as candidate features
                            "log" for selecting log(n)+1 as candidate features
                            "prob" for selecting the candidate features according to its probabilities
        :param feature_prob: <list> Feature probabilities
        :param min_gain_split: <float> Minimum split gain value to consider splitting a node
        """
        # Classifier parameters
        self._tree = None
        self._n_features = None
        self._n_instances = None
        self._tree_builder = None
        self._encoder = None
        self._n_classes = None

        # Tree parameters
        self._max_depth = None
        self._min_samples_leaf = None
        self._min_samples_split = None
        self._feature_prob = None
        self._min_gain_split = None
        self._split_chooser = None
        self._split_criterion = None
        self._feature_selection = None

        if max_depth is None or max_depth > 0:
            self._max_depth = max_depth
        else:
            raise ValueError('The depth of the tree must be greater than 0.')

        if min_samples_leaf is not None and min_samples_leaf > 0:
            self._min_samples_leaf = min_samples_leaf
        else:
            raise ValueError('The minimum number of instances to place in a leaf must be greater than 0.')

        if min_samples_split is not None and min_samples_split > 1:
            self._min_samples_split = min_samples_split
        else:
            raise ValueError('The minimum number of instances to make a split must be greater than 1')

        if feature_prob is None or (utils.check_array_sum_one(feature_prob) and
                                    utils.check_positive_array(feature_prob)):
            self._feature_prob = feature_prob
        else:
            raise ValueError('The features probabilities must be positive values and the sum must be one')

        if min_gain_split is not None and min_gain_split >= 0:
            self._min_gain_split = min_gain_split
        else:
            raise ValueError('The minimum value of gain to make a split must be greater or equal to 0')

        if split_chooser is not None:
            self._split_chooser = resolve_split_selection(split_chooser)
        else:
            raise ValueError('The split chooser can not be None.')

        if split_criterion is not None:
            self._split_criterion = resolve_split_criterion(split_criterion)
        else:
            raise ValueError('The split criterion can not be None.')

        if feature_selection is not None:
            self._feature_selection = resolve_feature_selection(feature_selection)
        else:
            raise ValueError('The feature selection criteria can not be None.')

    @property
    def max_depth(self):
        return self._max_depth

    @property
    def min_samples_leaf(self):
        return self._min_samples_leaf

    @property
    def min_samples_split(self):
        return self._min_samples_split

    @property
    def feature_prob(self):
        return self._feature_prob

    @property
    def min_gain_split(self):
        return self._min_gain_split

    @property
    def split_chooser(self):
        return self._split_chooser

    @property
    def split_criterion(self):
        return self._split_criterion

    @property
    def feature_selection(self):
        return self._feature_selection

    def fit(self, X, y):
        """
        Trains the decision tree classifier with (X, y).

        :param X: <numpy ndarray> An array containing the feature vectors
        :param y: <numpy array> An array containing the target features
        :return: self
        """
        X, y = check_X_y(X, y, dtype=None)
        self._encoder = LabelEncoder()
        y = self._encoder.fit_transform(y)
        self._n_instances, self._n_features = X.shape
        self._n_classes = utils.count_classes(y)

        self._tree_builder = TreeBuilder(split_criterion=self._split_criterion,
                                         feature_prob=self._feature_prob,
                                         feature_selection=self._feature_selection,
                                         max_depth=self._max_depth,
                                         min_samples_leaf=self._min_samples_leaf,
                                         min_gain_split=self._min_gain_split,
                                         min_samples_split=self._min_samples_split,
                                         split_chooser=self._split_chooser)
        self._tree = self._tree_builder.build_tree(X, y, self._n_classes)

        return self

    def predict(self, X, check_input=True):
        """
        Predicts the classes for the new instances in X.

        :param X: <numpy ndarray> An array containing the feature vectors
        :param check_input: <bool> If input array must be checked
        :return: <numpy array>
        """
        if check_input:
            X = self._validate_predict(X, check_input=check_input)

        sample_size, features_count = X.shape
        result = np.zeros(sample_size, dtype=int)
        for i in range(sample_size):
            x = X[i]
            result[i] = self._tree.predict(x)
        return self._encoder.inverse_transform(result)

    def predict_proba(self, X, check_input=True):
        """
        Predicts the class distribution probabilities for the new instances in X.

        :param X: <numpy ndarray> An array containing the feature vectors
        :param check_input: <bool> If input array must be checked
        :return: <numpy array>
        """
        if check_input:
            X = self._validate_predict(X, check_input=check_input)

        sample_size, features_count = X.shape
        result = list(range(sample_size))
        for i in range(sample_size):
            x = X[i]
            result[i] = self._tree.predict_proba(x)
        return result

    def _validate_predict(self, X, check_input):
        """
        Validate X whenever one tries to predict or predict_proba.

        :param X: <numpy ndarray> An array containing the feature vectors
        :param check_input: <bool> If input array must be checked
        :return: <bool>
        """
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
                 split_chooser='best',
                 split_criterion='gini',
                 min_samples_leaf=1,
                 feature_selection='log',
                 feature_prob=None,
                 min_gain_split=0,
                 min_samples_split=2):
        """
        Builds a decision forest for a classification problem.

        :param n_estimators: <int> Number of trees in the forest
        :param bootstrap: <bool> Whether to use bagging or not
        :param max_depth: <int> or <None> Defines the maximum depth of the tree
        :param split_chooser: <string> The name of the split chooser:
                            "best" for selecting the best possible split
                            "rand" for selecting a random split
        :param split_criterion: <string> The name of the split criterion:
                            "gini" for selecting the Gini criterion
                            "entropy" for selecting the Entropy criterion
        :param min_samples_leaf: <int> Minimum number of instances to place in a leaf
        :param feature_selection: <string> The name of the feature selection criteria:
                            "all" for selecting all features as candidate features
                            "log" for selecting log(n)+1 as candidate features
                            "prob" for selecting the candidate features according to its probabilities
        :param feature_prob: <list> Feature probabilities
        :param min_gain_split: <float> Minimum split gain value to consider splitting a node
        :param min_samples_split: <int> Minimum number of instances to consider creating a split
        """
        self._trees = None
        self._n_features = None
        self._n_instances = None
        self._tree_builder = None
        self._n_classes = None
        self._encoder = None

        # Ensemble parameters
        self._n_estimators = None
        self._bootstrap = bootstrap

        # Tree parameters
        self._max_depth = None
        self._min_samples_leaf = None
        self._min_samples_split = None
        self._feature_prob = None
        self._min_gain_split = None
        self._split_chooser = None
        self._split_criterion = None
        self._feature_selection = None

        if n_estimators is None or n_estimators > 0:
            self._n_estimators = n_estimators
        else:
            raise ValueError('The number of trees must be greater than 0.')

        if bootstrap is not None:
            self._bootstrap = bootstrap
        else:
            raise ValueError('The value of bootstrap can not be None.')

        if max_depth is None or max_depth > 0:
            self._max_depth = max_depth
        else:
            raise ValueError('The depth of the tree must be greater than 0.')

        if min_samples_leaf is not None and min_samples_leaf > 0:
            self._min_samples_leaf = min_samples_leaf
        else:
            raise ValueError('The minimum number of instances to place in a leaf must be greater than 0.')

        if min_samples_split is not None and min_samples_split > 1:
            self._min_samples_split = min_samples_split
        else:
            raise ValueError('The minimum number of instances to make a split must be greater than 1')

        if feature_prob is None or (utils.check_array_sum_one(feature_prob) and
                                    utils.check_positive_array(feature_prob)):
            self._feature_prob = feature_prob
        else:
            raise ValueError('The features probabilities must be positive values and the sum must be one')

        if min_gain_split is not None and min_gain_split >= 0:
            self._min_gain_split = min_gain_split
        else:
            raise ValueError('The minimum value of gain to make a split must be greater or equal to 0')

        if split_chooser is not None:
            self._split_chooser = resolve_split_selection(split_chooser)
        else:
            raise ValueError('The split chooser can not be None.')

        if split_criterion is not None:
            self._split_criterion = resolve_split_criterion(split_criterion)
        else:
            raise ValueError('The split criterion can not be None.')

        if feature_selection is not None:
            self._feature_selection = resolve_feature_selection(feature_selection)
        else:
            raise ValueError('The feature selection criteria can not be None.')

    def fit(self, X, y):
        """
        Trains the decision forest classifier with (X, y).

        :param X: <numpy ndarray> An array containing the feature vectors
        :param y: <numpy array> An array containing the target features
        :return: self
        """
        X, y = check_X_y(X, y, dtype=None)
        self._encoder = LabelEncoder()
        y = self._encoder.fit_transform(y)
        self._n_instances, self._n_features = X.shape
        self._n_classes = utils.count_classes(y)
        self._trees = []

        if self._bootstrap:
            set_generator = BaggingSet(self._n_instances)
        else:
            set_generator = SimpleSet(self._n_instances)

        self._tree_builder = TreeBuilder(split_criterion=self._split_criterion,
                                         feature_prob=self._feature_prob,
                                         feature_selection=self._feature_selection,
                                         max_depth=self._max_depth,
                                         min_samples_leaf=self._min_samples_leaf,
                                         min_gain_split=self._min_gain_split,
                                         min_samples_split=self._min_samples_split,
                                         split_chooser=self._split_chooser)

        for _ in range(self._n_estimators):
            ids = set_generator.training_ids()
            X_new = X[ids]
            y_new = y[ids]

            new_tree = self._tree_builder.build_tree(X_new, y_new, self._n_classes)

            if self._bootstrap:
                validation_ids = set_generator.oob_ids()
                new_tree.weight = accuracy_score(y[validation_ids], self._predict_on_tree(X[validation_ids], new_tree))

            self._trees.append(new_tree)
            set_generator.clear()

        return self

    def predict(self, X, check_input=True):
        """
        Predicts the classes for the new instances in X.

        :param X: <numpy ndarray> An array containing the feature vectors
        :param check_input: <bool> If input array must be checked
        :return: <numpy array>
        """
        if check_input:
            X = self._validate(X, check_input=check_input)

        voter = PerformanceWeightingVoter(self._trees, self._n_classes)

        sample_size, features_count = X.shape
        result = np.zeros(sample_size, dtype=int)
        for i in range(sample_size):
            x = X[i]
            result[i] = voter.predict(x)
        return self._encoder.inverse_transform(result)

    def predict_proba(self, X, check_input=True):
        """
        Predicts the class distribution probabilities for the new instances in X.

        :param X: <numpy ndarray> An array containing the feature vectors
        :param check_input: <bool> If input array must be checked
        :return: <numpy array>
        """
        if check_input:
            X = self._validate(X, check_input=check_input)

        voter = PerformanceWeightingVoter(self._trees, self._n_classes)

        sample_size, features_count = X.shape
        result = list(range(sample_size))
        for i in range(sample_size):
            x = X[i]
            result[i] = voter.predict_proba(x)
        return result

    def feature_importances(self):
        """
        Calculates the feature importances according to Breiman 2001.

        :return: <float>
        """
        importances = np.zeros(self._n_features)
        for tree in self._trees:
            importances += tree.feature_importances()
        importances /= len(self._trees)
        return importances

    def trees_mean_weight(self):
        """
        Calculates the mean weight of the trees in the forest.

        :return: <float>
        """
        weights = [tree.weight for tree in self._trees]
        mean_weight = np.mean(weights)
        return mean_weight

    def diversity_measure(self, X, y, diversity='pcd'):
        """
        Calculates the diversity measure for the forest.

        :param X: <numpy ndarray> An array containing the feature vectors
        :param y: <numpy array> An array containing the target features
        :param diversity: <string> The type of diversity to be calculated
                        "pcd" for Percentage of Correct Diversity
                        "qstat" for QStatistic Diversity
        :return: <float>
        """
        X, y = check_X_y(X, y, dtype=None)
        y = self._encoder.transform(y)

        if diversity == 'pcd':
            metric = PercentageCorrectDiversity()
        elif diversity == 'qstat':
            metric = QStatisticDiversity()
        else:
            raise ValueError("It was not possible to recognize the diversity measure.")

        forest_diversity = metric.get_measure(self._trees, X, y)
        return forest_diversity

    def _validate(self, X, check_input):
        """
        Validate X whenever one tries to predict or predict_proba.

        :param X: <numpy ndarray> An array containing the feature vectors
        :param check_input: <bool> If input array must be checked
        :return: <bool>
        """
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

    def _predict_on_tree(self, X, tree, check_input=True):
        """
        Predicts the classes for the new instances in X.

        :param X: <numpy ndarray> An array containing the feature vectors
        :param tree: <DecisionTree> The tree in which to predict
        :param check_input: <bool> If input array must be checked
        :return: <numpy array>
        """
        if check_input:
            X = self._validate(X, check_input=check_input)

        sample_size, features_count = X.shape
        result = np.zeros(sample_size, dtype=int)
        for i in range(sample_size):
            x = X[i]
            result[i] = tree.predict(x)
        return result


class ProactiveForestClassifier(DecisionForestClassifier):
    def __init__(self,
                 n_estimators=100,
                 bootstrap=True,
                 max_depth=None,
                 split_chooser='best',
                 split_criterion='gini',
                 min_samples_leaf=1,
                 feature_selection='log',
                 feature_prob=None,
                 min_gain_split=0,
                 min_samples_split=2,
                 alpha=0.1):
        """
        Builds a proactive forest for a classification problem.

        :param n_estimators: <int> Number of trees in the forest
        :param bootstrap: <bool> Whether to use bagging or not
        :param max_depth: <int> or <None> Defines the maximum depth of the tree
        :param split_chooser: <string> The name of the split chooser:
                            "best" for selecting the best possible split
                            "rand" for selecting a random split
        :param split_criterion: <string> The name of the split criterion:
                            "gini" for selecting the Gini criterion
                            "entropy" for selecting the Entropy criterion
        :param min_samples_leaf: <int> Minimum number of instances to place in a leaf
        :param feature_selection: <string> The name of the feature selection criteria:
                            "all" for selecting all features as candidate features
                            "log" for selecting log(n)+1 as candidate features
                            "prob" for selecting the candidate features according to its probabilities
        :param feature_prob: <list> Feature probabilities
        :param min_gain_split: <float> Minimum split gain value to consider splitting a node
        :param min_samples_split: <int> Minimum number of instances to consider creating a split
        :param alpha: <float> Diversity rate. It can take values from (0, 1]
        """
        if 0 < alpha <= 1:
            self.alpha = alpha
        else:
            raise ValueError("The diversity rate can only take values from (0, 1].")
        super().__init__(n_estimators=n_estimators,
                         bootstrap=bootstrap,
                         max_depth=max_depth,
                         split_chooser=split_chooser,
                         split_criterion=split_criterion,
                         min_samples_leaf=min_samples_leaf,
                         feature_selection=feature_selection,
                         feature_prob=feature_prob,
                         min_gain_split=min_gain_split,
                         min_samples_split=min_samples_split
                         )

    def fit(self, X, y):
        """
        Trains the decision forest classifier with (X, y).

        :param X: <numpy ndarray> An array containing the feature vectors
        :param y: <numpy array> An array containing the target features
        :return: self
        """
        X, y = check_X_y(X, y, dtype=None)
        self._encoder = LabelEncoder()
        y = self._encoder.fit_transform(y)
        self._n_instances, self._n_features = X.shape
        self._n_classes = utils.count_classes(y)
        self._trees = []

        if self._bootstrap:
            set_generator = BaggingSet(self._n_instances)
        else:
            set_generator = SimpleSet(self._n_instances)

        ledger = FIProbabilityLedger(probabilities=self._feature_prob, n_features=self._n_features, alpha=self.alpha)

        self._tree_builder = TreeBuilder(split_criterion=self._split_criterion,
                                         feature_prob=ledger.probabilities,
                                         feature_selection=self._feature_selection,
                                         max_depth=self._max_depth,
                                         min_samples_leaf=self._min_samples_leaf,
                                         min_gain_split=self._min_gain_split,
                                         min_samples_split=self._min_samples_split,
                                         split_chooser=self._split_chooser)

        for i in range(1, self._n_estimators+1):

            ids = set_generator.training_ids()
            X_new = X[ids]
            y_new = y[ids]

            new_tree = self._tree_builder.build_tree(X_new, y_new, self._n_classes)

            if self._bootstrap:
                validation_ids = set_generator.oob_ids()
                new_tree.weight = accuracy_score(y[validation_ids], self._predict_on_tree(X[validation_ids], new_tree))

            self._trees.append(new_tree)
            set_generator.clear()

            ledger.update_probabilities(new_tree, rate=i/self._n_estimators)
            self._tree_builder.feature_prob = ledger.probabilities

        return self
