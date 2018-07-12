from unittest import TestCase, mock
import numpy as np
from proactive_forest.estimator import DecisionTreeClassifier, DecisionForestClassifier, ProactiveForestClassifier
from proactive_forest.tree import DecisionTree
from proactive_forest.splits import BestSplitChooser
from proactive_forest.metrics import GiniCriterion
from proactive_forest.feature_selection import AllFeatureSelection
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError


class DecisionTreeClassifierInitializationTest(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_max_depth_exception_negative_value(self):
        with self.assertRaises(ValueError):
            self.decision_tree = DecisionTreeClassifier(max_depth=-1)

    def test_max_depth_none_value(self):
        self.decision_tree = DecisionTreeClassifier(max_depth=None)
        self.assertIsNone(self.decision_tree.max_depth)

    def test_max_depth_positive_value(self):
        self.decision_tree = DecisionTreeClassifier(max_depth=1)
        self.assertEqual(self.decision_tree.max_depth, 1)

    def test_min_samples_leaf_exception_negative_value(self):
        with self.assertRaises(ValueError):
            self.decision_tree = DecisionTreeClassifier(min_samples_leaf=-1)

    def test_min_samples_leaf_exception_none_value(self):
        with self.assertRaises(ValueError):
            self.decision_tree = DecisionTreeClassifier(min_samples_leaf=None)

    def test_min_samples_leaf_positive_value(self):
        self.decision_tree = DecisionTreeClassifier(min_samples_leaf=1)
        self.assertEqual(self.decision_tree.min_samples_leaf, 1)

    def test_min_samples_split_exception_none_value(self):
        with self.assertRaises(ValueError):
            self.decision_tree = DecisionTreeClassifier(min_samples_split=None)

    def test_min_samples_split_exception_less_than_two_instances(self):
        with self.assertRaises(ValueError):
            self.decision_tree = DecisionTreeClassifier(min_samples_split=1)

    def test_min_samples_split_positive_value_greater_than_one(self):
        self.decision_tree = DecisionTreeClassifier(min_samples_split=2)
        self.assertEqual(self.decision_tree.min_samples_split, 2)

    def test_feature_prob_none_value(self):
        self.decision_tree = DecisionTreeClassifier(feature_prob=None)
        self.assertIsNone(self.decision_tree.feature_prob)

    def test_feature_prob_exception_negative_values(self):
        with self.assertRaises(ValueError):
            self.decision_tree = DecisionTreeClassifier(feature_prob=[-0.2, 0.4, 0.4])

    def test_feature_prob_exception_not_sum_one(self):
        with self.assertRaises(ValueError):
            self.decision_tree = DecisionTreeClassifier(feature_prob=[0.2, 0.4, 0.5])

    def test_feature_prob_positive_values_sum_one(self):
        self.decision_tree = DecisionTreeClassifier(feature_prob=[0.25, 0.25, 0.25, 0.25])
        self.assertEqual(self.decision_tree.feature_prob, [0.25, 0.25, 0.25, 0.25])

    def test_min_gain_split_exception_none_value(self):
        with self.assertRaises(ValueError):
            self.decision_tree = DecisionTreeClassifier(min_gain_split=None)

    def test_min_gain_split_exception_negative_value(self):
        with self.assertRaises(ValueError):
            self.decision_tree = DecisionTreeClassifier(min_gain_split=-1)

    def test_min_gain_split_non_negative_value(self):
        self.decision_tree = DecisionTreeClassifier(min_gain_split=1)
        self.assertEqual(self.decision_tree.min_gain_split, 1)

    def test_split_chooser_exception_none_value(self):
        with self.assertRaises(ValueError):
            self.decision_tree = DecisionTreeClassifier(split_chooser=None)

    def test_split_chooser_exception_non_admissible_value(self):
        with self.assertRaises(ValueError):
            self.decision_tree = DecisionTreeClassifier(split_chooser='non')

    def test_split_chooser_admissible_value(self):
        self.decision_tree = DecisionTreeClassifier(split_chooser='best')
        self.assertIsInstance(self.decision_tree._split_chooser, BestSplitChooser)

    def test_split_criterion_exception_none_value(self):
        with self.assertRaises(ValueError):
            self.decision_tree = DecisionTreeClassifier(split_criterion=None)

    def test_split_criterion_exception_non_admissible_value(self):
        with self.assertRaises(ValueError):
            self.decision_tree = DecisionTreeClassifier(split_criterion='non')

    def test_split_criterion_admissible_value(self):
        self.decision_tree = DecisionTreeClassifier(split_criterion='gini')
        self.assertIsInstance(self.decision_tree._split_criterion, GiniCriterion)

    def test_feature_selection_exception_none_value(self):
        with self.assertRaises(ValueError):
            self.decision_tree = DecisionTreeClassifier(feature_selection=None)

    def test_feature_selection_exception_non_admissible_value(self):
        with self.assertRaises(ValueError):
            self.decision_tree = DecisionTreeClassifier(feature_selection='non')

    def test_feature_selection_admissible_value(self):
        self.decision_tree = DecisionTreeClassifier(feature_selection='all')
        self.assertIsInstance(self.decision_tree._feature_selection, AllFeatureSelection)


class DecisionTreeClassifierTest(TestCase):
    def setUp(self):
        self.decision_tree = DecisionTreeClassifier()
        self.X = np.array(['A', 'B', 'A', 'B', 'B', 'C', 'A', 'C', 'B']).reshape((3, 3))
        self.y = np.array([1, 1, 0])

    def tearDown(self):
        pass

    def test_fit(self):
        self.decision_tree.fit(self.X, self.y)
        self.assertIsNotNone(self.decision_tree._encoder)
        self.assertIsNotNone(self.decision_tree._tree_builder)
        self.assertIsNotNone(self.decision_tree._tree)
        self.assertIsInstance(self.decision_tree._tree, DecisionTree)
        print(self.decision_tree._tree.nodes)

    def test_predict_empty_vector(self):
        self.decision_tree.fit(self.X, self.y)
        x = np.array([]).reshape(0, 0)

        with self.assertRaises(ValueError):
            self.decision_tree.predict(x)

    def test_predict_one_instance(self):
        self.decision_tree._tree = mock.MagicMock(spec=DecisionTree)
        self.decision_tree._tree.predict.return_value = 1
        self.decision_tree._n_features = 3
        self.decision_tree._encoder = mock.MagicMock(spec=LabelEncoder)
        self.decision_tree._encoder.inverse_transform.return_value = [1]

        x = np.array(['A', 'B', 'A']).reshape((1, 3))
        expected_length_prediction = 1
        resulted_length_prediction = len(self.decision_tree.predict(x))

        self.assertEqual(resulted_length_prediction, expected_length_prediction)

    def test_predict_two_instances(self):
        self.decision_tree._tree = mock.MagicMock(spec=DecisionTree)
        self.decision_tree._tree.predict.side_effect = [1, 0]
        self.decision_tree._n_features = 3
        self.decision_tree._encoder = mock.MagicMock(spec=LabelEncoder)
        self.decision_tree._encoder.inverse_transform.return_value = [1, 0]

        x = np.array(['A', 'B', 'A', 'C', 'C', 'A']).reshape((2, 3))
        expected_prediction = 2
        resulted_prediction = len(self.decision_tree.predict(x))

        self.assertEqual(resulted_prediction, expected_prediction)

    def test_predict_proba_one_instance(self):
        self.decision_tree._tree = mock.MagicMock(spec=DecisionTree)
        self.decision_tree._tree.predict_proba.return_value = [0.25, 0.75]
        self.decision_tree._n_features = 3

        x = np.array(['A', 'B', 'A']).reshape((1, 3))
        expected_length_prediction = 2
        resulted_prediction = self.decision_tree.predict_proba(x)

        self.assertEqual(len(resulted_prediction), 1)

        for resulted in resulted_prediction:
            self.assertEqual(len(resulted), expected_length_prediction)

    def test_predict_proba_two_instance(self):
        self.decision_tree._tree = mock.MagicMock(spec=DecisionTree)
        self.decision_tree._tree.predict_proba.return_value = [[0.25, 0.75], [0.33, 0.67]]
        self.decision_tree._n_features = 3

        x = np.array(['A', 'B', 'A', 'C', 'C', 'A']).reshape((2, 3))
        expected_length_prediction = 2
        resulted_prediction = self.decision_tree.predict_proba(x)

        self.assertEqual(len(resulted_prediction), 2)

        for resulted in resulted_prediction:
            self.assertEqual(len(resulted), expected_length_prediction)


class DecisionForestClassifierInitializationTest(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_n_estimators_exception_inadmissible_value(self):
        with self.assertRaises(ValueError):
            self.decision_forest = DecisionForestClassifier(n_estimators=-1)

    def test_n_estimators_none_value(self):
        self.decision_forest = DecisionForestClassifier(n_estimators=None)
        self.assertIsNone(self.decision_forest.n_estimators)

    def test_n_estimators_admissible_value(self):
        self.decision_forest = DecisionForestClassifier(n_estimators=12)
        self.assertEqual(self.decision_forest.n_estimators, 12)

    def test_bootstrap_exception_none_value(self):
        with self.assertRaises(ValueError):
            self.decision_forest = DecisionForestClassifier(bootstrap=None)

    def test_bootstrap_inadmissible_value(self):
        pass

    def test_bootstrap_admissible_value(self):
        self.decision_forest = DecisionForestClassifier(bootstrap=True)
        self.assertTrue(self.decision_forest.bootstrap)


class DecisionForestClassifierTest(TestCase):
    def setUp(self):
        self.decision_forest = DecisionForestClassifier()
        self.X = np.array(['A', 'B', 'A', 'B', 'B', 'C', 'A', 'C', 'B']).reshape((3, 3))
        self.y = np.array([1, 1, 0])

    def tearDown(self):
        pass

    def test_fit(self):
        self.decision_forest.fit(self.X, self.y)
        self.assertIsNotNone(self.decision_forest._encoder)
        self.assertIsNotNone(self.decision_forest._tree_builder)
        self.assertIsNotNone(self.decision_forest._trees)
        self.assertIsInstance(self.decision_forest._trees, list)
        self.assertEqual(len(self.decision_forest._trees), self.decision_forest.n_estimators)

    def test_predict_one_instance(self):
        self.decision_forest._n_features = 3
        self.decision_forest._n_classes = 2

        tree_1 = mock.MagicMock(spec=DecisionTree)
        tree_1.weight = 1
        tree_1.predict.return_value = 1

        tree_2 = mock.MagicMock(spec=DecisionTree)
        tree_2.weight = 1
        tree_2.predict.return_value = 0

        tree_3 = mock.MagicMock(spec=DecisionTree)
        tree_3.weight = 1
        tree_3.predict.return_value = 1

        self.decision_forest._encoder = mock.MagicMock(spec=LabelEncoder)
        self.decision_forest._encoder.inverse_transform.return_value = 1

        self.decision_forest._trees = [tree_1, tree_2, tree_3]

        expected_prediction = 1
        a = np.array(['A', 'B', 'A']).reshape((1, 3))
        resulted_prediction = self.decision_forest.predict(a)
        self.assertEqual(expected_prediction, resulted_prediction)

    def test_predict_two_instances(self):
        self.decision_forest._n_features = 3
        self.decision_forest._n_classes = 2

        tree_1 = mock.MagicMock(spec=DecisionTree)
        tree_1.weight = 1
        tree_1.predict.return_value = 1

        tree_2 = mock.MagicMock(spec=DecisionTree)
        tree_2.weight = 1
        tree_2.predict.return_value = 0

        tree_3 = mock.MagicMock(spec=DecisionTree)
        tree_3.weight = 1
        tree_3.predict.return_value = 1

        self.decision_forest._encoder = mock.MagicMock(spec=LabelEncoder)
        self.decision_forest._encoder.inverse_transform.return_value = [1, 1]

        self.decision_forest._trees = [tree_1, tree_2, tree_3]

        expected_len_prediction = 2
        x = np.array(['A', 'B', 'A', 'C', 'A', 'A']).reshape((2, 3))
        resulted_prediction = self.decision_forest.predict(x)
        self.assertEqual(expected_len_prediction, len(resulted_prediction))

    def test_feature_importances(self):
        self.decision_forest._n_features = 3
        tree_1 = mock.MagicMock(spec=DecisionTree)
        tree_1.feature_importances.return_value = [0.2, 0.2, 0.2]

        tree_2 = mock.MagicMock(spec=DecisionTree)
        tree_2.feature_importances.return_value = [0.3, 0.3, 0.3]

        tree_3 = mock.MagicMock(spec=DecisionTree)
        tree_3.feature_importances.return_value = [0.4, 0.4, 0.4]

        self.decision_forest._trees = [tree_1, tree_2, tree_3]
        expected_feature_importances = [0.3, 0.3, 0.3]
        resulted_feature_importances = self.decision_forest.feature_importances()
        self.assertEqual(len(expected_feature_importances), len(resulted_feature_importances))
        for a, b in zip(expected_feature_importances, resulted_feature_importances):
            self.assertAlmostEqual(a, b, places=2)

    def test_trees_mean_weight(self):
        tree_1 = mock.MagicMock(spec=DecisionTree)
        tree_1.weight = 1

        tree_2 = mock.MagicMock(spec=DecisionTree)
        tree_2.weight = 0.8

        tree_3 = mock.MagicMock(spec=DecisionTree)
        tree_3.weight = 0.8

        self.decision_forest._trees = [tree_1, tree_2, tree_3]
        expected_weight = 0.87
        resulted_weight = self.decision_forest.trees_mean_weight()
        self.assertAlmostEqual(expected_weight, resulted_weight, places=2)

    def test_diversity_measure_exception(self):
        x = np.array(['A', 'B', 'A', 'C', 'A', 'A']).reshape((2, 3))
        y = [1, 0]

        self.decision_forest._encoder = mock.MagicMock(spec=LabelEncoder)
        self.decision_forest._encoder.transform.return_value = [1, 0]

        with self.assertRaises(ValueError):
            self.decision_forest.diversity_measure(x, y, diversity='kappa')

    def test_diversity_measure(self):
        tree_1 = mock.MagicMock(spec=DecisionTree)
        tree_1.predict.return_value = 1

        tree_2 = mock.MagicMock(spec=DecisionTree)
        tree_2.predict.return_value = 1

        tree_3 = mock.MagicMock(spec=DecisionTree)
        tree_3.predict.return_value = 0

        self.decision_forest._trees = [tree_1, tree_2, tree_3]
        x = np.array(['A', 'B', 'A', 'C', 'A', 'A']).reshape((2, 3))
        y = [1, 0]

        self.decision_forest._encoder = mock.MagicMock(spec=LabelEncoder)
        self.decision_forest._encoder.transform.return_value = [1, 0]

        self.assertIsNotNone(self.decision_forest.diversity_measure(x, y))

    def test__validate_exception_not_fitted(self):
        x = np.array(['A', 'B', 'A', 'C', 'A', 'A']).reshape((2, 3))
        with self.assertRaises(NotFittedError):
            self.decision_forest._validate(x, False)

    def test__validate_exception_n_instances(self):
        self.decision_forest._trees = mock.MagicMock(spec=DecisionTree)
        x = np.array(['A', 'B', 'A', 'C', 'A', 'A']).reshape((2, 3))
        self.decision_forest._n_features = 1
        with self.assertRaises(ValueError):
            self.decision_forest._validate(x, False)

    def test__predict_on_tree(self):
        tree = mock.MagicMock(spec=DecisionTree)
        tree.predict.return_value = 1
        x = np.array(['A', 'B', 'A', 'C', 'A', 'A']).reshape((2, 3))

        expected_prediction = [1, 1]
        resulted_prediction = self.decision_forest._predict_on_tree(x, tree, False)
        for expected, resulted in zip(expected_prediction, resulted_prediction):
            self.assertEqual(expected, resulted)


class ProactiveForestClassifierTest(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_alpha_exception_non_admissible_value(self):
        with self.assertRaises(ValueError):
            proactive_forest = ProactiveForestClassifier(alpha=2)

    def test_fit(self):
        proactive_forest = ProactiveForestClassifier()
        x = np.array(['A', 'B', 'A', 'B', 'B', 'C', 'A', 'C', 'B']).reshape((3, 3))
        y = np.array([1, 1, 0])
        proactive_forest.fit(x, y)
        self.assertIsNotNone(proactive_forest._encoder)
        self.assertIsNotNone(proactive_forest._tree_builder)
        self.assertIsNotNone(proactive_forest._trees)
        self.assertIsInstance(proactive_forest._trees, list)
        self.assertEqual(len(proactive_forest._trees), proactive_forest.n_estimators)
        self.assertIsNotNone(proactive_forest._tree_builder.feature_prob)
        self.assertEqual(len(proactive_forest._tree_builder.feature_prob), proactive_forest._n_features)