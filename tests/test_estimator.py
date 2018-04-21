from unittest import TestCase, mock
import numpy as np
from proactive_forest.estimator import DecisionTreeClassifier
from proactive_forest.tree import DecisionTree
from sklearn.preprocessing import LabelEncoder


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

    def test_predict_empty_vector(self):
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
        expected_prediction = [1, 0]
        resulted_prediction = self.decision_tree.predict(x)

        for resulted, expected in zip(resulted_prediction, expected_prediction):
            self.assertEqual(resulted, expected)

    def test_predict_proba(self):
        pass




