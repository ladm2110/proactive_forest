from unittest import TestCase, mock
import numpy as np
import proactive_forest.splits as splits
from proactive_forest.metrics import SplitCriterion


class ComputeSplitValuesTest(TestCase):
    def setUp(self):
        self.data = np.array([4, 3, 4, 1, 1, 1, 3, 1])
        self.data_cat = np.array(['a', 'a', 'b', 'c', 'c', 'c'])

    def tearDown(self):
        pass

    def test_computes_all_possible_split_values_for_numerical_data(self):
        expected = [2.0, 3.5]
        returned = splits.compute_split_values(self.data)

        for value in returned:
            self.assertIn(value, expected)

    def test_computes_all_possible_split_values_for_categorical_data(self):
        expected = ['a', 'b', 'c']
        returned = splits.compute_split_values(self.data_cat)

        for value in returned:
            self.assertIn(value, expected)


class ComputeSplitGainTest(TestCase):
    def test_compute_split_gain(self):
        y = [0, 1, 0, 1, 0, 0, 0]
        y_left = [0, 0, 0]
        y_right = [0, 1, 0, 1]
        split_criterion = mock.MagicMock(spec=SplitCriterion)

        def helper(x):
            if x == [0, 1, 0, 1, 0, 0, 0]:
                return 0.88
            elif x == [0, 1, 0, 1]:
                return 0.50
            else:
                return 0

        split_criterion.impurity_gain.side_effect = helper

        expected_value = 0.594
        returned_value = splits.compute_split_gain(split_criterion, y, y_left, y_right)

        self.assertAlmostEqual(returned_value, expected_value, places=2)


class SplitNumericalTest(TestCase):
    def setUp(self):
        self.data = np.array([1, 1, 4, 2, 3, 6, 10, 1, 2]).reshape((3, 3))
        self.target = np.array([0, 0, 1])

    def tearDown(self):
        pass

    def test_correct_split_of_data_on_0(self):
        split_value = 2

        X_left = [[1, 1, 4],
                  [2, 3, 6]]
        X_right = [[10, 1, 2]]
        y_left = [0, 0]
        y_right = [1]

        returned = splits.split_numerical_data(self.data, self.target, feature_id=0, value=split_value)

        self.assertListEqual(X_left, returned[0].tolist())
        self.assertListEqual(X_right, returned[1].tolist())
        self.assertListEqual(y_left, returned[2].tolist())
        self.assertListEqual(y_right, returned[3].tolist())

    def test_correct_split_of_data_on_1(self):
        split_value = 1

        X_left = [[1, 1, 4],
                  [10, 1, 2]]
        X_right = [[2, 3, 6]]
        y_left = [0, 1]
        y_right = [0]

        returned = splits.split_numerical_data(self.data, self.target, feature_id=1, value=split_value)

        self.assertListEqual(X_left, returned[0].tolist())
        self.assertListEqual(X_right, returned[1].tolist())
        self.assertListEqual(y_left, returned[2].tolist())
        self.assertListEqual(y_right, returned[3].tolist())

    def test_correct_split_of_data_on_2(self):
        split_value = 2

        X_left = [[10, 1, 2]]
        X_right = [[1, 1, 4],
                   [2, 3, 6]]
        y_left = [1]
        y_right = [0, 0]

        returned = splits.split_numerical_data(self.data, self.target, feature_id=2, value=split_value)

        self.assertListEqual(X_left, returned[0].tolist())
        self.assertListEqual(X_right, returned[1].tolist())
        self.assertListEqual(y_left, returned[2].tolist())
        self.assertListEqual(y_right, returned[3].tolist())


class SplitCategoricalTest(TestCase):
    def setUp(self):
        self.data = np.array(['A', 'B', 'C', 'A', 'A', 'C', 'B', 'B', 'D']).reshape((3, 3))
        self.target = np.array([0, 0, 1])

    def tearDown(self):
        pass

    def test_correct_split_of_data_on_0(self):
        split_value = 'A'

        X_left = [['A', 'B', 'C'],
                  ['A', 'A', 'C']]
        X_right = [['B', 'B', 'D']]
        y_left = [0, 0]
        y_right = [1]

        returned = splits.split_categorical_data(self.data, self.target, feature_id=0, value=split_value)

        self.assertListEqual(X_left, returned[0].tolist())
        self.assertListEqual(X_right, returned[1].tolist())
        self.assertListEqual(y_left, returned[2].tolist())
        self.assertListEqual(y_right, returned[3].tolist())

    def test_correct_split_of_data_on_1(self):
        split_value = 'B'

        X_left = [['A', 'B', 'C'],
                  ['B', 'B', 'D']]
        X_right = [['A', 'A', 'C']]
        y_left = [0, 1]
        y_right = [0]

        returned = splits.split_categorical_data(self.data, self.target, feature_id=1, value=split_value)

        self.assertListEqual(X_left, returned[0].tolist())
        self.assertListEqual(X_right, returned[1].tolist())
        self.assertListEqual(y_left, returned[2].tolist())
        self.assertListEqual(y_right, returned[3].tolist())

    def test_correct_split_of_data_on_2(self):
        split_value = 'C'

        X_left = [['A', 'B', 'C'],
                  ['A', 'A', 'C']]
        X_right = [['B', 'B', 'D']]
        y_left = [0, 0]
        y_right = [1]

        returned = splits.split_categorical_data(self.data, self.target, feature_id=2, value=split_value)

        self.assertListEqual(X_left, returned[0].tolist())
        self.assertListEqual(X_right, returned[1].tolist())
        self.assertListEqual(y_left, returned[2].tolist())
        self.assertListEqual(y_right, returned[3].tolist())


class SplitDataTest(TestCase):
    def setUp(self):
        self.data_num = np.array([1, 1, 4, 2, 3, 6, 10, 1, 2]).reshape((3, 3))
        self.data_cat = np.array(['A', 'B', 'C', 'A', 'A', 'C', 'B', 'B', 'D']).reshape((3, 3))
        self.target = np.array([0, 0, 1])

    def tearDown(self):
        pass

    def test_split_data_num(self):
        split_value = 2

        X_left = [[1, 1, 4],
                  [2, 3, 6]]
        X_right = [[10, 1, 2]]
        y_left = [0, 0]
        y_right = [1]

        returned = splits.split_data(self.data_num, self.target, feature_id=0, value=split_value)

        self.assertListEqual(X_left, returned[0].tolist())
        self.assertListEqual(X_right, returned[1].tolist())
        self.assertListEqual(y_left, returned[2].tolist())
        self.assertListEqual(y_right, returned[3].tolist())

    def test_split_data_cat(self):
        split_value = 'A'

        X_left = [['A', 'B', 'C'],
                  ['A', 'A', 'C']]
        X_right = [['B', 'B', 'D']]
        y_left = [0, 0]
        y_right = [1]

        returned = splits.split_data(self.data_cat, self.target, feature_id=0, value=split_value)

        self.assertListEqual(X_left, returned[0].tolist())
        self.assertListEqual(X_right, returned[1].tolist())
        self.assertListEqual(y_left, returned[2].tolist())
        self.assertListEqual(y_right, returned[3].tolist())