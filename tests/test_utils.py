from unittest import TestCase
import proactive_forest.utils as utils
from examples.load_data import load_iris, load_car, load_credit, load_vowel
from sklearn.utils import check_X_y
import numpy as np


class UtilsTest(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_all_instances_same_class_returns_true(self):
        y_true = np.array([1, 1, 1, 1, 1])

        expected_value = True
        returned_value = utils.all_instances_same_class(y_true)

        assert returned_value == expected_value

    def test_all_instances_same_class_returns_false(self):
        y_false = np.array([1, 0, 1, 0, 1])

        expected_value = False
        returned_value = utils.all_instances_same_class(y_false)

        assert returned_value == expected_value

    def test_categorical_data_returns_true(self):
        data = np.array(['a', 'b', 'c'])
        self.assertTrue(utils.categorical_data(data))

    def test_categorical_data_returns_false(self):
        data = np.array([1, 2, 3, 1, 1])
        self.assertFalse(utils.categorical_data(data))

    def test_categorical_data_on_real_data(self):
        X, y = load_car()
        X, y = check_X_y(X, y, dtype=None)
        self.assertTrue(utils.categorical_data(X[:, 0]))

    def test_categorical_data_on_real_data_2(self):
        X, y = load_iris()
        X, y = check_X_y(X, y, dtype=None)
        self.assertFalse(utils.categorical_data(X[:, 0]))

    def test_categorical_data_on_real_data_3(self):
        X, y = load_credit()
        X, y = check_X_y(X, y, dtype=None)
        self.assertTrue(utils.categorical_data(X[:, 0]))

    def test_categorical_data_on_real_data_4(self):
        X, y = load_vowel()
        X, y = check_X_y(X, y, dtype=None)
        self.assertFalse(utils.categorical_data(X[:, 10]))

    def test_bin_count(self):
        data = np.array([0, 1, 0, 5, 0, 3, 1, 0, 0])
        expected = [5, 2, 0, 1, 0, 1, 0]
        returned = utils.bin_count(data, length=7)
        assert expected == returned

    def test_bin_count_2(self):
        data = np.array([0, 1, 0, 0, 0, 1, 1, 0, 0])
        expected = [6, 3]
        returned = utils.bin_count(data, length=2)
        assert expected == returned

    def test_bin_count_3(self):
        data = np.array([0, 1, 0, 0, 0, 1, 1, 0, 0])
        expected = [6, 3, 0]
        returned = utils.bin_count(data, length=3)
        assert expected == returned

    def test_count_classes(self):
        data = np.array([0, 1, 1, 2, 0, 1])
        expected = 3
        returned = utils.count_classes(data)
        assert expected == returned

    def test_check_positive_array(self):
        array = [1, 3, 2, 8]
        self.assertTrue(utils.check_positive_array(array))

    def test_check_array_sum_one(self):
        array = [0.25, 0.25, 0.25, 0.25]
        self.assertTrue(utils.check_array_sum_one(array))
