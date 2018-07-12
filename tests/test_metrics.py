from unittest import TestCase
from proactive_forest.metrics import GiniCriterion, EntropyCriterion


class GiniMetricTest(TestCase):

    def setUp(self):
        self.criterion = GiniCriterion()

    def tearDown(self):
        pass

    def test_empty_list(self):
        x = []
        expected_value = 0
        result = self.criterion.impurity(x)

        assert expected_value == result

    def test_all_same_class(self):
        x = [0 for _ in range(5)]
        expected_value = 0
        result = self.criterion.impurity(x)

        assert expected_value == result

    def test_all_same_class_2(self):
        x = [2 for _ in range(10)]
        expected_value = 0
        result = self.criterion.impurity(x)

        assert expected_value == result

    def test_two_different_class(self):
        x = [0 if n % 2 == 0 else 1 for n in range(6)]
        expected_value = 0.5
        result = self.criterion.impurity(x)

        assert expected_value == result

    def test_three_different_class(self):
        x = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        expected_value = 0.66667
        result = self.criterion.impurity(x)

        self.assertAlmostEqual(expected_value, result, 4)


class EntropyMetricTest(TestCase):

    def setUp(self):
        self.criterion = EntropyCriterion()

    def tearDown(self):
        pass

    def test_empty_list(self):
        x = []
        expected_value = 0
        result = self.criterion.impurity(x)

        assert expected_value == result

    def test_all_same_class(self):
        x = [0 for _ in range(5)]
        expected_value = 0
        result = self.criterion.impurity(x)

        assert expected_value == result

    def test_all_same_class_2(self):
        x = [2 for _ in range(10)]
        expected_value = 0
        result = self.criterion.impurity(x)

        assert expected_value == result

    def test_two_different_class(self):
        x = [0 if n % 2 == 0 else 2 for n in range(6)]
        expected_value = 1
        result = self.criterion.impurity(x)

        assert expected_value == result

    def test_three_different_class(self):
        x = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        expected_value = 1.58496
        result = self.criterion.impurity(x)

        self.assertAlmostEqual(expected_value, result, 4)
