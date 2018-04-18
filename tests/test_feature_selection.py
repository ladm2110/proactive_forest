import numpy as np
from unittest import TestCase
from proactive_forest.feature_selection import AllFeatureSelection, LogFeatureSelection, ProbFeatureSelection


class AllFeatureSelectionTest(TestCase):
    def setUp(self):
        self.selection = AllFeatureSelection()

    def tearDown(self):
        pass

    def test_get_features(self):
        n_features = 5
        result = self.selection.get_features(5)
        expected = [x for x in range(n_features)]

        self.assertEqual(result, expected)


class LogFeatureSelectionTest(TestCase):
    def setUp(self):
        self.selection = LogFeatureSelection()

    def tearDown(self):
        pass

    def test_get_features(self):
        n_features = 5
        result = self.selection.get_features(n_features)
        size_expected = np.math.floor(np.math.log2(n_features)) + 1

        assert len(result) == size_expected

        for i in result:
            self.assertIn(i, [x for x in range(n_features)])


class ProbFeatureSelectionTest(TestCase):
    def setUp(self):
        self.selection = ProbFeatureSelection()

    def tearDown(self):
        pass

    def test_get_features(self):
        n_features = 5
        result = self.selection.get_features(n_features)

        for i in result:
            self.assertIn(i, [x for x in range(n_features)])
