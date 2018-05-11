from unittest import TestCase
import numpy as np
from proactive_forest.tree import DecisionTree, DecisionForkNumerical, DecisionLeaf, DecisionFork, \
    DecisionForkCategorical


class DecisionTreeTest(TestCase):
    def setUp(self):
        self.dt = DecisionTree(n_features=3)

        df_1 = DecisionForkNumerical([70, 95], 1, 0, 0.3, 1.5)
        df_1.left_branch = 1
        df_1.right_branch = 4

        df_2 = DecisionForkNumerical([40, 30], 2, 2, 0.65, 0.5)
        df_2.left_branch = 2
        df_2.right_branch = 3

        df_3 = DecisionForkNumerical([30, 65], 2, 1, 0.75, 3.0)
        df_3.left_branch = 5
        df_3.right_branch = 6

        dl_1 = DecisionLeaf([30, 10], 3, 0)
        dl_2 = DecisionLeaf([10, 20], 3, 1)
        dl_3 = DecisionLeaf([25, 15], 3, 0)
        dl_4 = DecisionLeaf([5, 50], 3, 1)

        self.dt.nodes = [df_1, df_2, dl_1, dl_2, df_3, dl_3, dl_4]

    def tearDown(self):
        self.dt = None

    def test_predict(self):
        self.assertEqual(0, self.dt.predict(np.array([0.5, 4, 0.3])))
        self.assertEqual(1, self.dt.predict(np.array([0.5, 2, 1.3])))
        self.assertEqual(0, self.dt.predict(np.array([2.5, 2.2, 0.3])))
        self.assertEqual(1, self.dt.predict(np.array([2.5, 3.9, 1.3])))

    def test_predict_proba(self):
        for a, b in zip([0.738, 0.2619], self.dt.predict_proba(np.array([0.5, 4, 0.3]))):
            self.assertAlmostEquals(a, b, places=2)
        for a, b in zip([0.343, 0.6562], self.dt.predict_proba(np.array([0.5, 2, 1.3]))):
            self.assertAlmostEquals(a, b, places=2)
        for a, b in zip([0.619, 0.3809], self.dt.predict_proba(np.array([2.5, 2.2, 0.3]))):
            self.assertAlmostEquals(a, b, places=2)
        for a, b in zip([0.1052, 0.8947], self.dt.predict_proba(np.array([2.5, 3.9, 1.3]))):
            self.assertAlmostEquals(a, b, places=2)

    def test_feature_importances(self):
        expected_value = [0.297, 0.428, 0.273]
        returned_value = self.dt.feature_importances()

        for a, b in zip(returned_value, expected_value):
            self.assertAlmostEqual(a, b, places=2)

    def test_total_nodes(self):
        expected_value = 7
        returned_value = self.dt.total_nodes()
        self.assertEqual(expected_value, returned_value)

    def test_total_splits(self):
        expected_value = 3
        returned_value = self.dt.total_splits()
        self.assertEqual(expected_value, returned_value)

    def test_total_leaves(self):
        expected_value = 4
        returned_value = self.dt.total_leaves()
        self.assertEqual(expected_value, returned_value)


class DecisionForkNumericalTest(TestCase):
    def setUp(self):
        self.df = DecisionForkNumerical([70, 95], 1, 0, 0.3, 1.5)
        self.df.left_branch = 1
        self.df.right_branch = 2

    def tearDown(self):
        pass

    def test_result_branch(self):
        x = [1.6, 1.8, 2.1]
        expected_value = 2
        returned_value = self.df.result_branch(x)

        self.assertEqual(expected_value, returned_value)


class DecisionForkCategoricalTest(TestCase):
    def setUp(self):
        self.df = DecisionForkCategorical([60, 85], 1, 1, 0.25, 'A')
        self.df.left_branch = 1
        self.df.right_branch = 2

    def tearDown(self):
        pass

    def test_result_branch(self):
        x = ['A', 'B', 'B']
        expected_value = 2
        returned_value = self.df.result_branch(x)

        self.assertEqual(expected_value, returned_value)
