import numpy as np
from unittest import TestCase, mock
from proactive_forest.tree import DecisionTree
from proactive_forest.probabilites import FIProbabilityLedger


class FIProbabilityLedgerTest(TestCase):
    def setUp(self):
        self.tree = mock.MagicMock(spec=DecisionTree)
        self.tree.feature_importances.return_value = np.array([0.2, 0.2, 0.6])
        self.ledger = FIProbabilityLedger(probabilities=[0.33, 0.33, 0.33], n_features=3, alpha=0.1)

    def tearDown(self):
        pass

    def test_correctly_updating_probabilities(self):
        self.ledger.update_probabilities(self.tree, 1/100)
        result_probabilities = self.ledger.probabilities
        expected = [0.3334, 0.3334, 0.3332]

        for i, j in zip(result_probabilities, expected):
            self.assertAlmostEqual(i, j, places=4)
