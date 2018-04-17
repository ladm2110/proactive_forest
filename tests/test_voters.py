from unittest import TestCase, mock
from proactive_forest.voters import MajorityVoter, DistributionSummationVoter, PerformanceWeightingVoter
from proactive_forest.tree import DecisionTree


class MajorityVoterTest(TestCase):
    def setUp(self):
        model_1 = mock.MagicMock(spec=DecisionTree)
        model_1.predict.return_value = 1
        model_1.predict_proba.return_value = [0, 1, 0]

        model_2 = mock.MagicMock(spec=DecisionTree)
        model_2.predict.return_value = 2
        model_2.predict_proba.return_value = [0.4, 0, 0.6]

        model_3 = mock.MagicMock(spec=DecisionTree)
        model_3.predict.return_value = 0
        model_3.predict_proba.return_value = [0.8, 0.2, 0]

        model_4 = mock.MagicMock(spec=DecisionTree)
        model_4.predict.return_value = 0
        model_4.predict_proba.return_value = [1, 0, 0]

        self.ensemble = [model_1, model_2, model_3, model_4]
        self.voter = MajorityVoter(self.ensemble, n_classes=3)

    def tearDown(self):
        pass

    def test_correct_prediction(self):
        expected_value = 0
        predicted_value = self.voter.predict(x=None)

        assert predicted_value == expected_value

    def test_predict_proba(self):
        expected_proba = [0.55, 0.3, 0.15]
        predicted_proba = self.voter.predict_proba(x=None)

        assert predicted_proba == expected_proba


class PerformanceWeightingVoterTest(TestCase):
    def setUp(self):
        model_1 = mock.MagicMock(spec=DecisionTree)
        model_1.predict.return_value = 1
        model_1.weight = 0.9

        model_2 = mock.MagicMock(spec=DecisionTree)
        model_2.predict.return_value = 2
        model_2.weight = 0.85

        model_3 = mock.MagicMock(spec=DecisionTree)
        model_3.predict.return_value = 0
        model_3.weight = 0.6

        model_4 = mock.MagicMock(spec=DecisionTree)
        model_4.predict.return_value = 0
        model_4.weight = 0.8

        self.ensemble = [model_1, model_2, model_3, model_4]
        self.voter = PerformanceWeightingVoter(self.ensemble, n_classes=3)

    def tearDown(self):
        pass

    def test_correct_prediction(self):
        expected_value = 0
        predicted_value = self.voter.predict(x=None)

        assert predicted_value == expected_value


class DistributionSummationVoterTest(TestCase):
    def setUp(self):
        model_1 = mock.MagicMock(spec=DecisionTree)
        model_1.predict_proba.return_value = [0, 1, 0]

        model_2 = mock.MagicMock(spec=DecisionTree)
        model_2.predict_proba.return_value = [0.4, 0, 0.6]

        model_3 = mock.MagicMock(spec=DecisionTree)
        model_3.predict_proba.return_value = [0.8, 0.2, 0]

        model_4 = mock.MagicMock(spec=DecisionTree)
        model_4.predict_proba.return_value = [1, 0, 0]

        self.ensemble = [model_1, model_2, model_3, model_4]
        self.voter = DistributionSummationVoter(self.ensemble, n_classes=3)

    def tearDown(self):
        pass

    def test_correct_prediction(self):
        expected_value = 0
        predicted_value = self.voter.predict(x=None)

        assert predicted_value == expected_value

