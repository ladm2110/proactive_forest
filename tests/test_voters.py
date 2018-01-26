from unittest import TestCase, mock
from proactive_forest.voters import MajorityVoter
from proactive_forest.tree import DecisionTree


class MajorityVoterTest(TestCase):
    def setUp(self):

        model_1 = mock.MagicMock(spec=DecisionTree)
        model_1.predict.return_value = 1

        model_2 = mock.MagicMock(spec=DecisionTree)
        model_2.predict.return_value = 2

        model_3 = mock.MagicMock(spec=DecisionTree)
        model_3.predict.return_value = 0

        model_4 = mock.MagicMock(spec=DecisionTree)
        model_4.predict.return_value = 0

        self.ensemble = [model_1, model_2, model_3, model_4]
        self.voter = MajorityVoter(self.ensemble, n_classes=3)

    def tearDown(self):
        pass

    def test_one_correct_prediction(self):
        expected_value = 0
        predicted_value = self.voter.predict(x=None)

        assert predicted_value == expected_value

    def test_one_incorrect_prediction(self):
        expected_value = 1
        predicted_value = self.voter.predict(x=None)

        assert predicted_value != expected_value

