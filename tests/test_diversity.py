from unittest import TestCase, mock
from proactive_forest.tree import DecisionTree
import numpy as np
from proactive_forest.diversity import PercentageCorrectDiversity, QStatisticDiversity


class PCDMeasureTest(TestCase):
    def setUp(self):
        self.target = np.array([0, 1, 0])

        self.instances = mock.MagicMock()
        self.instances.shape = [3, 3]
        self.instances[0] = [1, 1, 1]
        self.instances[1] = [2, 2, 2]
        self.instances[2] = [3, 3, 3]
        self.instances[3] = [4, 4, 4]

        model_1 = mock.MagicMock(spec=DecisionTree)
        model_1.predict.side_effect = [0, 1, 0]

        model_2 = mock.MagicMock(spec=DecisionTree)
        model_2.predict.side_effect = [0, 1, 0]

        model_3 = mock.MagicMock(spec=DecisionTree)
        model_3.predict.side_effect = [0, 1, 0]

        model_4 = mock.MagicMock(spec=DecisionTree)
        model_4.predict.side_effect = [0, 1, 0]

        self.ensemble = [model_1, model_2, model_3, model_4]

        model_5 = mock.MagicMock(spec=DecisionTree)
        model_5.predict.side_effect = [0, 1, 0]

        model_6 = mock.MagicMock(spec=DecisionTree)
        model_6.predict.side_effect = [1, 1, 0]

        model_7 = mock.MagicMock(spec=DecisionTree)
        model_7.predict.side_effect = [0, 1, 0]

        model_8 = mock.MagicMock(spec=DecisionTree)
        model_8.predict.side_effect = [0, 0, 0]

        self.ensemble_2 = [model_5, model_6, model_7, model_8]
        self.measure = PercentageCorrectDiversity()

    def tearDown(self):
        pass

    def test_get_measure_all_same_pred(self):
        expected_value = 0
        predicted_value = self.measure.get_measure(self.ensemble, X=self.instances, y=self.target)

        assert predicted_value == expected_value

    def test_get_measure_all_wrong_pred(self):
        expected_value = 0
        predicted_value = self.measure.get_measure(self.ensemble, X=self.instances, y=self.target)

        assert predicted_value == expected_value

    def test_get_measure_calculation(self):
        expected_value = 0.667
        predicted_value = self.measure.get_measure(self.ensemble_2, X=self.instances, y=self.target)

        self.assertAlmostEqual(expected_value, predicted_value, places=2)


class QStatDiversityTest(TestCase):
    def setUp(self):
        self.target = np.array([0, 1, 0])

        self.instances = mock.MagicMock()
        self.instances.shape = [3, 3]
        self.instances[0] = [1, 1, 1]
        self.instances[1] = [2, 2, 2]
        self.instances[2] = [3, 3, 3]
        self.instances[3] = [4, 4, 4]

        model_1 = mock.MagicMock(spec=DecisionTree)
        model_1.predict.side_effect = [0, 1, 0, 0, 1, 0, 0, 1, 0]

        model_2 = mock.MagicMock(spec=DecisionTree)
        model_2.predict.side_effect = [1, 1, 0, 1, 1, 0, 1, 1, 0]

        model_3 = mock.MagicMock(spec=DecisionTree)
        model_3.predict.side_effect = [0, 1, 0, 0, 1, 0, 0, 1, 0]

        model_4 = mock.MagicMock(spec=DecisionTree)
        model_4.predict.side_effect = [0, 0, 0, 0, 0, 0, 0, 0, 0]

        self.ensemble = [model_1, model_2, model_3, model_4]
        self.measure = QStatisticDiversity()

    def tearDown(self):
        pass

    def test_get_measure_calculation(self):
        expected_value = 0.303
        predicted_value = self.measure.get_measure(self.ensemble, X=self.instances, y=self.target)

        self.assertAlmostEqual(expected_value, predicted_value, places=2)


