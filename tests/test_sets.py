from unittest import TestCase
import numpy as np
from proactive_forest.sets import SimpleSet, BaggingSet


class SimpleSetTest(TestCase):

    def setUp(self):
        self.set_generator = SimpleSet(n_instances=10)

    def tearDown(self):
        pass

    def test_correct_generation_ids(self):
        result = self.set_generator.training_ids()
        for i in range(0, 10):
            self.assertIn(i, result)

    def test_correct_amount_ids(self):
        result = self.set_generator.training_ids()
        expected = 10

        self.assertEqual(len(result), expected)


class BaggingSetTest(TestCase):

    def setUp(self):
        self.set_generator = BaggingSet(n_instances=10)

    def tearDown(self):
        pass

    def test_correct_generation_ids(self):
        result = self.set_generator.training_ids()

        self.assertNotIn(10, result)
        self.assertNotIn(11, result)
        self.assertNotIn(12, result)
        self.assertNotIn(13, result)
        self.assertNotIn(-1, result)
        self.assertNotIn(-2, result)

        for r in result:
            self.assertIn(r, range(10))

    def test_correct_amount_ids(self):
        result = self.set_generator.training_ids()
        expected = 10

        self.assertEqual(len(result), expected)

    def test_oob_ids(self):
        self.set_generator._set_ids = np.array([1, 1, 4, 5, 4, 6, 9, 7, 9, 9])

        expected_ids = [0, 2, 3, 8]
        returned_ids = self.set_generator.oob_ids()

        assert returned_ids == expected_ids

