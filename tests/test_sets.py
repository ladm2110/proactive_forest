from unittest import TestCase
from proactive_forest.sets import SimpleSet, BaggingSet


class SimpleSetTest(TestCase):

    def setUp(self):
        self.set_generator = SimpleSet(n_instances=10)

    def tearDown(self):
        pass

    def test_correct_generation_ids(self):
        result = self.set_generator.training_ids()

        self.assertIn(0, result)
        self.assertIn(1, result)
        self.assertIn(2, result)
        self.assertIn(3, result)
        self.assertIn(4, result)
        self.assertIn(5, result)
        self.assertIn(6, result)
        self.assertIn(7, result)
        self.assertIn(8, result)
        self.assertIn(9, result)

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
