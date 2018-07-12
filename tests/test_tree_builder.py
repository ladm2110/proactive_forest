from unittest import TestCase
from proactive_forest.tree import DecisionTree, DecisionLeaf, DecisionForkCategorical, DecisionForkNumerical
from proactive_forest.tree_builder import TreeBuilder
from proactive_forest.metrics import GiniCriterion
from proactive_forest.splits import BestSplitChooser
from proactive_forest.feature_selection import AllFeatureSelection
import numpy as np


class TreeBuilderInitializationTest(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_max_depth_exception_negative_value(self):
        with self.assertRaises(ValueError):
            self.tree_builder = TreeBuilder(split_criterion=GiniCriterion(), split_chooser=BestSplitChooser(),
                                            feature_selection=AllFeatureSelection(), max_depth=-1)

    def test_max_depth_none_value(self):
        self.tree_builder = TreeBuilder(split_criterion=GiniCriterion(), split_chooser=BestSplitChooser(),
                                        feature_selection=AllFeatureSelection(), max_depth=None)
        self.assertIsNone(self.tree_builder.max_depth)

    def test_max_depth_positive_value(self):
        self.tree_builder = TreeBuilder(split_criterion=GiniCriterion(), split_chooser=BestSplitChooser(),
                                        feature_selection=AllFeatureSelection(), max_depth=1)
        self.assertEqual(self.tree_builder.max_depth, 1)

    def test_min_samples_leaf_exception_negative_value(self):
        with self.assertRaises(ValueError):
            self.tree_builder = TreeBuilder(split_criterion=GiniCriterion(), split_chooser=BestSplitChooser(),
                                            feature_selection=AllFeatureSelection(), min_samples_leaf=-1)

    def test_min_samples_leaf_exception_none_value(self):
        with self.assertRaises(ValueError):
            self.tree_builder = TreeBuilder(split_criterion=GiniCriterion(), split_chooser=BestSplitChooser(),
                                            feature_selection=AllFeatureSelection(), min_samples_leaf=None)

    def test_min_samples_leaf_positive_value(self):
        self.tree_builder = TreeBuilder(split_criterion=GiniCriterion(), split_chooser=BestSplitChooser(),
                                        feature_selection=AllFeatureSelection(), min_samples_leaf=1)
        self.assertEqual(self.tree_builder.min_samples_leaf, 1)

    def test_min_samples_split_exception_none_value(self):
        with self.assertRaises(ValueError):
            self.tree_builder = TreeBuilder(split_criterion=GiniCriterion(), split_chooser=BestSplitChooser(),
                                            feature_selection=AllFeatureSelection(), min_samples_split=None)

    def test_min_samples_split_exception_less_than_two_instances(self):
        with self.assertRaises(ValueError):
            self.tree_builder = TreeBuilder(split_criterion=GiniCriterion(), split_chooser=BestSplitChooser(),
                                            feature_selection=AllFeatureSelection(), min_samples_split=1)

    def test_min_samples_split_positive_value_greater_than_one(self):
        self.tree_builder = TreeBuilder(split_criterion=GiniCriterion(), split_chooser=BestSplitChooser(),
                                        feature_selection=AllFeatureSelection(), min_samples_split=2)
        self.assertEqual(self.tree_builder.min_samples_split, 2)

    def test_min_gain_split_exception_none_value(self):
        with self.assertRaises(ValueError):
            self.tree_builder = TreeBuilder(split_criterion=GiniCriterion(), split_chooser=BestSplitChooser(),
                                            feature_selection=AllFeatureSelection(), min_gain_split=None)

    def test_min_gain_split_exception_negative_value(self):
        with self.assertRaises(ValueError):
            self.tree_builder = TreeBuilder(split_criterion=GiniCriterion(), split_chooser=BestSplitChooser(),
                                            feature_selection=AllFeatureSelection(), min_gain_split=-1)

    def test_min_gain_split_non_negative_value(self):
        self.tree_builder = TreeBuilder(split_criterion=GiniCriterion(), split_chooser=BestSplitChooser(),
                                        feature_selection=AllFeatureSelection(), min_gain_split=1)
        self.assertEqual(self.tree_builder.min_gain_split, 1)

    def test_split_chooser_exception_none_value(self):
        with self.assertRaises(ValueError):
            self.tree_builder = TreeBuilder(split_criterion=GiniCriterion(),
                                            feature_selection=AllFeatureSelection(), split_chooser=None)

    #def test_split_chooser_exception_non_admissible_value(self):
     #   with self.assertRaises(ValueError):
      #      self.tree_builder = TreeBuilder(split_criterion=GiniCriterion(),
       #                                     feature_selection=AllFeatureSelection(), split_chooser='non')

    def test_split_chooser_admissible_value(self):
        self.tree_builder = TreeBuilder(split_criterion=GiniCriterion(), split_chooser=BestSplitChooser(),
                                        feature_selection=AllFeatureSelection())
        self.assertIsInstance(self.tree_builder.split_chooser, BestSplitChooser)

    def test_split_criterion_exception_none_value(self):
        with self.assertRaises(ValueError):
            self.tree_builder = TreeBuilder(split_criterion=None, split_chooser=BestSplitChooser(),
                                            feature_selection=AllFeatureSelection())

    #def test_split_criterion_exception_non_admissible_value(self):
     #   with self.assertRaises(ValueError):
      #      self.decision_tree = DecisionTreeClassifier(split_criterion='non')

    def test_split_criterion_admissible_value(self):
        self.tree_builder = TreeBuilder(split_criterion=GiniCriterion(),  split_chooser=BestSplitChooser(),
                                        feature_selection=AllFeatureSelection())
        self.assertIsInstance(self.tree_builder.split_criterion, GiniCriterion)

    def test_feature_selection_exception_none_value(self):
        with self.assertRaises(ValueError):
            self.tree_builder = TreeBuilder(split_criterion=GiniCriterion(),  split_chooser=BestSplitChooser(),
                                            feature_selection=None)

    def test_feature_selection_admissible_value(self):
        self.tree_builder = TreeBuilder(split_criterion=GiniCriterion(),  split_chooser=BestSplitChooser(),
                                        feature_selection=AllFeatureSelection())
        self.assertIsInstance(self.tree_builder.feature_selection, AllFeatureSelection)


class TreeBuilderTest(TestCase):
    def setUp(self):
        self.tree_builder = TreeBuilder(split_criterion=GiniCriterion(),
                                        split_chooser=BestSplitChooser(),
                                        feature_selection=AllFeatureSelection())

    def tearDown(self):
        pass

    def test_build_tree_only_root(self):
        n_classes = 1
        x = np.array([1, 1]).reshape((2, 1))
        y = np.array([0, 0])
        returned = self.tree_builder.build_tree(x, y, n_classes)
        expected_length = 1
        expected_root_samples = [2]

        self.assertEqual(len(returned.nodes), expected_length)
        self.assertEqual(returned.nodes[returned.last_node_id].samples, expected_root_samples)
        self.assertIsInstance(returned.nodes[returned.last_node_id], DecisionLeaf)

    def test_build_tree(self):
        n_classes = 2
        x = np.array(['A', 'B', 'A', 'B', 'B', 'C', 'A', 'C', 'B']).reshape((3, 3))
        y = np.array([1, 1, 0])
        expected_length = 3
        expected_root_value = 'B'
        expected_root_feature_id = 1

        returned = self.tree_builder.build_tree(x, y, n_classes)
        self.assertEqual(len(returned.nodes), expected_length)
        self.assertEqual(returned.nodes[0].value, expected_root_value)
        self.assertEqual(returned.nodes[0].feature_id, expected_root_feature_id)
        self.assertEqual([returned.nodes[1].result, returned.nodes[2].result], [1, 0])
        self.assertIsInstance(returned.nodes[0], DecisionForkCategorical)

    def test_build_tree_recursive_all_same_class_two_classes(self):
        x = np.array(['A', 'B', 'A', 'B', 'B', 'C', 'A', 'C', 'B']).reshape((3, 3))
        y = np.array([1, 1, 1])
        self.tree_builder._n_classes = 2
        tree = DecisionTree(n_features=3)
        tree.last_node_id = tree.root()
        self.tree_builder._build_tree_recursive(tree, tree.last_node_id, x, y, depth=1)
        expected_length = 1
        expected_root_samples = [0, 3]

        self.assertEqual(len(tree.nodes), expected_length)
        self.assertIsInstance(tree.nodes[tree.last_node_id], DecisionLeaf)
        self.assertEqual(tree.nodes[tree.last_node_id].samples, expected_root_samples)

    def test_build_tree_recursive_min_samples_split(self):
        x = np.array(['A', 'B', 'A', 'B', 'B', 'C', 'A', 'C', 'B']).reshape((3, 3))
        y = np.array([1, 1, 0])
        self.tree_builder._n_classes = 2
        self.tree_builder._min_samples_split = 4
        tree = DecisionTree(n_features=3)
        tree.last_node_id = tree.root()
        self.tree_builder._build_tree_recursive(tree, tree.last_node_id, x, y, depth=1)
        expected_length = 1
        expected_root_samples = [1, 2]

        self.assertEqual(len(tree.nodes), expected_length)
        self.assertIsInstance(tree.nodes[tree.last_node_id], DecisionLeaf)
        self.assertEqual(tree.nodes[tree.last_node_id].samples, expected_root_samples)

    def test_build_tree_recursive_max_depth(self):
        x = np.array(['A', 'B', 'A', 'B', 'B', 'C', 'A', 'C', 'B']).reshape((3, 3))
        y = np.array([1, 1, 0])
        self.tree_builder._n_classes = 2
        self.tree_builder._max_depth = 0
        tree = DecisionTree(n_features=3)
        tree.last_node_id = tree.root()
        self.tree_builder._build_tree_recursive(tree, tree.last_node_id, x, y, depth=1)
        expected_length = 1
        expected_root_samples = [1, 2]

        self.assertEqual(len(tree.nodes), expected_length)
        self.assertIsInstance(tree.nodes[tree.last_node_id], DecisionLeaf)
        self.assertEqual(tree.nodes[tree.last_node_id].samples, expected_root_samples)

    def test_build_tree_recursive(self):
        x = np.array([0, 1, 0, 1, 1, 2, 0, 2, 1]).reshape((3, 3))
        y = np.array([1, 1, 0])
        self.tree_builder._n_classes = 2
        tree = DecisionTree(n_features=3)
        tree.last_node_id = tree.root()
        self.tree_builder._build_tree_recursive(tree, tree.last_node_id, x, y, depth=1)
        expected_length = 3
        expected_root_feature_id = 1
        expected_root_value = 1.5

        self.assertEqual(len(tree.nodes), expected_length)
        self.assertIsInstance(tree.nodes[0], DecisionForkNumerical)
        self.assertEqual(tree.nodes[0].feature_id, expected_root_feature_id)
        self.assertEqual(tree.nodes[0].value, expected_root_value)
        self.assertEqual([tree.nodes[1].result, tree.nodes[2].result], [1, 0])

    def test_find_best_split_categorical(self):
        x = np.array(['A', 'B', 'A', 'B', 'B', 'C', 'A', 'C', 'B']).reshape((3, 3))
        y = np.array([1, 1, 0])
        expected_split_value = 'B'
        expected_split_feature_id = 1
        expected_split_gain = 0.44
        returned_split = self.tree_builder._find_split(x, y, 3)

        self.assertEqual(returned_split.value, expected_split_value)
        self.assertEqual(returned_split.feature_id, expected_split_feature_id)
        self.assertAlmostEqual(returned_split.gain, expected_split_gain, places=2)

    def test_find_best_split_numerical(self):
        x = np.array([0, 1, 0, 1, 1, 2, 0, 2, 1]).reshape((3, 3))
        y = np.array([1, 1, 0])
        expected_split_value = 1.5
        expected_split_feature_id = 1
        expected_split_gain = 0.44
        returned_split = self.tree_builder._find_split(x, y, 3)

        self.assertEqual(returned_split.value, expected_split_value)
        self.assertEqual(returned_split.feature_id, expected_split_feature_id)
        self.assertAlmostEqual(returned_split.gain, expected_split_gain, places=2)

    def test_find_best_split_without_examples(self):
        x = np.array([]).reshape((0, 0))
        y = np.array([])
        expected_split = None
        returned_split = self.tree_builder._find_split(x, y, 0)

        self.assertEqual(returned_split, expected_split)
