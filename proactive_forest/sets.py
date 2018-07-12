from abc import ABC, abstractmethod
import numpy as np


class SetGenerator(ABC):
    def __init__(self, n_instances):
        """
        Generates a training set for the classifiers.

        :param n_instances: <int> Amount of instances to consider.
        """
        self._n_instances = n_instances
        self._set_ids = None

    @abstractmethod
    def training_ids(self):
        pass

    @abstractmethod
    def oob_ids(self):
        pass

    def clear(self):
        """
        Clears the set ids.
        """
        self._set_ids = None


class SimpleSet(SetGenerator):
    def training_ids(self):
        """
        Generates the ids of the training instances.
        :return: <numpy array>
        """
        if self._set_ids is None:
            self._set_ids = np.array(range(self._n_instances))
        return self._set_ids

    def oob_ids(self):
        """
        Returns an empty array. No out-of-bag instances for SimpleSet.
        :return: <numpy array>
        """
        return np.array([])


class BaggingSet(SetGenerator):
    def training_ids(self):
        """
        Generates the ids of the training instances.
        :return: <numpy array>
        """
        if self._set_ids is None:
            self._set_ids = np.random.choice(self._n_instances, replace=True, size=self._n_instances)
        return self._set_ids

    def oob_ids(self):
        """
        Returns the ids for the out-of-bag set.
        :return: <numpy array>
        """
        return [i for i in range(self._n_instances) if i not in self._set_ids]
