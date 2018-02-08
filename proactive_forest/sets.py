from abc import ABC, abstractmethod
import numpy as np


class SetGenerator(ABC):
    def __init__(self, n_instances):
        self.n_instances = n_instances
        self.set_ids = None

    @abstractmethod
    def training_ids(self):
        pass

    @abstractmethod
    def oob_ids(self):
        pass

    def clear(self):
        self.set_ids = None


class SimpleSet(SetGenerator):
    def __init__(self, n_instances):
        super().__init__(n_instances)

    def training_ids(self):
        if self.set_ids is None:
            self.set_ids = np.array(range(self.n_instances))
        return self.set_ids

    def oob_ids(self):
        return np.array([])


class BaggingSet(SetGenerator):
    def training_ids(self):
        if self.set_ids is None:
            self.set_ids = np.random.choice(self.n_instances, replace=True, size=self.n_instances)
        return self.set_ids

    def oob_ids(self):
        return [i for i in range(self.n_instances) if i not in self.set_ids]
