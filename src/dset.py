"""Implementation of MNIST dataset."""

from typing import Iterable, Tuple
import scipy.io
import numpy as np
from src.autograd import Tensor

class MNIST():
    """Implementation of MNIST dataset."""

    def __init__(self, file_name: str) -> None:
        data = scipy.io.loadmat(file_name)
        self.x = data['X_train'] if 'X_train' in data else data['X_test']
        self.labels = data['y_train'].T[0] if 'y_train' in data \
            else data['y_test'].T[0]
        self.y = np.zeros(shape=(self.x.shape[0], 10))
        self.y[list(range(len(self.labels))), (self.labels-1)] = 1

        self.mean = np.mean(self.x)
        self.std = np.std(self.x)

    def get_data(self,) -> Iterable[Tuple[Tensor, Tensor]]:
        """Get sampled data."""
        samples = np.random.choice(len(self.x), (len(self.x),), replace=False)
        for sample in samples:
            x = np.array([self.x[sample]]).T
            yield (
                Tensor((x-self.mean)/self.std),
                Tensor(np.array([self.y[sample]]).T)
                )

    def get_labeled_data(self, ) -> Iterable[Tuple[Tensor, float]]:
        """Get labeled data."""
        for x,y in zip(self.x, self.labels):
            yield (Tensor(np.array([(x-self.mean)/self.std]).T), y)
