"""Implementation of NN layers using autograd."""

from __future__ import annotations
from typing import Collection
import numpy as np
from src.autograd import Tensor

class Layer():
    """Base class for all layers."""

    def __call__(self, x: Tensor) -> Tensor:
        """Forward propagation behavior."""
        raise NotImplementedError("Override me!")

    def get_params(self,) -> Collection[Tensor]:
        """Return set of model params."""
        raise NotImplementedError("Override me!")


class Linear(Layer):
    """Implementation of a linear layer."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        self.weights = Tensor(
            np.random.uniform(low = -1, high=1, size=(in_dim, out_dim)))
        self.biases = Tensor(
            np.random.uniform(low = -1, high=1, size=(out_dim, 1)))

    def __call__(self, x: Tensor) -> Tensor:
        return self.weights.T @ x + self.biases

    def get_params(self, ) -> Collection[Tensor]:
        return {self.weights, self.biases}


class ReLU(Layer):
    """Implementation of a ReLU layer."""
    def __call__(self, x: Tensor) -> Tensor:
        return x.relu

    def get_params(self, ) -> Collection[Tensor]:
        return set()


class Sigmoid(Layer):
    """Implementation of a Sigmoid layer."""
    def __call__(self, x: Tensor) -> Tensor:
        return x.sigmoid

    def get_params(self, ) -> Collection[Tensor]:
        return set()
