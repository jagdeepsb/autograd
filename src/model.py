"""Implementation of simple NN using autograd."""

from __future__ import annotations
from typing import Collection
from src.autograd import Tensor
from src.layers import Linear, Sigmoid

class MLP():
    """Implementation of a simple NN model."""
    def __init__(self, in_dims: int, out_dims: int) -> None:
        self.layers = [
            Linear(in_dims, 64),
            Sigmoid(),
            Linear(64, 64),
            Sigmoid(),
            Linear(64, 64),
            Sigmoid(),
            Linear(64, 64),
            Sigmoid(),
            Linear(64, out_dims),
            Sigmoid()
        ]

    def __call__(self, x: Tensor) -> Tensor:
        """Forwards pass."""
        for layer in self.layers:
            x = layer(x)
        return x

    def get_params(self, ) -> Collection[Tensor]:
        """Return set of all params."""
        params = set()
        for layer in self.layers:
            for param in layer.get_params():
                params.add(param)
        return params
