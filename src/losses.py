"""Implementation of NN losses and optimizers using autograd"""

from __future__ import annotations
from typing import Collection
from autograd import Tensor

class Loss():
    """Base class for loss function."""
    def __call__(self, predictions: Tensor, ground_truth: Tensor) -> Tensor:
        """Compute loss given predictions and ground truth."""
        raise NotImplementedError("Override me!")

class MSE(Loss):
    """Implementation of MSE loss."""
    def __call__(self, predictions: Tensor, ground_truth: Tensor) -> Tensor:
        return ((predictions - ground_truth)**2).sum

class Optimizer():
    """Base class for optimizer function."""
    def __init__(
        self, model_params: Collection[Tensor],
        lr: float = 1e-3) -> None:

        self.params: Collection[Tensor] = model_params
        self.lr: float = lr

    def zero_grad(self, ) -> None:
        """Reset all gradients."""
        for param in self.params:
            param.zero_grad()

    def step(self, ) -> None:
        """Update all the weights given the gradients."""
        raise NotImplementedError("Override me!")

class SGD(Optimizer):
    """Implementation of SGD optimizer."""
    def __init__(
        self, model_params: Collection[Tensor],
        lr: float = 1e-3) -> None:

        Optimizer.__init__(model_params, lr)

    def step(self, ) -> None:
        for param in self.params:
            if param.grad is None:
                continue
            param.data -= self.lr * param.grad
