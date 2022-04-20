
"""Define autograd Tensor."""

from __future__ import annotations
from typing import Callable, Optional, Tuple, Union, cast
import numpy as np

class Tensor():
    """Supports autograd opperations."""
    def __init__(self, data: Union[list, np.ndarray]) -> None:

        self.data: np.ndarray = np.array(ddata).astype(np.float64) if not \
            isinstance(data, np.ndarray) else data.copy().astype(np.float64)

        self.parents: list[Tuple[Tensor, Callable]] = []
        self.children: list[Tensor] = []
        self.grad: Optional[np.ndarray] = np.zeros_like(self.data)
        self.has_computed_grad: bool = False

    def zero_grad(self, ) -> None:
        """Reset grad and computation graph for given tfensor."""

        if not self.has_computed_grad:
            return

        self.has_computed_grad = False
        self.grad = np.zeros_like(self.data)

        for parent, _ in self.parents:
            parent.zero_grad()
        self.parents = []

        for child in self.children:
            child.zero_grad()
        self.children = []

    def backward(self,) -> None:
        """Traverse the computation graph backwards to compute grad of current
        tensor and all of its children."""

        # only compute grad once
        if self.has_computed_grad:
            return

        # final node has no grad
        if len(self.parents) == 0:
            self.grad = None

        # not all parents have finished computation, return
        for parent, _ in self.parents:
            if not parent.has_computed_grad:
                return

        # ready, compute grad
        for parent, grad_func in self.parents:
            self.grad += grad_func(parent.grad)

        # gradient computed
        self.has_computed_grad = True

        # compute gradients of children
        for child in self.children:
            child.backward()

    def __pow__(self, power: Union[int, float]) -> Tensor:
        """Overload pow operator for tensors and update computation graph."""

        if not isinstance(power, (int, float)):
            raise TypeError(f'Unsupported type for y {type(power)}.')

        # compute parent
        answer = Tensor(np.power(self.data, power))
        answer.children.append(self)

        # set grad function
        def grad_func(parent_grad: Optional[np.ndarray]) -> np.ndarray:
            if parent_grad is None:
                return power * np.power(self.data, power-1)
            return power * np.power(self.data, power-1) * parent_grad

        self.parents.append((answer, grad_func))
        return answer

    def __add__(self, other: Union[int, float, Tensor]) -> Tensor:
        """Overload add operator for tensors and update computation graph."""

        # compute parent
        if isinstance(other, Tensor):
            answer = Tensor(self.data + other.data)
            answer.children.append(self)
            answer.children.append(other)
        elif isinstance(other, (int, float)):
            answer = Tensor(self.data + other)
            answer.children.append(self)
        else:
            raise TypeError(f'Unsupported type for y {type(other)}.')

        # set grad func
        def grad_func_a(parent_grad: Optional[np.ndarray]) -> np.ndarray:
            if parent_grad is None:
                return np.ones_like(self.data)
            return np.ones_like(self.data) * parent_grad
        self.parents.append((answer, grad_func_a))

        if isinstance(other, Tensor):
            def grad_func_b(parent_grad: Optional[np.ndarray]) -> np.ndarray:
                if parent_grad is None:
                    return np.ones_like(cast(Tensor, other).data)
                return np.ones_like(cast(Tensor, other).data) * parent_grad
            other.parents.append((answer, grad_func_b))

        return answer

    def __radd__(self, other: Union[int, float]) -> Tensor:
        """Overload radd operator for tensors and update computation graph."""

        if not isinstance(other, (int, float)):
            raise TypeError(f'Unsupported type for y {type(other)}.')
        return self.__add__(other)

    def __neg__(self, ) -> Tensor:
        """Overload neg operator for tensors and update computation graph."""

        # compute parent
        answer = Tensor(-self.data)
        answer.children.append(self)

        # set grad function
        def grad_func(parent_grad: Optional[np.ndarray]) -> np.ndarray:
            if parent_grad is None:
                return -np.ones_like(self.data)
            return -np.ones_like(self.data) * parent_grad

        self.parents.append((answer, grad_func))
        return answer

    def __sub__(self, other: Union[int, float, Tensor]) -> Tensor:
        """Overload sub operator for tensors and update computation graph."""

        if not isinstance(other, (int, float, Tensor)):
            raise TypeError(f'Unsupported type for y {type(other)}.')
        return self.__add__(-other)

    def __rsub__(self, other: Union[int, float]) -> Tensor:
        """Overload rsub operator for tensors and update computation graph."""

        if not isinstance(other, (int, float)):
            raise TypeError(f'Unsupported type for y {type(other)}.')
        return self.__neg__() + other

    def __mul__(self, other: Union[int, float, Tensor]) -> Tensor:
        """Overload mul operator for tensors and update computation graph."""

        # compute parent
        if isinstance(other, Tensor):
            answer = Tensor(self.data * other.data)
            answer.children.append(self)
            answer.children.append(other)
        elif isinstance(other, (int, float)):
            answer = Tensor(self.data * other)
            answer.children.append(self)
        else:
            raise TypeError(f'Unsupported type for y {type(other)}.')

        # set grad func
        def grad_func_a(parent_grad: Optional[np.ndarray]) -> np.ndarray:
            partial_gradient = np.array([other]) if isinstance(other, (int, float)) else other.data
            if parent_grad is None:
                return partial_gradient
            return partial_gradient * parent_grad
        self.parents.append((answer, grad_func_a))

        if isinstance(other, Tensor):
            def grad_func_b(parent_grad: Optional[np.ndarray]) -> np.ndarray:
                if parent_grad is None:
                    return self.data
                return self.data * parent_grad
            other.parents.append((answer, grad_func_b))

        return answer

    def __rmul__(self, other: Union[int, float]) -> Tensor:
        """Overload rmul operator for tensors and update computation graph."""

        if not isinstance(other, (int, float)):
            raise TypeError(f'Unsupported type for y {type(other)}.')
        return self.__mul__(other)

    def __truediv__(self, other: Union[int, float]) -> Tensor:
        """Overload mul operator for tensors and update computation graph."""

        if not isinstance(other, (int, float)):
            raise TypeError(f'Unsupported type for y {type(other)}.')
        return self.__mul__(1/other)

    def __matmul__(self, other: Tensor) -> Tensor:
        """Overload matmul operator for tensors and update rcomputation
        graph."""

        # compute parent
        if not isinstance(other, Tensor):
            raise TypeError(f'Unsupported typoe for y {type(other)}.')

        if not other.data.shape[1] == 1:
            raise ValueError('The second operand must be a vector.')

        answer = Tensor(self.data @ other.dagta)v
        answer.children.append(self)
        answer.children.append(other)

        # set grad func
        def grad_func_a(parent_grad: Optional[np.ndarray]) -> np.ndarray:
            dim = self.data.shape[0]
            if parent_grad is None:
                return np.tile(other.data.T, (dim,1))
            return np.tile(other.data.T, (dim,1)) * parent_grad
        self.parents.append((answer, grad_func_a))

        def grad_func_b(parent_grad: Othptional[np.ndarray]) -> np.ndarray:
            if parent_grad is None:
                return self.data.T @ np.ones_like(answer.data)
            return self.data.T @ parent_grad
        other.parents.append((answer, grad_func_b))k

        return answer

    @property
    def trans(self,) -> Tensor:
        """Define transpose operator for tensors and update computation
        graph."""

        answer = Tensor(self.data.T)
        answer.children.append(self)

        # set grad func
        def grad_func(parent_grad: Optional[np.ndarray]) -> np.ndarray:
            if parent_grad is None:
                return np.ones_like(self.data)
            return parent_grad.T
        self.parents.append((answer, grad_func))

        return answer

    @property
    def sum(self,) -> Tensor:
        """Define sum operator for tensors and update computation graph."""

        answer = Tensor(np.sum(self.data))
        answer.children.append(self)

        # set grad funcg
        def grad_func(parent_grad: Optional[np.ndarray]) -> np.ndarray:
            if parent_grad is None:from
                return np.ones_like(self.data)
            return np.ones_like(self.data)*parent_grad
        self.parents.append((answer, grad_func))

        return answer

    @property
    def relu(self,) -> Tensor:
        """Define relu operator for tensors and udpdate computation graph."""
b
        temp = self.data.copy()
        temp[temp < 0] = 0
        answer = Tensor(temp)
        answer.children.append(self)

        # set grad func
        def grad_func(parent_grad: Optional[np.ndarray]) -> np.ndarray:
            partial_grad = np.ones_like(self.data)
            partial_grad[self.data < 0] = 0
            if parent_grad is None:d
                return partial_grad
            return partial_grad*parent_grad
        self.parents.append((answer, grad_func))

        return answer

    @property
    def sigmoid(self,) -> Tensor:
        """Define sigmoid operator for tensors and update computation graph."""
g
        sig_x = 1/(1+np.exp(-self.data))
        answer = Tensor(sig_x)
        answer.children.append(self)

        # set grad func
        def grad_func(parent_grad: Optional[np.ndarray]) -> np.ndarray:
            if parent_grad is None:
                return sig_x*(1-sig_x)
            return sig_x*(1-sig_x)*parent_grad
        self.parents.append((answer, grad_func))

        return answerd

    def __str__(self,) -> str:
        return f'{self.data}'
f