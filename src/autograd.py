
from __future__ import annotations
from typing import Callable, Optional, Sequence, Tuple, Union
import numpy as np

class Tensor():
    """Supports autograd opperations"""
    def __init__(self, x: Union[Sequence, np.ndarray]) -> None:

        if isinstance(x, list):
            self.x: np.ndarray = np.array(x).astype(np.float64)
        else:
            self.x: np.ndarray = x.copy().astype(np.float64)

        self.parents: Sequence[Tuple(Tensor, Callable)] = []
        self.children: Sequence[Tensor] = []
        self.grad: Optional[np.ndarray] = np.zeros_like(self.x)
        self.has_computed_grad: bool = False

    def zero_grad(self, ) -> None:
        """Reset grad and computation graph for given tensor."""

        if self.has_computed_grad == False:
            return

        self.has_computed_grad = False
        self.grad = np.zeros_like(self.x)

        for parent, grad_func in self.parents:
            parent.zero_grad()
        self.parents = []

        for child in self.children:
            child.zero_grad()
        self.children = []

    def backward(self,) -> None:
        """Traverse the computation graph backwards to compute grad of
        current tensor and all of its children."""

        # only compute grad once
        if self.has_computed_grad:
            return

        # final node has no grad
        if len(self.parents) == 0:
            self.grad = None

        # not all parents have finished computation, return
        for parent, _ in self.parents:
            if parent.has_computed_grad == False:
                return

        # ready, compute grad
        for parent, grad_func in self.parents:
            self.grad += grad_func(parent.grad)

        # gradient computed
        self.has_computed_grad = True

        # compute gradients of children
        for child in self.children:
            child.backward()

    def __pow__(self, b: Union[int, float]) -> Tensor:
        """Overload pow operator for tensors and update computation graph."""

        if not isinstance(b, (int, float)):
            raise TypeError(f'Unsupported type for y {type(b)}.')

        # compute parent
        answer = Tensor(np.power(self.x, b))
        answer.children.append(self)

        # set grad function
        def grad_func(parent_grad):
            if parent_grad is None:
                return b * np.power(self.x, b-1)
            return b * np.power(self.x, b-1) * parent_grad
        
        self.parents.append((answer, grad_func))
        return answer

    def __add__(self, b: Union[int, float, Tensor]) -> Tensor:
        """Overload add operator for tensors and update computation graph."""

        # compute parent
        if isinstance(b, Tensor):
            answer = Tensor(self.x + b.x)
            answer.children.append(self)
            answer.children.append(b)
        elif isinstance(b, (int, float)):
            answer = Tensor(self.x + b)
            answer.children.append(self)
        else:
            raise TypeError(f'Unsupported type for y {type(b)}.')

        # set grad func
        def grad_func_a(parent_grad):
            if parent_grad is None:
                return np.ones_like(self.x)
            return np.ones_like(self.x) * parent_grad
        self.parents.append((answer, grad_func_a))

        if isinstance(b, Tensor):
            def grad_func_b(parent_grad):
                if parent_grad is None:
                    return np.ones_like(b.x)
                return np.ones_like(b.x) * parent_grad
            b.parents.append((answer, grad_func_b))

        return answer

    def __radd__(self, b: Union[int, float]) -> Tensor:
        """Overload radd operator for tensors and update computation graph."""

        if not isinstance(b, (int, float)):
            raise TypeError(f'Unsupported type for y {type(b)}.')
        return self.__add__(b)

    def __neg__(self, ) -> Tensor:
        """Overload neg operator for tensors and update computation graph."""

        # compute parent
        answer = Tensor(-self.x)
        answer.children.append(self)

        # set grad function
        def grad_func(parent_grad):
            if parent_grad is None:
                return -np.ones_like(self.x)
            return -np.ones_like(self.x) * parent_grad
        
        self.parents.append((answer, grad_func))
        return answer

    def __sub__(self, b: Union[int, float, Tensor]) -> Tensor:
        """Overload sub operator for tensors and update computation graph."""

        if not isinstance(b, (int, float, Tensor)):
            raise TypeError(f'Unsupported type for y {type(b)}.')
        return self.__add__(-b)

    def __rsub__(self, b: Union[int, float]) -> Tensor:
        """Overload rsub operator for tensors and update computation graph."""

        if not isinstance(b, (int, float)):
            raise TypeError(f'Unsupported type for y {type(b)}.')
        return self.__neg__() + b

    def __mul__(self, b: Union[int, float, Tensor]) -> Tensor:
        """Overload mul operator for tensors and update computation graph."""

        # compute parent
        if isinstance(b, Tensor):
            answer = Tensor(self.x * b.x)
            answer.children.append(self)
            answer.children.append(b)
        elif isinstance(b, (int, float)):
            answer = Tensor(self.x * b)
            answer.children.append(self)
        else:
            raise TypeError(f'Unsupported type for y {type(b)}.')

        # set grad func
        def grad_func_a(parent_grad):
            partial_gradient = b if isinstance(b, (int, float)) else b.x
            if parent_grad is None:
                return partial_gradient
            return partial_gradient * parent_grad
        self.parents.append((answer, grad_func_a))

        if isinstance(b, Tensor):
            def grad_func_b(parent_grad):
                if parent_grad is None:
                    return self.x
                return self.x * parent_grad
            b.parents.append((answer, grad_func_b))

        return answer

    def __rmul__(self, b: Union[int, float]) -> Tensor:
        """Overload rmul operator for tensors and update computation graph."""

        if not isinstance(b, (int, float)):
            raise TypeError(f'Unsupported type for y {type(b)}.')
        return self.__mul__(b)

    def __truediv__(self, b: Union[int, float]) -> Tensor:
        """Overload mul operator for tensors and update computation graph."""

        if not isinstance(b, (int, float)):
            raise TypeError(f'Unsupported type for y {type(b)}.')
        return self.__mul__(1/b)

    def __matmul__(self, b: Tensor) -> Tensor:
        """Overload matmul operator for tensors and update computation graph."""

        # compute parent
        if not isinstance(b, Tensor):
            raise TypeError(f'Unsupported type for y {type(b)}.')

        if not b.x.shape[1] == 1:
            raise ValueError(f'The second operand must be a vector.')
        
        answer = Tensor(self.x @ b.x)
        answer.children.append(self)
        answer.children.append(b)

        # set grad func
        def grad_func_a(parent_grad):
            dim = self.x.shape[0]
            if parent_grad is None:
                return np.tile(b.x.T, (dim,1))
            return np.tile(b.x.T, (dim,1)) * parent_grad
        self.parents.append((answer, grad_func_a))

        def grad_func_b(parent_grad):
            if parent_grad is None:
                return self.x.T @ np.ones_like(answer.x)
            return self.x.T @ parent_grad
        b.parents.append((answer, grad_func_b))

        return answer

    @property
    def T(self,) -> Tensor:
        """Define transpose operator for tensors and update computation graph."""

        answer = Tensor(self.x.T)
        answer.children.append(self)

        # set grad func
        def grad_func(parent_grad):
            if parent_grad is None:
                return np.ones_like(self.x)
            return parent_grad.T
        self.parents.append((answer, grad_func))

        return answer

    @property
    def sum(self,) -> Tensor:
        """Define sum operator for tensors and update computation graph."""

        answer = Tensor(np.sum(self.x))
        answer.children.append(self)

        # set grad func
        def grad_func(parent_grad):
            if parent_grad is None:
                return np.ones_like(self.x)
            return np.ones_like(self.x)*parent_grad
        self.parents.append((answer, grad_func))

        return answer

    @property
    def relu(self,) -> Tensor:
        """Define relu operator for tensors and update computation graph."""

        temp = self.x.copy()
        temp[temp < 0] = 0 
        answer = Tensor(temp)
        answer.children.append(self)

        # set grad func
        def grad_func(parent_grad):
            partial_grad = np.ones_like(self.x)
            partial_grad[self.x < 0] = 0
            if parent_grad is None:
                return partial_grad
            return partial_grad*parent_grad
        self.parents.append((answer, grad_func))

        return answer

    @property
    def sigmoid(self,) -> Tensor:
        """Define sigmoid operator for tensors and update computation graph."""

        sig_x = 1/(1+np.exp(-self.x))
        answer = Tensor(sig_x)
        answer.children.append(self)

        # set grad func
        def grad_func(parent_grad):
            if parent_grad is None:
                return sig_x*(1-sig_x)
            return sig_x*(1-sig_x)*parent_grad
        self.parents.append((answer, grad_func))

        return answer

    def __str__(self,) -> str:
        return (f'{self.x}')


if __name__ == "__main__":

    a = Tensor(np.array([
        [-1, -2, 3],
        [-4, -5, 6],
        [-7, -8, 9]]))
    b = Tensor(np.array([
        [7],
        [9],
        [11]]))

    w = a.T@b
    x = w.relu
    y = x.sum

    y.backward()

    print('Grad of y:\n', y.grad)
    print('Grad of x:\n', x.grad)
    print('Grad of w:\n', w.grad)
    print('Grad of a:\n', a.grad)
    print('Grad of b:\n', b.grad)