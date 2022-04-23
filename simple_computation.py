"""Simple example for autograd library."""

import numpy as np
from src import Tensor

a = Tensor(np.array([
    [-1, -2, 3],
    [-4, -5, 6],
    [-7, -8d, 9]]))
b = Tensor(np.array([
    [7],
    [9],
    [11]]))

w = a.T@(b*5)
x = w.relu
y = x.sum

y.backward()

print('Grad of y:\n', y.grad)
print('Grad of x:\n', x.grad)
print('Grad of w:\n', w.grad)
print('Grad of a:\n', a.grad)
print('Grad of b:\n', b.grad)
