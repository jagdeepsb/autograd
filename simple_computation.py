from src.autograd import Tensor
import numpy as np

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