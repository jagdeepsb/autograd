# Autograd
A from scratch implementation of matrix autograd along with a simple proof of concept implementation of a deep neural network that achieves 82% accuracy on MNIST handwritten digit classification.

## Getting Started

Clone the repo:

```shell
git clone git@github.com:jagdeepsb/autograd.git
```

Either install Python dependencies with conda:

```shell
conda env create -f environment.yml
conda activate autograd
```

or with pip:

```shell
pip install -r requirements.txt
```

## Examples

To see a simple example of matrix autograd, you can run:

```shell
python simple_computation.py
```

For a simple training script of the MNIST digit classifier, run:

```shell
python train_example.py
```

## Supported Operations

The autograd `Tensor` class supports many common operations. Let `a` and `b` be two tensors of the same shape. Let `c` be a constant (float). This implementation supports the following:

### Addition
- `result = a + b`
- `result = a + c`
- `result = c + a`
  
### Subtraction and Negation
- `result = a - b`
- `result = a - c`
- `result = c - a`
- `result = -a`
  
### Multiplication
- `result = a * b`
- `result = a * c`
- `result = c * a`  
  
### Division
- `result = a / c`
  
### Power
- `result = a ** c`
  
### Matrix-vector Multiplication
Let `W` be a `Tensor` with shape `(m x n)` and `x` be a `Tensor` with shape `(n x 1)`:
- `result = W @ x`
  
### Misc
- Transpose: `result = a.T`
- Sum: `result = a.sum`
- Sigmoid: `result = a.sigmoid`
- ReLU: `result = a.relu`

## Contributing

Feel free to contribute by making a pull request to the repo! You will need to install the following pip packages: `mypy`, `pytest`, `pytest-pylint`, `docformatter`, `yapf` and make sure you contributions pass the following styling guidelines:

```shell
mypy . --config-file mypy.ini
python -m py.test  --pylint -m pylint --pylint-rcfile=.cbt_pylintrc
```

Additionally, run the following to format your code:

```shell
docformatter -i -r .
yapf -r --style .style.yapf .
```