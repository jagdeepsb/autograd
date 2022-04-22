"""Example training script on mnist."""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src import MNIST, MLP, MSE, SGD

train_dset = MNIST(os.path.join('data', 'mnist_training.mat'))
test_dset = MNIST(os.path.join('data', 'mnist_test.mat'))

# Data visualization
samples = np.random.choice(len(train_dset.x), (5*5,), replace=False)
fig, ax = plt.subplots(nrows=5, ncols=5, figsize=(7,7))

for i, (x,y) in enumerate(train_dset.get_data()):
    if i == 25:
        break
    col = i%5
    row = i//5
    ax[row][col].imshow(x.data.reshape((28, 28)).T)
plt.show()

# Train
def eval_model(mmodel: MLP, dset: MNIST) -> float:
    """Evaluate the accuracy of a model on a dataset."""
    eval_acc = 0
    eval_count = 0
    for eval_x, eval_y in dset.get_labeled_data():
        eval_yp = mmodel(eval_x).data.flatten()
        eval_yp_label = np.argmax(eval_yp)
        if (eval_y-1) == eval_yp_label:
            eval_acc += 1
        eval_count += 1
    return eval_acc/eval_count

model = MLP(28**2, 10)
loss_func = MSE()
optim = SGD(model.get_params(), lr=1e-2)

num_epochs = 50

for i in (pbar := tqdm(range(num_epochs))):

    avg_loss = 0
    count = 0

    for x, y in train_dset.get_data():
        optim.zero_grad()
        yp = model(x)
        loss = loss_func(yp, y)
        loss.backward()
        avg_loss += loss.data.flatten()[0]
        count += 1
        optim.step()

    print_loss = round(10000*avg_loss/count)/10000
    print_acc = round(1000*eval_model(model, train_dset))/1000
    pbar.set_description(f'loss = {print_loss}, acc = {print_acc}')


# Evaluate model accuracy
print_acc = round(1000*eval_model(model, test_dset))/1000
print(f'Test acc = {print_acc}')
