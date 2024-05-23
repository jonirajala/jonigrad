import time
import numpy as np
import matplotlib.pyplot as plt

from jonigrad.layers import Linear, ReLU, CrossEntropyLoss, Conv, Flatten
from jonigrad.utils import load_mnist, compute_accuracy

BATCH_SIZE = 32
ITERS = 300
LR = 0.001
g = np.random.default_rng()  # create a random generator

def main():
    train_X, train_y, test_X, test_y = load_mnist(flatten=False)

    jonigrad_model = [Conv(1, 3, 3), ReLU(), Conv(3, 1, 3), ReLU(), Flatten(), Linear(576, 10)]
    joni_loss_f = CrossEntropyLoss()

    train_losses = []
    test_losses = []
    test_iterations = []

    start_time = time.time()

    for i in range(ITERS):
        
        ix = g.integers(low=0, high=train_X.shape[0], size=BATCH_SIZE)
        Xb, Yb = train_X[ix], train_y[ix]
        for layer in jonigrad_model:
            Xb = layer(Xb)
        out = Xb

        loss = joni_loss_f(out, Yb)
        
        for layer in jonigrad_model:
            layer.zero_grad()

        dL_dy = joni_loss_f.backward()
        for layer in reversed(jonigrad_model):
            dL_dy = layer.backward(dL_dy)
        for layer in jonigrad_model:
            layer.step(LR)
        train_losses.append(loss.item() / BATCH_SIZE)
        if i % 10 == 0:
            print(i)

    end_time = time.time()
    accuracy = compute_accuracy(jonigrad_model, test_X, test_y)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    print(f"Execution time: {end_time - start_time} seconds")

    plt.plot(range(ITERS), train_losses, label='Training Loss')
    plt.plot(test_iterations, test_losses, label='Test Loss')
    plt.show()

if __name__ == '__main__':
    main()