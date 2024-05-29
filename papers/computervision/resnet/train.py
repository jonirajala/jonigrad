"""
https://arxiv.org/pdf/1512.03385
What is a resne?
Traditional Approach
In traditional neural networks, each layer tries to learn the exact mapping from input to output.
Think of it as trying to directly learn the function that maps your input data to the correct output.
Residual Networks (ResNets)
ResNets take a different approach. Instead of trying to learn the direct mapping,
they learn the "residual" – the difference between the desired output and the input to that layer.

the concept of learning residuals in ResNets (Residual Networks) is implemented through what are known as skip connections

model
- Input Layer
    - Conv
    - Maxpool

- Multiple Residual blocks (The sets progressively increase the depth (number of channels) of the feature maps while reducing their spatial dimensions.)
    - 2 conv layers
    - batch normalization and a ReLU after each conv
    - skip connection
- Global average Pooling
- fully connected layer


Residual blocks can be divided into two main types:
- Identity Blocks: Used when the input and output dimensions are the same.
- Convolutional Blocks: Used when the input and output dimensions differ, involving a convolution operation to match dimensions.

"""

import time
import numpy as np
import matplotlib.pyplot as plt

from jonigrad.layers import (
    CrossEntropyLoss,
)
from jonigrad.utils import load_mnist, compute_accuracy
from resnet import ResNet

BATCH_SIZE = 32
ITERS = 100
LR = 0.001
g = np.random.default_rng()  # create a random generator


def main():
    train_X, train_y, test_X, test_y = load_mnist(flatten=False)

    print("Initializing the resnet")

    joni_loss_f = CrossEntropyLoss()

    train_losses = []
    resnet = ResNet()

    print("Starting training")
    from tqdm import tqdm

    pbar = tqdm(range(ITERS), desc="Training Progress")

    resnet.train()
    for i in pbar:
        ix = g.integers(low=0, high=train_X.shape[0], size=BATCH_SIZE)
        Xb, Yb = train_X[ix], train_y[ix]

        out = resnet(Xb)
        loss = joni_loss_f(out, Yb)
        resnet.zero_grad()
        dL_dy = joni_loss_f.backward()
        resnet.backward(dL_dy)
        resnet.step(LR)

        train_losses.append(loss.item())
        pbar.set_postfix({"train_loss": loss.item()})
    # for layer in resnet:
    #     Xb = layer.eval()
    resnet.eval()
    accuracy = compute_accuracy(
        resnet, test_X[: (ITERS // 5) * BATCH_SIZE], test_y[: (ITERS // 5) * BATCH_SIZE]
    )
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    plt.plot(range(ITERS), train_losses, label="Training Loss")
    # plt.plot(test_iterations, test_losses, label='Test Loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
