# https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
"""
This landmark paper introduced AlexNet, which achieved unprecedented performance on the ImageNet challenge and popularized the use of deep convolutional neural networks. It also utilized GPUs for training large-scale neural networks.

model:
- eight learned layers
    - five convolutional
    - three fully-connected
- ReLU Nonlinearity
- LocalResponseNorm after the ReLU activation in certain layers.
    - Specifically, it takes the output of each neuron and divides it by a term that depends on the outputs of neighboring neurons.
    - This term is calculated using the sum of squared outputs of nearby neurons within the same layer.
- MaxPooling
- Dropout
    - We use dropout in the first two fully-connected layers of Figure

Training:
- stochastic gradient descent
- batch size of 128 examples
- momentum of 0.9
- weight decay of 0.0005 
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from jonigrad.layers import Conv, ReLU, Linear, LRNorm, MaxPool, Dropout,CrossEntropyLoss, Flatten
from jonigrad.utils import load_mnist, compute_accuracy
from alexnet import AlexNet

BATCH_SIZE = 32
ITERS = 100
LR = 0.001
g = np.random.default_rng()  # create a random generator

def main():
    train_X, train_y, test_X, test_y = load_mnist(flatten=False)

    print("Initializing the alexnet")
    
    joni_loss_f = CrossEntropyLoss()

    train_losses = []
    alexnet = AlexNet()

    print("Starting training")
    from tqdm import tqdm
    pbar = tqdm(range(ITERS), desc="Training Progress")

    alexnet.train()
    for i in pbar:
        ix = g.integers(low=0, high=train_X.shape[0], size=BATCH_SIZE)
        Xb, Yb = train_X[ix], train_y[ix]

        out = alexnet(Xb)
   
        loss = joni_loss_f(out, Yb)
        alexnet.zero_grad()

        dL_dy = joni_loss_f.backward()
        alexnet.backward(dL_dy)
        alexnet.step(LR)

        train_losses.append(loss.item())
        pbar.set_postfix({'train_loss': loss.item()})

    # for layer in alexnet:
    #     Xb = layer.eval()
    alexnet.eval()
    accuracy = compute_accuracy(alexnet, test_X[:(ITERS//5)*BATCH_SIZE], test_y[:(ITERS//5)*BATCH_SIZE])
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    plt.plot(range(ITERS), train_losses, label='Training Loss')
    # plt.plot(test_iterations, test_losses, label='Test Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()




