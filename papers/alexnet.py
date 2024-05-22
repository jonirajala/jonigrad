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

from keras.datasets import mnist
import time
import numpy as np
import matplotlib.pyplot as plt

from jonigrad.layers import Conv, ReLU, Linear, LRNorm, MaxPool, Dropout,CrossEntropyLoss, Flatten

BATCH_SIZE = 32
ITERS = 100
LR = 0.00001
g = np.random.default_rng()  # create a random generator

def load_data():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    WIDTH, HEIGHT = train_X.shape[1], train_X.shape[2]
    train_X = train_X.reshape(-1, 1,  HEIGHT, WIDTH).astype(np.float32) / 255.0
    test_X = test_X.reshape(-1, 1, HEIGHT, WIDTH).astype(np.float32) / 255.0

    return train_X, train_y, test_X, test_y


def main():
    train_X, train_y, test_X, test_y = load_data()

    print("Initializing the alexnet")
    alexnet = []
    alexnet.append(Conv(in_channels=1, out_channels=3, kernel_size=11, stride=1, padding=2))
    alexnet.append(ReLU())
    alexnet.append(LRNorm(size=5, alpha=1e-4, beta=0.75, k=2))
    alexnet.append(MaxPool(kernel_size=3, stride=2))

    # Second Convolutional Layer
    alexnet.append(Conv(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=2))
    alexnet.append(ReLU())
    alexnet.append(LRNorm(size=5, alpha=1e-4, beta=0.75, k=2))
    alexnet.append(MaxPool(kernel_size=3, stride=2))

    # Third Convolutional Layer
    alexnet.append(Conv(in_channels=6, out_channels=9, kernel_size=3, stride=1, padding=1))
    alexnet.append(ReLU())

    # Fourth Convolutional Layer
    alexnet.append(Conv(in_channels=9, out_channels=9, kernel_size=3, stride=1, padding=1))
    alexnet.append(ReLU())

    # Fifth Convolutional Layer
    alexnet.append(Conv(in_channels=9, out_channels=6, kernel_size=3, stride=1, padding=1))
    alexnet.append(ReLU())

    # Max Pooling Layer
    alexnet.append(MaxPool(kernel_size=3, stride=1))

    # First Fully Connected Layer
    alexnet.append(Flatten())
    alexnet.append(Linear(in_features=54, out_features=18))
    alexnet.append(ReLU())
    alexnet.append(Dropout(p=0.5))

    # Second Fully Connected Layer
    alexnet.append(Linear(in_features=18, out_features=18))
    alexnet.append(ReLU())
    alexnet.append(Dropout(p=0.5))

    # Output Layer
    alexnet.append(Linear(in_features=18, out_features=10))

    joni_loss_f = CrossEntropyLoss()

    train_losses = []
    test_losses = []
    test_iterations = []

    print("Starting training")
    for i in range(ITERS):
        
        ix = g.integers(low=0, high=train_X.shape[0], size=BATCH_SIZE)
        Xb, Yb = train_X[ix], train_y[ix]

        for layer in alexnet:
            Xb = layer(Xb)
        out = Xb

        # Xb = alexnet[0](Xb)
        # Xb = alexnet[1](Xb)
        # Xb = alexnet[2](Xb)
        # Xb = alexnet[3](Xb)
        # Xb = Xb.reshape(-1, 24*24)
        # out = alexnet[4](Xb)
   
        loss = joni_loss_f(out, Yb)
        
        for layer in alexnet:
            layer.zero_grad()

        dL_dy = joni_loss_f.backward()
        for layer in reversed(alexnet):
            dL_dy = layer.backward(dL_dy)
        for layer in alexnet:
            layer.step(LR)
        train_losses.append(loss.item() / BATCH_SIZE)
        if i % 10 == 0:
            print(i)

    end_time = time.time()

    plt.plot(range(ITERS), train_losses, label='Training Loss')
    plt.plot(test_iterations, test_losses, label='Test Loss')
    plt.show()

if __name__ == '__main__':
    main()




