import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt

from jonigrad.layers import Linear, ReLU, CrossEntropyLoss, Conv, Module, Flatten
from jonigrad.utils import load_mnist

BATCH_SIZE = 32
ITERS = 100
LR = 0.001
g = np.random.default_rng()  # create a random generator

comparison_template = """
Performance Comparison
----------------------------------------------------------------
|     Layer      | Custom Time (s)      | Torch Time (s)       |
|----------------|----------------------|----------------------|
| Convolutional  | {:<20} | {:<20} |
| MLP            | {:<20} | {:<20} |
----------------------------------------------------------------
"""


class ConvModel(Module):
    def __init__(self):
        self.conv1 = Conv(1, 3, 3)
        self.conv2 = Conv(3, 1, 3)
        self.fc1 = Linear(576, 10)
        self.relu1 = ReLU()
        self.relu2 = ReLU()
        self.flatten = Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.flatten(x)
        y = self.fc1(x)
        return y

    def backward(self, dL_dy):
        dL_dx = self.fc1.backward(dL_dy)
        dL_dx = self.flatten.backward(dL_dx)
        dL_dx = self.relu2.backward(dL_dx)
        dL_dx = self.conv2.backward(dL_dx)
        dL_dx = self.relu1.backward(dL_dx)
        dL_dx = self.conv1.backward(dL_dx)
        return dL_dx


class MLP(Module):
    def __init__(self):
        self.fc1 = Linear(28 * 28, 256)
        self.relu = ReLU()
        self.fc2 = Linear(256, 10)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        y = self.fc2(x)
        return y

    def backward(self, dL_dy):
        dL_dx = self.fc2.backward(dL_dy)
        dL_dx = self.relu.backward(dL_dx)
        dL_dx = self.fc1.backward(dL_dx)
        return dL_dx


def custom_conv_test(Xb, Yb):
    model = ConvModel()
    loss_f = CrossEntropyLoss()

    start_time = time.time()

    model.train()
    for i in range(ITERS):
        x = Xb
        out = model(x)
        loss = loss_f(out, Yb)
        model.zero_grad()
        dL_dy = loss_f.backward()
        model.backward(dL_dy)
        model.step(LR)

    end_time = time.time()

    return end_time - start_time


def torch_conv_test(Xb, Yb):
    # Define the model
    model = nn.Sequential(
        nn.Conv2d(1, 3, kernel_size=3, padding=0, bias=False),
        nn.ReLU(),
        nn.Conv2d(3, 1, kernel_size=3, padding=0, bias=False),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(24 * 24, 10),
    )

    # Define the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR)

    Xb = torch.tensor(Xb)
    Yb = torch.tensor(Yb)

    start_time = time.time()
    for i in range(ITERS):
        out = model(Xb)
        loss = loss_fn(out, Yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    end_time = time.time()

    return end_time - start_time


def custom_mlp_test(Xb, Yb):

    model = MLP()
    loss_f = CrossEntropyLoss()

    start_time = time.time()

    model.train()
    for i in range(ITERS):
        x = Xb
        out = model(x)
        loss = loss_f(out, Yb)
        model.zero_grad()
        dL_dy = loss_f.backward()
        model.backward(dL_dy)
        model.step(LR)

    end_time = time.time()

    return end_time - start_time


def torch_mlp_test(Xb, Yb):
    # Define the model

    model = nn.Sequential(
        nn.Linear(Xb.shape[1], 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR)

    Xb = torch.tensor(Xb)
    Yb = torch.tensor(Yb)

    start_time = time.time()
    for i in range(ITERS):
        out = model(Xb)
        loss = loss_fn(out, Yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    end_time = time.time()

    return end_time - start_time


def main():
    train_X, train_y, test_X, test_y = load_mnist(flatten=False)
    ix = g.integers(low=0, high=train_X.shape[0], size=BATCH_SIZE)
    Xb, Yb = train_X[ix], train_y[ix]
    print("Running Conv tests")
    custom_dt_conv = custom_conv_test(Xb, Yb)
    torch_dt_conv = torch_conv_test(Xb, Yb)

    Xb = Xb.reshape(BATCH_SIZE, -1)
    print("Running MLP tests")
    custom_dt_mlp = custom_mlp_test(Xb, Yb)
    torch_dt_mlp = torch_mlp_test(Xb, Yb)

    print(
        comparison_template.format(
            custom_dt_conv, torch_dt_conv, custom_dt_mlp, torch_dt_mlp
        )
    )


if __name__ == "__main__":
    main()
