import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt

from jonigrad.layers import Linear, ReLU, CrossEntropyLoss, Conv
from jonigrad.utils import load_mnist

BATCH_SIZE = 32
ITERS = 100
LR = 0.001
g = np.random.default_rng()  # create a random generator

def custom_conv_test(Xb, Yb):
    joni_model = [Conv(1, 3, 3), ReLU(), Conv(3, 1, 3), ReLU(), Linear(576, 10)]
    joni_loss_f = CrossEntropyLoss()

    start_time = time.time()

    for i in range(ITERS):
        x = Xb
        x = joni_model[0](x)
        x = joni_model[1](x)
        x = joni_model[2](x)
        x = joni_model[3](x)
        x = x.reshape(-1, 24*24)
        out = joni_model[4](x)
        loss = joni_loss_f(out, Yb)
        for layer in joni_model:
            layer.zero_grad()

        dL_dy = joni_loss_f.backward()
        for layer in reversed(joni_model):
            dL_dy = layer.backward(dL_dy)
        for layer in joni_model:
            layer.step(LR)

    end_time = time.time()

    
    return end_time-start_time

def torch_conv_test(Xb, Yb):
    # Define the model
    model = nn.Sequential(
        nn.Conv2d(1, 3, kernel_size=3, padding=0, bias=False),
        nn.ReLU(),
        nn.Conv2d(3, 1, kernel_size=3, padding=0, bias=False),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(24*24, 10)
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
    
    joni_model = [Linear(Xb.shape[1], 256), ReLU(), Linear(256, 10)]
    joni_loss_f = CrossEntropyLoss()

    start_time = time.time()

    for i in range(ITERS):
        x = Xb
        for layer in joni_model:
            x = layer(x)
        out = x 
        loss = joni_loss_f(out, Yb)
        for layer in joni_model:
            layer.zero_grad()

        dL_dy = joni_loss_f.backward()
        for layer in reversed(joni_model):
            dL_dy = layer.backward(dL_dy)

        for layer in joni_model:
            layer.step(LR)

    end_time = time.time()

    return end_time-start_time

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

    print(f"Custom Conv time: {custom_dt_conv} seconds")
    print(f"Torch Conv time: {torch_dt_conv} seconds")
    print(f"Custom MLP time: {custom_dt_mlp} seconds")
    print(f"Torch MLP time: {torch_dt_mlp} seconds")

if __name__ == "__main__":
    main()
