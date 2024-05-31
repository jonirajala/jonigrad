import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt

from jonigrad.layers import (
    Linear,
    ReLU,
    CrossEntropyLoss,
    Conv,
    Module,
    Flatten,
    LSTM,
    MSELoss,
)
from jonigrad.utils import load_mnist, load_temperature_data

BATCH_SIZE = 32
ITERS = 100
LR = 0.001
g = np.random.default_rng()  # create a random generator

comparison_template = """
Performance Comparison
-------------------------------------------------------------------------------------------
|     Test        | Convolutional         | MLP                   | LSTM                  |
|-----------------|-----------------------|-----------------------|-----------------------|
| Custom Time (s) | {:<20}  | {:<20}  | {:<20}  |
| Torch Time  (s) | {:<20}  | {:<20}  | {:<20}  |
-------------------------------------------------------------------------------------------
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


class MLPModel(Module):
    def __init__(self):
        self.fc1 = Linear(28 * 28, 512)
        self.relu1 = ReLU()
        self.fc2 = Linear(512, 512)
        self.relu2 = ReLU()
        self.fc3 = Linear(512, 10)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        y = self.fc3(x)
        return y

    def backward(self, dL_dy):
        dL_dx = self.fc3.backward(dL_dy)
        dL_dx = self.relu2.backward(dL_dx)
        dL_dx = self.fc2.backward(dL_dx)
        dL_dx = self.relu1.backward(dL_dx)
        dL_dx = self.fc1.backward(dL_dx)
        return dL_dx


class LSTMModel(Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = LSTM(input_size, hidden_layer_size)
        self.linear = Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _, _ = self.lstm.forward(input_seq)
        predictions = self.linear.forward(lstm_out[:, -1])
        return predictions

    def backward(self, dL_dy):
        dL_dy = self.linear.backward(dL_dy)
        dL_dy = np.expand_dims(dL_dy, 2)
        dl_dx = self.lstm.backward(dL_dy)
        return dl_dx


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

    model = MLPModel()
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
        nn.Linear(Xb.shape[1], 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10),
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


def custom_lstm_test(Xb, Yb):
    model = LSTMModel(input_size=1, hidden_layer_size=50, output_size=1)
    loss_function = MSELoss()

    start_time = time.time()

    model.train()
    for iter in range(ITERS):
        y_pred = model.forward(Xb)
        loss = loss_function(y_pred, Yb)
        model.zero_grad()
        dL_dy = loss_function.backward()
        model.backward(dL_dy)
        model.step(LR)

    end_time = time.time()

    return end_time - start_time


def torch_lstm_test(Xb, Yb):
    class LSTMModelTorch(nn.Module):
        def __init__(self, input_size, hidden_layer_size, output_size):
            super(LSTMModelTorch, self).__init__()
            self.hidden_layer_size = hidden_layer_size
            self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
            self.linear = nn.Linear(hidden_layer_size, output_size)

        def forward(self, input_seq):
            lstm_out, _ = self.lstm(input_seq)
            predictions = self.linear(lstm_out[:, -1])
            return predictions

    model = LSTMModelTorch(input_size=1, hidden_layer_size=50, output_size=1)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    Xb = torch.tensor(Xb, dtype=torch.float32)
    Yb = torch.tensor(Yb, dtype=torch.float32)

    start_time = time.time()
    model.train()
    for iter in range(ITERS):
        optimizer.zero_grad()
        y_pred = model(Xb)
        loss = loss_function(y_pred, Yb)
        loss.backward()
        optimizer.step()

    end_time = time.time()

    return end_time - start_time


def main():
    X_train, y_train, test_X, test_y = load_mnist(flatten=False)
    ix = g.integers(low=0, high=X_train.shape[0], size=BATCH_SIZE)
    Xb, Yb = X_train[ix], y_train[ix]
    print("Running Conv tests")
    custom_dt_conv = custom_conv_test(Xb, Yb)
    torch_dt_conv = torch_conv_test(Xb, Yb)

    Xb = Xb.reshape(BATCH_SIZE, -1)
    print("Running MLP tests")
    custom_dt_mlp = custom_mlp_test(Xb, Yb)
    torch_dt_mlp = torch_mlp_test(Xb, Yb)

    seq_length = 30
    X_train, y_train, X_test, y_test, data, temperatures, scaler = (
        load_temperature_data(seq_length)
    )
    ix = g.integers(low=0, high=X_train.shape[0], size=BATCH_SIZE)
    Xb, Yb = X_train[ix], y_train[ix]
    print("Running LSTM tests")
    custom_dt_lstm = custom_lstm_test(Xb, Yb)
    torch_dt_lstm = torch_lstm_test(Xb, Yb)

    print(
        comparison_template.format(
            custom_dt_conv,
            custom_dt_mlp,
            custom_dt_lstm,
            torch_dt_conv,
            torch_dt_mlp,
            torch_dt_lstm,
        )
    )


if __name__ == "__main__":
    main()
