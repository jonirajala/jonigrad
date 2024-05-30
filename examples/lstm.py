import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

from jonigrad.layers import Module, Linear, LSTM, ReLU, MSELoss
from jonigrad.utils import load_temperature_data

ITERS = 1000
BATCH_SIZE = 16
LR = 0.001
g = np.random.default_rng()


# Define the LSTM model
class LSTMModel(Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = LSTM(input_size, hidden_layer_size)
        self.linear = Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1])
        return predictions

    def backward(self, dL_dy):
        dL_dy = self.linear.backward(dL_dy)
        dL_dy = np.expand_dims(dL_dy, 2)
        dl_dx = self.lstm.backward(dL_dy)
        return dl_dx


class LinearModel(Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super().__init__()
        self.fc1 = Linear(input_size, 256)
        self.fc2 = Linear(256, 128)
        self.fc3 = Linear(128, output_size)
        self.relu1 = ReLU()
        self.relu2 = ReLU()

    def forward(self, input_seq):
        y = self.relu1(self.fc1(input_seq))
        y = self.relu2(self.fc2(y))
        y = self.fc3(y)
        return y

    def backward(self, dL_dy):
        dL_dy = self.fc3.backward(dL_dy)
        dL_dy = self.relu2.backward(dL_dy)
        dL_dy = self.fc2.backward(dL_dy)
        dL_dy = self.relu1.backward(dL_dy)
        dl_dx = self.fc1.backward(dL_dy)
        return dl_dx


def train():
    seq_length = 30
    X_train, y_train, X_test, y_test, data, temperatures, scaler = (
        load_temperature_data(seq_length)
    )
    test_size = len(X_test) + 30
    model = LSTMModel(input_size=1, hidden_layer_size=50, output_size=1)
    loss_function = MSELoss()

    model.train()
    losses = []
    for iter in tqdm(range(ITERS)):
        ix = g.integers(low=0, high=X_train.shape[0], size=BATCH_SIZE)
        Xb, Yb = X_train[ix], y_train[ix]
        y_pred = model(Xb)
        loss = loss_function(y_pred, Yb)
        model.zero_grad()
        dL_dy = loss_function.backward()
        model.backward(dL_dy)
        model.step(LR)
        losses.append(loss.item())
    plt.figure(figsize=(6, 5))

    plt.subplot(2, 1, 1)
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")

    # Make predictions on the test set
    model.eval()
    predictions = []
    for i in range(len(X_test)):
        # seq = X_test[i].reshape(1, 30)
        seq = np.expand_dims(X_test[i], 0)  # add batch dimension
        prediction = model(seq)
        predictions.append(prediction.item())

    # Denormalize the predictions
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    # Denormalize the actual test values
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Generate dates for the test data
    test_dates = data.index[-test_size + seq_length :]

    # Plot the results
    plt.subplot(2, 1, 2)
    plt.plot(data.index, temperatures, label="True Data")

    plt.plot(test_dates, predictions, label="Predicted Data", color="red")
    plt.title("Temperature Prediction")
    plt.xlabel("Date")
    plt.ylabel("Temperature")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train()
