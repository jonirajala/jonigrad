import numpy as np
from keras.datasets import mnist
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def compute_accuracy(model, X_test, y_test):
    correct_predictions = 0
    total_predictions = X_test.shape[0]
    for i in range(total_predictions):
        Xb = X_test[i : i + 1]
        out = model(Xb)
        prediction = np.argmax(out, axis=1)
        if prediction == y_test[i]:
            correct_predictions += 1
    return correct_predictions / total_predictions


def load_mnist(flatten=True):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
        X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
    else:
        WIDTH, HEIGHT = X_train.shape[1], X_train.shape[2]
        X_train = X_train.reshape(-1, 1, HEIGHT, WIDTH).astype(np.float32) / 255.0
        X_test = X_test.reshape(-1, 1, HEIGHT, WIDTH).astype(np.float32) / 255.0

    return X_train, y_train, X_test, y_test


def load_temperature_data(seq_length):
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
    data = pd.read_csv(url, parse_dates=["Date"], index_col="Date")
    temperatures = data["Temp"].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_temperatures = scaler.fit_transform(temperatures)

    n_test_years = 3
    test_size = n_test_years * 365  # number of days in 3 years
    train_data = normalized_temperatures[:-test_size]
    test_data = normalized_temperatures[-test_size:]

    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)

    return X_train, y_train, X_test, y_test, data, temperatures, scaler


def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i : i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)
