import numpy as np
from keras.datasets import mnist

def compute_accuracy(model, test_X, test_y):
    correct_predictions = 0
    total_predictions = test_X.shape[0]
    for i in range(total_predictions):
        Xb = test_X[i:i+1]
        out = model(Xb)
        prediction = np.argmax(out, axis=1)
        if prediction == test_y[i]:
            correct_predictions += 1
    return correct_predictions / total_predictions

def load_mnist(flatten=True):
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    if flatten:
        train_X = train_X.reshape(train_X.shape[0], -1) / 255.0
        test_X = test_X.reshape(test_X.shape[0], -1) / 255.0
    else:
        WIDTH, HEIGHT = train_X.shape[1], train_X.shape[2]
        train_X = train_X.reshape(-1, 1,  HEIGHT, WIDTH).astype(np.float32) / 255.0
        test_X = test_X.reshape(-1, 1, HEIGHT, WIDTH).astype(np.float32) / 255.0

    return train_X, train_y, test_X, test_y