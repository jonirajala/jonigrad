import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from jonigrad.layers import Linear, ReLU, CrossEntropyLoss, Module
from jonigrad.utils import compute_accuracy, load_mnist


BATCH_SIZE = 64
ITERS = 10000
LR = 0.001
g = np.random.default_rng()  # create a random generator


class LinearModel(Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = Linear(input_size, 256)
        self.fc2 = Linear(256, 128)
        self.fc3 = Linear(128, output_size)
        self.relu1 = ReLU()
        self.relu2 = ReLU()

    def forward(self, input_seq):
        y = self.relu1.forward(self.fc1.forward(input_seq))
        y = self.relu2.forward(self.fc2.forward(y))
        y = self.fc3.forward(y)
        return y

    def backward(self, dL_dy):
        dL_dy = self.fc3.backward(dL_dy)
        dL_dy = self.relu2.backward(dL_dy)
        dL_dy = self.fc2.backward(dL_dy)
        dL_dy = self.relu1.backward(dL_dy)
        dl_dx = self.fc1.backward(dL_dy)
        return dl_dx


def train():
    train_X, train_y, test_X, test_y = load_mnist(flatten=True)

    model = LinearModel(train_X.shape[1], 10)
    loss_f = CrossEntropyLoss()

    train_losses = []

    for iter in tqdm(range(ITERS)):
        ix = g.integers(low=0, high=train_X.shape[0], size=BATCH_SIZE)
        Xb, Yb = train_X[ix], train_y[ix]

        out = model(Xb)
        loss = loss_f(out, Yb)
        model.zero_grad()

        dL_dy = loss_f.backward()
        model.backward(dL_dy)

        model.step(LR)

        train_losses.append(loss.item() / BATCH_SIZE)

    accuracy = compute_accuracy(model, test_X, test_y)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    plt.plot(range(ITERS), train_losses, label="Training Loss")
    plt.show()


if __name__ == "__main__":
    train()
