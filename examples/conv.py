import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from jonigrad.layers import Linear, ReLU, CrossEntropyLoss, Conv, Flatten, Module
from jonigrad.utils import load_mnist, compute_accuracy

BATCH_SIZE = 32
ITERS = 300
LR = 0.001
g = np.random.default_rng()  # create a random generator


class ConvModel(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv(1, 6, 3)
        self.relu1 = ReLU()
        self.conv2 = Conv(6, 1, 3)
        self.relu2 = ReLU()
        self.flatten = Flatten()
        self.fc = Linear(576, 10)

    def forward(self, input_seq):
        y = self.conv1.forward(input_seq)
        y = self.relu1.forward(y)
        y = self.conv2.forward(y)
        y = self.relu2.forward(y)
        y = self.flatten.forward(y)
        y = self.fc.forward(y)
        return y

    def backward(self, dL_dy):
        dL_dy = self.fc.backward(dL_dy)
        dL_dy = self.flatten.backward(dL_dy)
        dL_dy = self.relu2.backward(dL_dy)
        dL_dy = self.conv2.backward(dL_dy)
        dL_dy = self.relu1.backward(dL_dy)
        dL_dy = self.conv1.backward(dL_dy)
        return dL_dy


def main():
    train_X, train_y, test_X, test_y = load_mnist(flatten=False)

    model = ConvModel()
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
    main()
