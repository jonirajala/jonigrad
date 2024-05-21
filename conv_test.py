from keras.datasets import mnist
import time
import numpy as np
import matplotlib.pyplot as plt

from layers import Linear, ReLU, CrossEntropyLoss, Conv

BATCH_SIZE = 32
ITERS = 100
LR = 0.001
g = np.random.default_rng()  # create a random generator


def load_data():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    WIDTH, HEIGHT = train_X.shape[1], train_X.shape[2]
    train_X = train_X.reshape(-1, 1,  HEIGHT, WIDTH).astype(np.float32) / 255.0
    test_X = test_X.reshape(-1, 1, HEIGHT, WIDTH).astype(np.float32) / 255.0

    return train_X, train_y, test_X, test_y

def main():
    train_X, train_y, test_X, test_y = load_data()

    joni_model = [Conv(1, 3, 3), ReLU(), Conv(3, 1, 3), ReLU(), Linear(576, 10)]
    joni_loss_f = CrossEntropyLoss()

    train_losses = []
    test_losses = []
    test_iterations = []

    start_time = time.time()

    for i in range(ITERS):
        
        ix = g.integers(low=0, high=train_X.shape[0], size=BATCH_SIZE)
        Xb, Yb = train_X[ix], train_y[ix]
        # for layer in joni_model:
        #     Xb = layer(Xb)

        Xb = joni_model[0](Xb)
        Xb = joni_model[1](Xb)
        Xb = joni_model[2](Xb)
        Xb = joni_model[3](Xb)
        Xb = Xb.reshape(-1, 24*24)
        out = joni_model[4](Xb)
   
        loss = joni_loss_f(out, Yb)
        
        for layer in joni_model:
            layer.zero_grad()

        dL_dy = joni_loss_f.backward()
        for layer in reversed(joni_model):
            dL_dy = layer.backward(dL_dy)
        for layer in joni_model:
            layer.step(LR)
        train_losses.append(loss.item() / BATCH_SIZE)
        if i % 10 == 0:
            print(i)

    end_time = time.time()

    print(f"Execution time: {end_time - start_time} seconds")

    plt.plot(range(ITERS), train_losses, label='Training Loss')
    plt.plot(test_iterations, test_losses, label='Test Loss')
    plt.show()

if __name__ == '__main__':
    main()