from keras.datasets import mnist
import time
import numpy as np
import matplotlib.pyplot as plt

from layers import Linear, ReLU, CrossEntropyLoss



BATCH_SIZE = 64
ITERS = 1000
LR = 0.001
g = np.random.default_rng()  # create a random generator


def load_data():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = train_X.reshape(train_X.shape[0], -1)
    test_X = test_X.reshape(test_X.shape[0], -1)

    return train_X, train_y, test_X, test_y

def main():
    train_X, train_y, test_X, test_y = load_data()

    joni_model = [Linear(train_X.shape[1], 256), ReLU(), Linear(256, 10)]
    joni_loss_f = CrossEntropyLoss()

    train_losses = []
    test_losses = []
    test_iterations = []

    start_time = time.time()

    for i in range(ITERS):
        print(i)
        ix = g.integers(low=0, high=train_X.shape[0], size=BATCH_SIZE)
        Xb, Yb = train_X[ix], train_y[ix]

        for layer in joni_model:
            Xb = layer(Xb)
   
        out = Xb
        loss = joni_loss_f(out, Yb)
        
        for layer in joni_model:
            layer.zero_grad()
        
        dL_dy = joni_loss_f.backward()
        for layer in reversed(joni_model):
            dL_dy = layer.backward(dL_dy)
        
        for layer in joni_model:
            layer.step(LR)

        train_losses.append(loss.item() / BATCH_SIZE)

        if i % 100 == 0:
            Xb, Yb = test_X, test_y # batch X,Y
            for layer in joni_model:
                Xb = layer(Xb)
                
            test_out = Xb
            test_loss = joni_loss_f(test_out, Yb)
            test_losses.append(test_loss.item() / Xb.shape[0])
            test_iterations.append(i)

    end_time = time.time()

    print(f"Execution time: {end_time - start_time} seconds")

    plt.plot(range(ITERS), train_losses, label='Training Loss')
    plt.plot(test_iterations, test_losses, label='Test Loss')
    plt.show()

if __name__ == '__main__':
    main()