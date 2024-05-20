import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import time

# Hyperparameters
BATCH_SIZE = 32
ITERS = 100
LR = 0.001
g = np.random.default_rng()  # create a random generator

# Load data
def load_data():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    WIDTH, HEIGHT = train_X.shape[1], train_X.shape[2]
    train_X = train_X.reshape(-1, 1, HEIGHT, WIDTH).astype(np.float32) / 255.0
    test_X = test_X.reshape(-1, 1, HEIGHT, WIDTH).astype(np.float32) / 255.0

    train_X = torch.tensor(train_X)
    train_y = torch.tensor(train_y, dtype=torch.long)
    test_X = torch.tensor(test_X)
    test_y = torch.tensor(test_y, dtype=torch.long)

    return train_X, train_y, test_X, test_y

# Define the model
class JoniModel(nn.Module):
    def __init__(self):
        super(JoniModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 3)
        self.conv2 = nn.Conv2d(3, 1, 3)
        self.fc = nn.Linear(24 * 24, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 24 * 24)
        x = self.fc(x)
        return x

def main():
    train_X, train_y, test_X, test_y = load_data()

    model = JoniModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    train_losses = []
    test_losses = []
    test_iterations = []

    start_time = time.time()

    for i in range(ITERS):
        model.train()
        
        # Sample a batch of data
        ix = g.integers(low=0, high=train_X.shape[0], size=BATCH_SIZE)
        Xb, Yb = train_X[ix], train_y[ix]
        
        optimizer.zero_grad()
        output = model(Xb)
        loss = criterion(output, Yb)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        if i % 10 == 0:
            print(f"Iteration {i}/{ITERS}")
            
            # Evaluate on the test set
            model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                output = model(test_X)
                test_loss = criterion(output, test_y).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(test_y.view_as(pred)).sum().item()

            test_loss /= len(test_X)
            test_losses.append(test_loss)
            test_iterations.append(i)
            print(f"Test Loss: {test_loss}, Accuracy: {correct / len(test_X)}")

    end_time = time.time()

    print(f"Execution time: {end_time - start_time} seconds")

    plt.plot(range(ITERS), train_losses, label='Training Loss')
    plt.plot(test_iterations, test_losses, label='Test Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
