"""
This landmark paper introduced AlexNet, which achieved unprecedented performance on the ImageNet challenge and popularized the use of deep convolutional neural networks. It also utilized GPUs for training large-scale neural networks.

model:
- eight learned layers
    - five convolutional
    - three fully-connected
- ReLU Nonlinearity
- LocalResponseNorm after the ReLU activation in certain layers.
    - Specifically, it takes the output of each neuron and divides it by a term that depends on the outputs of neighboring neurons.
    - This term is calculated using the sum of squared outputs of nearby neurons within the same layer.
- MaxPooling
- Dropout
    - We use dropout in the first two fully-connected layers of Figure

Training:
- stochastic gradient descent
- batch size of 128 examples
- momentum of 0.9
- weight decay of 0.0005 
"""

from jonigrad.layers import (
    Conv,
    ReLU,
    Linear,
    LocalResponseNorm,
    MaxPool,
    Dropout,
    Flatten,
    Module,
)


class AlexNet(Module):
    def __init__(self):
        self.conv1 = Conv(
            in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=1
        )
        self.conv2 = Conv(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = Conv(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv4 = Conv(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv5 = Conv(
            in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.relu1 = ReLU()
        self.relu2 = ReLU()
        self.relu3 = ReLU()
        self.relu4 = ReLU()
        self.relu5 = ReLU()
        self.relu6 = ReLU()
        self.relu7 = ReLU()
        self.lr_norm1 = LocalResponseNorm(size=5, alpha=1e-4, beta=1, k=2)
        self.lr_norm2 = LocalResponseNorm(size=5, alpha=1e-4, beta=1, k=2)
        self.max_pool1 = MaxPool(kernel_size=3, stride=2)
        self.max_pool2 = MaxPool(kernel_size=3, stride=2)
        self.max_pool3 = MaxPool(kernel_size=2, stride=1)
        self.flatten = Flatten()
        self.fc1 = Linear(in_features=512, out_features=256)
        self.fc2 = Linear(in_features=256, out_features=128)
        self.fc3 = Linear(in_features=128, out_features=10)
        self.dropout1 = Dropout(p=0.5)
        self.dropout2 = Dropout(p=0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.lr_norm1(x)
        x = self.max_pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.lr_norm2(x)
        x = self.max_pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)

        x = self.max_pool3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu7(x)
        x = self.dropout2(x)

        x = self.fc3(x)

        return x

    def backward(self, dL_dy):
        dL_dx = self.fc3.backward(dL_dy)

        dL_dx = self.dropout2.backward(dL_dx)
        dL_dx = self.relu7.backward(dL_dx)
        dL_dx = self.fc2.backward(dL_dx)

        dL_dx = self.dropout1.backward(dL_dx)
        dL_dx = self.relu6.backward(dL_dx)
        dL_dx = self.fc1.backward(dL_dx)

        dL_dx = self.flatten.backward(dL_dx)

        dL_dx = self.max_pool3.backward(dL_dx)

        dL_dx = self.relu5.backward(dL_dx)
        dL_dx = self.conv5.backward(dL_dx)

        dL_dx = self.relu4.backward(dL_dx)
        dL_dx = self.conv4.backward(dL_dx)

        dL_dx = self.relu3.backward(dL_dx)
        dL_dx = self.conv3.backward(dL_dx)

        dL_dx = self.max_pool2.backward(dL_dx)
        dL_dx = self.lr_norm2.backward(dL_dx)
        dL_dx = self.relu2.backward(dL_dx)
        dL_dx = self.conv2.backward(dL_dx)

        dL_dx = self.max_pool1.backward(dL_dx)
        dL_dx = self.lr_norm1.backward(dL_dx)
        dL_dx = self.relu1.backward(dL_dx)
        dL_dx = self.conv1.backward(dL_dx)

        return dL_dx
