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


from jonigrad.layers import Conv, ReLU, Linear, LRNorm, MaxPool, Dropout,CrossEntropyLoss, Flatten

alexnet = []
alexnet.append(Conv(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=1))
alexnet.append(ReLU())
alexnet.append(LRNorm(size=5, alpha=1e-4, beta=1, k=2))
alexnet.append(MaxPool(kernel_size=3, stride=2))

# # Second Convolutional Layer
# alexnet.append(Conv(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1))
# alexnet.append(ReLU())
# alexnet.append(LRNorm(size=5, alpha=1e-4, beta=1, k=2))
# alexnet.append(MaxPool(kernel_size=3, stride=2))

# # Third Convolutional Layer
# alexnet.append(Conv(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1))
# alexnet.append(ReLU())

# Fourth Convolutional Layer
alexnet.append(Conv(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1))
alexnet.append(ReLU())

# Fifth Convolutional Layer
alexnet.append(Conv(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1))
alexnet.append(ReLU())

# Max Pooling Layer
alexnet.append(MaxPool(kernel_size=2, stride=1))

# First Fully Connected Layer
alexnet.append(Flatten())
alexnet.append(Linear(in_features=3872, out_features=1024))
alexnet.append(ReLU())
alexnet.append(Dropout(p=0.5))

# Second Fully Connected Layer
alexnet.append(Linear(in_features=1024, out_features=1024))
alexnet.append(ReLU())
alexnet.append(Dropout(p=0.5))

# Output Layer
alexnet.append(Linear(in_features=1024, out_features=10))
