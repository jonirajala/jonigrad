# https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
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

from jonigrad.layers import Conv, ReLU, Linear, LRNorm, MaxPool, Dropout

alexnet = []
alexnet.append(Conv())
alexnet.append(ReLU())
alexnet.append(LRNorm())
alexnet.append(MaxPool())

alexnet.append(Conv())
alexnet.append(ReLU())
alexnet.append(LRNorm())
alexnet.append(MaxPool())

alexnet.append(Conv())
alexnet.append(ReLU())
alexnet.append(Conv())
alexnet.append(ReLU())
alexnet.append(Conv())
alexnet.append(ReLU())

alexnet.append(MaxPool())

alexnet.append(Linear())
alexnet.append(ReLU())
alexnet.append(Dropout())

alexnet.append(Linear())
alexnet.append(ReLU())
alexnet.append(Dropout())

alexnet.append(Linear())



