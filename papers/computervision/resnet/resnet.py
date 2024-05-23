"""
https://arxiv.org/pdf/1512.03385
What is a resne?
Traditional Approach
In traditional neural networks, each layer tries to learn the exact mapping from input to output.
Think of it as trying to directly learn the function that maps your input data to the correct output.
Residual Networks (ResNets)
ResNets take a different approach. Instead of trying to learn the direct mapping,
they learn the "residual" – the difference between the desired output and the input to that layer.

the concept of learning residuals in ResNets (Residual Networks) is implemented through what are known as skip connections

model
- Input Layer
    - Conv
    - Maxpool

- Multiple Residual blocks (The sets progressively increase the depth (number of channels) of the feature maps while reducing their spatial dimensions.)
    - 2 conv layers
    - batch normalization and a ReLU after each conv
    - skip connection
- Global average Pooling
- fully connected layer


Residual blocks can be divided into two main types:
- Identity Blocks: Used when the input and output dimensions are the same.
- Convolutional Blocks: Used when the input and output dimensions differ, involving a convolution operation to match dimensions.

"""

from jonigrad.layers import Conv, BatchNorm, ReLU, Module, MaxPool, AvgPool, Linear, Flatten

class ResNet(Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.input_layer = InputLayer(in_channels=1, out_channels=16, stride=1)
        self.res_block1 = ResidualBlock(in_channels=16, out_channels=16, stride=1)
        self.res_block2 = ResidualBlock(in_channels=16, out_channels=32, stride=1)
        self.res_block3 = ResidualBlock(in_channels=32, out_channels=32, stride=1)
        self.res_block4 = ResidualBlock(in_channels=32, out_channels=64, stride=1)
        self.fc_layer = FullyConnectedLayer(num_classes=num_classes)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        y = self.fc_layer(x)
        return y
    
    def backward(self, dL_dy):
        dL_dx = self.fc_layer.backward(dL_dy)
        dL_dx = self.res_block4.backward(dL_dx)
        dL_dx = self.res_block3.backward(dL_dx)
        dL_dx = self.res_block2.backward(dL_dx)
        dL_dx = self.res_block1.backward(dL_dx)
        dL_dx = self.input_layer.backward(dL_dx)
        return dL_dx


class ResidualBlock(Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.conv1 = Conv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = Conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu1 = ReLU()
        self.relu2 = ReLU()
        self.bn1 = BatchNorm(out_channels)
        self.bn2 = BatchNorm(out_channels)
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Conv(in_channels, out_channels, kernel_size=1, stride=stride)
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.shortcut is not None:
            identity = self.shortcut(x)
        
        out += identity
        out = self.relu2(out)
        return out
    
    def backward(self, dL_dy):
        dL_dx = self.relu2.backward(dL_dy)
        didentity = dL_dx
        dL_dx = self.bn2.backward(dL_dx)
        dL_dx = self.conv2.backward(dL_dx)
        
        # Backprop through the ReLU, BatchNorm, and Conv layers in the first part of forward
        dL_dx = self.relu1.backward(dL_dx)
        dL_dx = self.bn1.backward(dL_dx)
        dL_dx = self.conv1.backward(dL_dx)
        
        # Handle the shortcut path
        if self.shortcut is not None:
            dshortcut = self.shortcut.backward(didentity)
        else:
            dshortcut = didentity
        
        # Sum the gradients from the main path and shortcut path
        dL_dx += dshortcut
        
        return dL_dx


class InputLayer(Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = Conv(in_channels, out_channels, kernel_size=7, stride=stride, padding=3)
        self.bn = BatchNorm(out_channels)
        self.relu = ReLU()
        self.maxpool = MaxPool(kernel_size=3, stride=2)
        
    
    def forward(self, x):
        y = self.relu(self.bn(self.conv(x)))
        y = self.maxpool(y)
        return y

    def backward(self, dL_dy):
        dL_dx = self.maxpool.backward(dL_dy)
        dL_dx = self.relu.backward(dL_dx)
        dL_dx = self.bn.backward(dL_dx)
        dL_dx = self.conv.backward(dL_dx)
        return dL_dx

class FullyConnectedLayer(Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc = Linear(in_features=3136, out_features=num_classes)
        self.avg_pool = AvgPool(kernel_size=7, stride=1)
        self.flatten = Flatten()
    
    def forward(self, x):
        x = self.avg_pool(x)
        x = self.flatten(x)
        y = self.fc(x)
        
        return y

    def backward(self, dL_dy):
        dL_dx = self.fc.backward(dL_dy)
        dL_dx = self.flatten.backward(dL_dx)
        dL_dx = self.avg_pool.backward(dL_dx)
        return dL_dx