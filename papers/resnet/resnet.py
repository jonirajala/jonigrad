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

from jonigrad.layers import Conv, BatchNorm, ReLU, Module, MaxPool, AvgPool, Linear

class ResNet(Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.input_layer = InputLayer(in_channels=3, out_channels=64, stride=2)
        self.res_block1 = ResidualBlock(in_channels=64, out_channels=64, stride=1)
        self.res_block2 = ResidualBlock(in_channels=64, out_channels=128, stride=2)
        self.res_block3 = ResidualBlock(in_channels=128, out_channels=256, stride=2)
        self.res_block4 = ResidualBlock(in_channels=256, out_channels=512, stride=2)
        self.fc_layer = FullyConnectedLayer(num_classes=num_classes)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        y = self.fc_layer(x)
        return y


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
        y = self.relu1(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        
        if self.shortcut is not None:
            x = self.shortcut(x)
        
        y += x
        y = self.relu2(y)
        return y
    
    def backward(self, dL_dy):
        dL_dy = self.relu2.backward(dL_dy)
        
        if self.shortcut is not None:
            dL_dx_shortcut = dL_dy
        else:
            dL_dx_shortcut = dL_dy

        dL_dy = self.bn2.backward(self.conv2.backward(dL_dy))
        dL_dy = self.relu1.backward(dL_dy)
        dL_dx = self.bn1.backward(self.conv1.backward(dL_dy))

        if self.shortcut is not None:
            dL_dx_shortcut = self.shortcut.backward(dL_dx_shortcut)
            dL_dx += dL_dx_shortcut

        return dL_dx

class InputLayer(Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = Conv(in_channels, out_channels, kernel_size=7, stride=stride, padding=3)
        self.bn = BatchNorm(out_channels)
        self.relu = ReLU()
        self.maxpool = MaxPool(kernel_size=3, stride=2, padding=1 )
    
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
        self.fc = Linear(in_features=512, out_features=num_classes)
        self.avg_pool = AvgPool(kernel_size=7, stride=1)
    
    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        
        return y

    def backward(self, dL_dy):
        dL_dx = self.fc.backward(dL_dy)
        dL_dx = dL_dx.view(dL_dx.size(0), 512, 1, 1)
        dL_dx = self.avg_pool.backward(dL_dy)
        return dL_dx