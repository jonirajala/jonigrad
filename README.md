# JoniGrad

JoniGrad is a basic deep learning framework designed for implementing research papers. The repository includes the core framework, examples, and tests to facilitate deep learning research and development.


## How to Run Tests
To run the tests, use the following command:

```sh
python3 -m pytest tests/tests.py
```

or

```sh
python3  tests/speed_test.py
```

to get comparision against pytorch



## Papers implemented
Papers that have been implemented with joni grad can be found under papers/

- AlexNet: Introduces a deep convolutional neural network architecture that won the ImageNet Large Scale Visual Recognition Challenge in 2012.
- EfficientNet: Proposes a model scaling method that uniformly scales all dimensions of depth, width, and resolution using a compound coefficient.
- ResNet: Introduces residual learning to ease the training of deep neural networks, enabling the training of networks with significantly deeper layers.
- Vision Transformer: Applies the transformer architecture, traditionally used for natural language processing, to vision tasks, demonstrating high performance with sufficient data.