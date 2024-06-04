import unittest
import numpy as np
import torch
import torch.nn as nn
from jonigrad.layers import (
    Linear,
    ReLU,
    Conv,
    CrossEntropyLoss,
    MSELoss,
    MaxPool,
    LocalResponseNorm,
    BatchNorm,
    AvgPool,
    Tanh,
    Sigmoid,
    LSTM,
    LayerNorm
)


class TestLinearLayer(unittest.TestCase):
    def setUp(self):
        self.input_size = 10
        self.output_size = 5
        self.batch_size = 3

        self.custom_linear = Linear(self.input_size, self.output_size)
        self.torch_linear = nn.Linear(self.input_size, self.output_size).float()

        self.torch_linear.weight.data = torch.from_numpy(
            self.custom_linear._params["W"].data.copy()
        ).float()
        self.torch_linear.bias.data = torch.from_numpy(
            self.custom_linear._params["B"].data.flatten().copy()
        ).float()

        # Create random input
        self.x = np.random.rand(self.batch_size, 12, self.input_size).astype(np.float32)
        self.x_torch = torch.tensor(self.x)
        self.x_torch.requires_grad_(True)

    def test_forward_pass(self):
        # Forward pass through both layers
        custom_output = self.custom_linear(self.x)
        torch_output = self.torch_linear(self.x_torch).detach().numpy()

        # Check if outputs are the same
        self.assertTrue(
            np.allclose(custom_output, torch_output, atol=1e-6),
            "Forward pass outputs do not match!",
        )

    def test_backward_pass(self):
        # ForwaQrd pass through both layers to set input
        custom_output = self.custom_linear(self.x)
        torch_output = self.torch_linear(self.x_torch)

        # Create random gradient for backward pass
        grad_output = np.random.rand(self.batch_size, 12, self.output_size).astype(
            np.float32
        )
        grad_output_torch = torch.from_numpy(grad_output).float()

        # Backward pass through custom layer
        self.custom_linear.zero_grad()
        print(grad_output.shape)
        custom_grad_input = self.custom_linear.backward(grad_output)

        # Backward pass through PyTorch layer
        self.torch_linear.zero_grad()
        torch_output.backward(
            grad_output_torch, retain_graph=True
        )  # Perform the backward pass
        torch_grad_input = (
            self.x_torch.grad.numpy()
        )  # Get the gradient of the input tensor

        # Check if gradients are the same
        self.assertTrue(
            np.allclose(
                self.custom_linear._params["W"].grad,
                self.torch_linear.weight.grad.numpy(),
                atol=1e-6,
            ),
            "Weight gradients do not match!",
        )
        self.assertTrue(
            np.allclose(
                self.custom_linear._params["B"].grad.flatten(),
                self.torch_linear.bias.grad.numpy(),
                atol=1e-6,
            ),
            "Bias gradients do not match!",
        )
        self.assertTrue(
            np.allclose(custom_grad_input, torch_grad_input, atol=1e-6),
            "Backward pass gradients do not match!",
        )


class TestReLULayer(unittest.TestCase):
    def setUp(self):
        self.batch_size = 3
        self.input_size = 10

        # Initialize custom ReLU layer
        self.custom_relu = ReLU()

        # Initialize PyTorch ReLU layer
        self.torch_relu = nn.ReLU()

        # Create random input
        self.x = np.random.randn(self.batch_size, self.input_size).astype(np.float32)
        self.x_torch = torch.from_numpy(self.x).float()
        self.x_torch.requires_grad_(True)  # Ensure that x_torch requires gradients

    def test_forward_pass(self):
        # Forward pass through both layers
        custom_output = self.custom_relu(self.x)
        torch_output = self.torch_relu(self.x_torch).detach().numpy()

        # Check if outputs are the same
        self.assertTrue(
            np.allclose(custom_output, torch_output, atol=1e-6),
            "Forward pass outputs do not match!",
        )

    def test_backward_pass(self):
        # Forward pass through both layers to set input
        custom_output = self.custom_relu(self.x)
        torch_output = self.torch_relu(self.x_torch)

        # Create random gradient for backward pass
        grad_output = np.random.randn(self.batch_size, self.input_size).astype(
            np.float32
        )
        grad_output_torch = torch.from_numpy(grad_output).float()

        # Backward pass through custom layer
        custom_grad_input = self.custom_relu.backward(grad_output)

        # Backward pass through PyTorch layer
        torch_output.backward(grad_output_torch)
        torch_grad_input = self.x_torch.grad.numpy()

        # Check if gradients are the same
        self.assertTrue(
            np.allclose(custom_grad_input, torch_grad_input, atol=1e-6),
            "Backward pass gradients do not match!",
        )


class TestTanhLayer(unittest.TestCase):
    def setUp(self):
        self.batch_size = 3
        self.input_size = 10

        self.custom_tanh = Tanh()

        self.torch_tanh = nn.Tanh()

        # Create random input
        self.x = np.random.randn(self.batch_size, self.input_size).astype(np.float32)
        self.x_torch = torch.from_numpy(self.x).float()
        self.x_torch.requires_grad_(True)  # Ensure that x_torch requires gradients

    def test_forward_pass(self):
        # Forward pass through both layers
        custom_output = self.custom_tanh(self.x)
        torch_output = self.torch_tanh(self.x_torch).detach().numpy()

        # Check if outputs are the same
        self.assertTrue(
            np.allclose(custom_output, torch_output, atol=1e-6),
            "Forward pass outputs do not match!",
        )

    def test_backward_pass(self):
        # Forward pass through both layers to set input
        custom_output = self.custom_tanh(self.x)
        torch_output = self.torch_tanh(self.x_torch)

        # Create random gradient for backward pass
        grad_output = np.random.randn(self.batch_size, self.input_size).astype(
            np.float32
        )
        grad_output_torch = torch.from_numpy(grad_output).float()

        # Backward pass through custom layer
        custom_grad_input = self.custom_tanh.backward(grad_output)

        # Backward pass through PyTorch layer
        torch_output.backward(grad_output_torch)
        torch_grad_input = self.x_torch.grad.numpy()

        # Check if gradients are the same
        self.assertTrue(
            np.allclose(custom_grad_input, torch_grad_input, atol=1e-6),
            "Backward pass gradients do not match!",
        )


class TestSigmoidLayer(unittest.TestCase):
    def setUp(self):
        self.batch_size = 3
        self.input_size = 10

        self.custom_sigmoid = Sigmoid()

        self.torch_sigmoid = nn.Sigmoid()

        # Create random input
        self.x = np.random.randn(self.batch_size, self.input_size).astype(np.float32)
        self.x_torch = torch.from_numpy(self.x).float()
        self.x_torch.requires_grad_(True)  # Ensure that x_torch requires gradients

    def test_forward_pass(self):
        # Forward pass through both layers
        custom_output = self.custom_sigmoid(self.x)
        torch_output = self.torch_sigmoid(self.x_torch).detach().numpy()

        # Check if outputs are the same
        self.assertTrue(
            np.allclose(custom_output, torch_output, atol=1e-6),
            "Forward pass outputs do not match!",
        )

    def test_backward_pass(self):
        # Forward pass through both layers to set input
        custom_output = self.custom_sigmoid(self.x)
        torch_output = self.torch_sigmoid(self.x_torch)

        # Create random gradient for backward pass
        grad_output = np.random.randn(self.batch_size, self.input_size).astype(
            np.float32
        )
        grad_output_torch = torch.from_numpy(grad_output).float()

        # Backward pass through custom layer
        custom_grad_input = self.custom_sigmoid.backward(grad_output)

        # Backward pass through PyTorch layer
        torch_output.backward(grad_output_torch)
        torch_grad_input = self.x_torch.grad.numpy()

        # Check if gradients are the same
        self.assertTrue(
            np.allclose(custom_grad_input, torch_grad_input, atol=1e-6),
            "Backward pass gradients do not match!",
        )


class TestConvLayer(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.n_channels = 2
        self.out_channels = 2
        self.kernel_size = 3
        self.stride = 1
        self.padding = 1
        self.input_size = 4

        # Initialize custom conv layer
        self.custom_conv = Conv(
            self.n_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
        )

        # Initialize PyTorch conv layer
        self.torch_conv = nn.Conv2d(
            self.n_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            bias=False,
        ).float()

        # Copy weights from custom layer to PyTorch layer
        self.torch_conv.weight.data = torch.from_numpy(
            self.custom_conv._params["F"].data
        ).float()

        # Create random input
        self.x = np.random.randn(
            self.batch_size, self.n_channels, self.input_size, self.input_size
        ).astype(np.float32)
        self.x_torch = torch.from_numpy(self.x).float()
        self.x_torch.requires_grad_(True)

    def test_forward_pass(self):
        # Forward pass through both layers

        custom_output = self.custom_conv(self.x)
        torch_output = (
            self.torch_conv(self.x_torch).detach().numpy()
        )  # Change to (N, H, W, C)

        # Check if outputs are the same
        print(custom_output.shape, torch_output.shape)
        self.assertTrue(
            np.allclose(custom_output, torch_output, atol=1e-6),
            "Forward pass outputs do not match!",
        )

    def test_backward_pass(self):
        # Forward pass through both layers to set input
        custom_output = self.custom_conv(self.x)
        torch_output = self.torch_conv(self.x_torch)

        # Create random gradient for backward pass
        grad_output = np.random.randn(*custom_output.shape).astype(np.float32)
        grad_output_torch = torch.from_numpy(
            grad_output
        ).float()  # Change to (N, C, H, W)

        # Backward pass through custom layer
        self.custom_conv.zero_grad()
        custom_grad_input = self.custom_conv.backward(grad_output)

        # Backward pass through PyTorch layer
        self.torch_conv.zero_grad()
        torch_output.backward(grad_output_torch)
        torch_grad_input = self.x_torch.grad.numpy()  # Change to (N, H, W, C)

        # Check if gradients are the same
        self.assertTrue(
            np.allclose(
                self.custom_conv._params["F"].grad,
                self.torch_conv.weight.grad.numpy(),
                atol=1e-6,
            ),
            "Weight gradients do not match!",
        )
        self.assertTrue(
            np.allclose(custom_grad_input, torch_grad_input, atol=1e-6),
            "Backward pass gradients do not match!",
        )


class TestCrossEntropyLoss(unittest.TestCase):
    def setUp(self):
        self.batch_size = 5
        self.num_classes = 3

        # Initialize custom cross-entropy loss
        self.custom_loss = CrossEntropyLoss()

        # Initialize PyTorch cross-entropy loss
        self.torch_loss = torch.nn.CrossEntropyLoss()

        # Create random predictions and targets
        self.preds = np.random.randn(self.batch_size, self.num_classes).astype(
            np.float32
        )
        self.targs = np.random.randint(0, self.num_classes, self.batch_size)

        # Convert to PyTorch tensors
        self.preds_torch = torch.tensor(self.preds, requires_grad=True)
        self.targs_torch = torch.tensor(self.targs, dtype=torch.long)

    def test_forward_pass(self):
        # Calculate loss using custom loss
        custom_loss_value = self.custom_loss(self.preds, self.targs)

        # Calculate loss using PyTorch loss
        torch_loss_value = self.torch_loss(self.preds_torch, self.targs_torch).item()

        # Check if the loss values are the same
        self.assertAlmostEqual(
            custom_loss_value,
            torch_loss_value,
            places=6,
            msg="Loss values do not match!",
        )

    def test_backward_pass(self):
        # Calculate loss and backward pass using custom loss
        self.custom_loss(self.preds, self.targs)
        custom_grad_input = self.custom_loss.backward()

        # Calculate loss and backward pass using PyTorch loss
        torch_loss_value = self.torch_loss(self.preds_torch, self.targs_torch)
        torch_loss_value.backward()
        torch_grad_input = self.preds_torch.grad.numpy()

        # Check if the gradients are the same
        self.assertTrue(
            np.allclose(custom_grad_input, torch_grad_input, atol=1e-6),
            "Gradients do not match!",
        )


class TestMSELoss(unittest.TestCase):
    def setUp(self):
        self.batch_size = 5
        self.num_features = 3

        # Initialize custom MSE loss
        self.custom_loss = MSELoss()

        # Initialize PyTorch MSE loss
        self.torch_loss = torch.nn.MSELoss()

        # Create random predictions and targets
        self.preds = np.random.randn(self.batch_size, self.num_features).astype(
            np.float32
        )
        self.targs = np.random.randn(self.batch_size, self.num_features).astype(
            np.float32
        )

        # Convert to PyTorch tensors
        self.preds_torch = torch.tensor(self.preds, requires_grad=True)
        self.targs_torch = torch.tensor(self.targs, requires_grad=False)

    def test_forward_pass(self):
        # Calculate loss using custom loss
        custom_loss_value = self.custom_loss(self.preds, self.targs)
        print(self.preds_torch.shape, self.targs_torch.shape)

        # Calculate loss using PyTorch loss
        torch_loss_value = self.torch_loss(self.preds_torch, self.targs_torch).item()
        # Check if the loss values are the same
        self.assertAlmostEqual(
            custom_loss_value,
            torch_loss_value,
            places=6,
            msg="Loss values do not match!",
        )

    def test_backward_pass(self):
        # Calculate loss and backward pass using custom loss
        self.custom_loss(self.preds, self.targs)
        custom_grad_input = self.custom_loss.backward()

        # Calculate loss and backward pass using PyTorch loss
        torch_loss_value = self.torch_loss(self.preds_torch, self.targs_torch)
        torch_loss_value.backward()
        torch_grad_input = self.preds_torch.grad.numpy()

        # Check if the gradients are the same
        print(custom_grad_input)
        print(torch_grad_input)
        self.assertTrue(
            np.allclose(custom_grad_input, torch_grad_input, atol=1e-6),
            "Gradients do not match!",
        )


class TestMaxPool(unittest.TestCase):
    def setUp(self):
        self.batch_size = 5
        self.num_channels = 3
        self.height = 10
        self.width = 10
        self.kernel_size = 2
        self.stride = 2

        # Initialize custom max pooling layer
        self.custom_maxpool = MaxPool(kernel_size=self.kernel_size, stride=self.stride)

        # Initialize PyTorch max pooling layer
        self.torch_maxpool = torch.nn.MaxPool2d(
            kernel_size=self.kernel_size, stride=self.stride
        )

        # Create random input tensor
        self.input = np.random.randn(
            self.batch_size, self.num_channels, self.height, self.width
        ).astype(np.float32)

        # Convert to PyTorch tensor
        self.input_torch = torch.tensor(self.input, requires_grad=True)

    def test_forward_pass(self):
        # Calculate output using custom max pooling
        custom_output = self.custom_maxpool(self.input)

        # Calculate output using PyTorch max pooling
        torch_output = self.torch_maxpool(self.input_torch).detach().numpy()

        # Check if the output values are the same
        print(custom_output, torch_output)
        np.testing.assert_allclose(
            custom_output,
            torch_output,
            rtol=1e-6,
            err_msg="Forward pass outputs do not match!",
        )

    def test_backward_pass(self):
        # Forward pass through both layers to set input
        custom_output = self.custom_maxpool(self.input)
        torch_output = self.torch_maxpool(self.input_torch)

        # Create random gradient for backward pass
        grad_output = np.random.randn(*custom_output.shape).astype(np.float32)
        grad_output_torch = torch.from_numpy(grad_output).float()

        # Backward pass through custom layer
        custom_grad_input = self.custom_maxpool.backward(grad_output)

        # Backward pass through PyTorch layer
        torch_output.backward(grad_output_torch)
        torch_grad_input = self.input_torch.grad.numpy()

        # Check if gradients are the same
        np.testing.assert_allclose(
            custom_grad_input,
            torch_grad_input,
            rtol=1e-6,
            err_msg="Backward pass gradients do not match!",
        )


class TestAvgPool(unittest.TestCase):
    def setUp(self):
        self.batch_size = 5
        self.num_channels = 3
        self.height = 10
        self.width = 10
        self.kernel_size = 2
        self.stride = 2

        # Initialize custom average pooling layer
        self.custom_avgpool = AvgPool(kernel_size=self.kernel_size, stride=self.stride)

        # Initialize PyTorch average pooling layer
        self.torch_avgpool = torch.nn.AvgPool2d(
            kernel_size=self.kernel_size, stride=self.stride
        )

        # Create random input tensor
        self.input = np.random.randn(
            self.batch_size, self.num_channels, self.height, self.width
        ).astype(np.float32)

        # Convert to PyTorch tensor
        self.input_torch = torch.tensor(self.input, requires_grad=True)

    def test_forward_pass(self):
        # Calculate output using custom average pooling
        custom_output = self.custom_avgpool.forward(self.input)

        # Calculate output using PyTorch average pooling
        torch_output = self.torch_avgpool(self.input_torch).detach().numpy()

        # Check if the output values are the same
        np.testing.assert_allclose(
            custom_output,
            torch_output,
            rtol=1e-6,
            err_msg="Forward pass outputs do not match!",
        )

    def test_backward_pass(self):
        # Forward pass through both layers to set input
        custom_output = self.custom_avgpool.forward(self.input)
        torch_output = self.torch_avgpool(self.input_torch)

        # Create random gradient for backward pass
        grad_output = np.random.randn(*custom_output.shape).astype(np.float32)
        grad_output_torch = torch.from_numpy(grad_output).float()

        # Backward pass through custom layer
        custom_grad_input = self.custom_avgpool.backward(grad_output)

        # Backward pass through PyTorch layer
        torch_output.backward(grad_output_torch)
        torch_grad_input = self.input_torch.grad.numpy()

        # Check if gradients are the same
        np.testing.assert_allclose(
            custom_grad_input,
            torch_grad_input,
            rtol=1e-6,
            err_msg="Backward pass gradients do not match!",
        )


class TestLRNorm(unittest.TestCase):
    def setUp(self):
        self.batch_size = 5
        self.num_channels = 3
        self.height = 4
        self.width = 4
        self.size = 5
        self.alpha = 1e-4
        self.beta = 1  ## not working with bias < 1
        self.k = 2.0

        # Initialize custom LRN layer
        self.custom_lrn = LocalResponseNorm(
            size=self.size, alpha=self.alpha, beta=self.beta, k=self.k
        )

        # Initialize PyTorch LRN layer
        self.torch_lrn = torch.nn.LocalResponseNorm(
            size=self.size, alpha=self.alpha, beta=self.beta, k=self.k
        )

        # Create random input tensor
        self.input = np.random.randn(
            self.batch_size, self.num_channels, self.height, self.width
        ).astype(np.float32)

        # Copy input for PyTorch tensor
        self.input_torch = torch.tensor(
            self.input, requires_grad=True, dtype=torch.float32
        )

    def test_forward_pass(self):
        custom_output = self.custom_lrn(self.input)
        torch_output = self.torch_lrn(self.input_torch).detach().numpy()
        np.testing.assert_allclose(
            custom_output,
            torch_output,
            rtol=1e-6,
            err_msg="Forward pass outputs do not match!",
        )

    def test_backward_pass(self):
        custom_output = self.custom_lrn(self.input)
        torch_output = self.torch_lrn(self.input_torch)

        grad_output = np.random.randn(*custom_output.shape).astype(np.float32)
        grad_output_torch = torch.from_numpy(grad_output)

        custom_grad_input = self.custom_lrn.backward(grad_output)
        torch_output.backward(grad_output_torch)
        torch_grad_input = self.input_torch.grad.clone().detach().numpy()

        np.testing.assert_allclose(
            custom_grad_input,
            torch_grad_input,
            rtol=1e-6,
            err_msg="Backward pass gradients do not match!",
        )


class TestBatchNormLayer(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.num_features = 2
        self.height = 2
        self.width = 2
        self.eps = 1e-5
        self.momentum = 0.1

        # Initialize custom BatchNorm layer
        self.custom_bn = BatchNorm(
            self.num_features, eps=self.eps, momentum=self.momentum
        )

        # Initialize PyTorch BatchNorm layer
        self.torch_bn = nn.BatchNorm2d(
            self.num_features,
            eps=self.eps,
            momentum=self.momentum,
            affine=True,
            track_running_stats=True,
        )

        # Create random input
        self.x = np.random.randn(
            self.batch_size, self.num_features, self.height, self.width
        ).astype(np.float32)
        self.x_torch = torch.from_numpy(self.x).float()
        self.x_torch.requires_grad_(True)  # Ensure that x_torch requires gradients

    def test_forward_pass(self):
        # Forward pass through both layers
        self.custom_bn.train()
        self.torch_bn.train()
        custom_output = self.custom_bn(self.x)
        torch_output = self.torch_bn(self.x_torch).detach().numpy()

        # Check if outputs are the same
        self.assertTrue(
            np.allclose(custom_output, torch_output, atol=1e-5),
            "Forward pass outputs do not match!",
        )

    def test_backward_pass(self):
        # Forward pass through both layers to set input
        self.custom_bn.train()
        self.torch_bn.train()
        custom_output = self.custom_bn(self.x)
        torch_output = self.torch_bn(self.x_torch)

        # Create random gradient for backward pass
        grad_output = np.random.randn(*torch_output.shape).astype(np.float32)
        grad_output_torch = torch.from_numpy(grad_output).float()

        # Backward pass through custom layer
        custom_grad_input = self.custom_bn.backward(grad_output)

        # Backward pass through PyTorch layer
        torch_output.backward(grad_output_torch)
        torch_grad_input = self.x_torch.grad.numpy()

        # Check if gradients are the same
        self.assertTrue(
            np.allclose(custom_grad_input, torch_grad_input, atol=1e-5),
            "Backward pass gradients do not match!",
        )

class TestLayerNormLayer(unittest.TestCase):
    def setUp(self):
        self.batch_size = 5
        self.num_features = 2
        self.height = 4
        self.width = 4
        self.eps = 1e-5
        self.input_shape = (self.batch_size, self.num_features, self.height, self.width)

        # Initialize custom LayerNorm layer
        self.custom_ln = LayerNorm([self.num_features, self.height, self.width], eps=self.eps)

        # Initialize PyTorch LayerNorm layer
        self.torch_ln = nn.LayerNorm([self.num_features, self.height, self.width], eps=self.eps)
        # Create random input
        self.x = np.random.randn(*self.input_shape).astype(np.float32)
        self.x_torch = torch.from_numpy(self.x).float()
        self.x_torch.requires_grad_(True)  # Ensure that x_torch requires gradients

    def test_forward_pass(self):
        # Forward pass through both layers
        custom_output = self.custom_ln.forward(self.x)
        torch_output = self.torch_ln(self.x_torch).detach().numpy()
        # Check if outputs are the same
        self.assertTrue(
            np.allclose(custom_output, torch_output, atol=1e-5),
            "Forward pass outputs do not match!",
        )

    def test_backward_pass(self):
        # Forward pass through both layers to set input
        custom_output = self.custom_ln.forward(self.x)
        torch_output = self.torch_ln(self.x_torch)

        # Create random gradient for backward pass
        grad_output = np.random.randn(*torch_output.shape).astype(np.float32)
        grad_output_torch = torch.from_numpy(grad_output).float()

        # Backward pass through custom layer
        custom_grad_input = self.custom_ln.backward(grad_output)

        # Backward pass through PyTorch layer
        torch_output.backward(grad_output_torch)
        torch_grad_input = self.x_torch.grad.numpy()

        # Check if gradients are the same
        self.assertTrue(
            np.allclose(
                self.custom_ln._params["G"].grad,
                self.torch_ln.weight.grad.numpy(),
                atol=1e-6,
            ),
            "Weight gradients do not match!",
        )
        self.assertTrue(
            np.allclose(custom_grad_input, torch_grad_input, atol=1e-5),
            "Backward pass gradients do not match!",
        )


class TestLSTM(unittest.TestCase):
    def setUp(self):
        self.input_size = 3
        self.hidden_size = 2
        self.num_layers = 1
        self.batch_size = 1
        self.seq_length = 1

        self.custom_lstm = LSTM(self.input_size, self.hidden_size, self.num_layers)
        self.torch_lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers)

        # Initialize custom LSTM with the same weights as PyTorch LSTM
        self.custom_lstm._params["W_ih_l0"].data[
            :
        ] = self.torch_lstm.weight_ih_l0.detach().numpy()
        self.custom_lstm._params["W_hh_l0"].data[
            :
        ] = self.torch_lstm.weight_hh_l0.detach().numpy()
        self.custom_lstm._params["B_ih_l0"].data[
            :
        ] = self.torch_lstm.bias_ih_l0.detach().numpy()
        self.custom_lstm._params["B_hh_l0"].data[
            :
        ] = self.torch_lstm.bias_hh_l0.detach().numpy()

        self.x = np.random.randn(
            self.batch_size, self.seq_length, self.input_size
        ).astype(np.float32)
        self.h0 = np.random.randn(
            self.num_layers, self.batch_size, self.hidden_size
        ).astype(np.float32)
        self.c0 = np.random.randn(
            self.num_layers, self.batch_size, self.hidden_size
        ).astype(np.float32)

        self.x_torch = torch.from_numpy(self.x).float().requires_grad_(True)
        self.h0_torch = torch.from_numpy(self.h0).float().requires_grad_(True)
        self.c0_torch = torch.from_numpy(self.c0).float().requires_grad_(True)

    def test_forward_pass(self):
        custom_output, custom_hn, custom_cn = self.custom_lstm.forward(
            self.x, self.h0, self.c0
        )
        torch_output, (torch_hn, torch_cn) = self.torch_lstm(
            self.x_torch, (self.h0_torch, self.c0_torch)
        )

        # Check if forward outputs match
        np.testing.assert_allclose(
            custom_output, torch_output.detach().numpy(), atol=1e-6
        )
        np.testing.assert_allclose(custom_hn, torch_hn.detach().numpy(), atol=1e-6)
        np.testing.assert_allclose(custom_cn, torch_cn.detach().numpy(), atol=1e-6)

    # def test_backward_pass(self):
    #     # Forward pass through both layers to set input
    #     custom_output, custom_hn, custom_cn = self.custom_lstm.forward(self.x, self.h0, self.c0)
    #     torch_output, (torch_hn, torch_cn) = self.torch_lstm(self.x_torch, (self.h0_torch, self.c0_torch))

    #     # Create random gradient for backward pass
    #     grad_output = np.random.randn(*custom_output.shape).astype(np.float32)
    #     grad_hn = np.random.randn(*custom_hn.shape).astype(np.float32)
    #     grad_cn = np.random.randn(*custom_cn.shape).astype(np.float32)

    #     grad_output_torch = torch.from_numpy(grad_output).float()

    #     # Backward pass through custom layer
    #     self.custom_lstm.zero_grad()
    #     custom_grad_input, custom_dh0, custom_dc0 = self.custom_lstm.backward(grad_output, grad_hn, grad_cn)

    #     # Backward pass through PyTorch layer
    #     self.x_torch.retain_grad()
    #     torch_output.backward(grad_output_torch, retain_graph=True)
    #     torch_grad_input = self.x_torch.grad.numpy()

    #     # Print gradients for debugging
    #     print("Custom W_ih_l0 grad:\n", self.custom_lstm._params["W_ih_l0"].grad)
    #     print("Torch W_ih_l0 grad:\n", self.torch_lstm.weight_ih_l0.grad.numpy())
    #     print("Custom B_ih_l0 grad:\n", self.custom_lstm._params["B_ih_l0"].grad)
    #     print("Torch B_ih_l0 grad:\n", self.torch_lstm.bias_ih_l0.grad.numpy())
    #     print("Custom W_hh_l0 grad:\n", self.custom_lstm._params["W_hh_l0"].grad)
    #     print("Torch W_hh_l0 grad:\n", self.torch_lstm.weight_hh_l0.grad.numpy())
    #     print("Custom B_hh_l0 grad:\n", self.custom_lstm._params["B_hh_l0"].grad)
    #     print("Torch B_hh_l0 grad:\n", self.torch_lstm.bias_hh_l0.grad.numpy())

    #     # Check if gradients are the same
    #     self.assertTrue(np.allclose(self.custom_lstm._params["W_ih_l0"].grad, self.torch_lstm.weight_ih_l0.grad.numpy(), atol=1e-6), "Input weight gradients do not match!")
    #     self.assertTrue(np.allclose(self.custom_lstm._params["B_ih_l0"].grad, self.torch_lstm.bias_ih_l0.grad.numpy(), atol=1e-6), "Input bias gradients do not match!")
    #     self.assertTrue(np.allclose(self.custom_lstm._params["W_hh_l0"].grad, self.torch_lstm.weight_hh_l0.grad.numpy(), atol=1e-6), "Hidden weight gradients do not match!")
    #     self.assertTrue(np.allclose(self.custom_lstm._params["B_hh_l0"].grad, self.torch_lstm.bias_hh_l0.grad.numpy(), atol=1e-6), "Hidden bias gradients do not match!")


if __name__ == "__main__":
    unittest.main()
