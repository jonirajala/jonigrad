import unittest
import numpy as np
import torch
import torch.nn as nn

from jonigrad.layers import Layer, Parameter, Sigmoid, Tanh
import numpy as np

class LSTM(Layer):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Initialize parameters for the first LSTM layer
        self._params["W_ih_l0"] = Parameter(np.random.randn(4 * hidden_size, input_size).astype(np.float32), True)
        self._params["B_ih_l0"] = Parameter(np.zeros((4 * hidden_size), dtype=np.float32), True)
        self._params["W_hh_l0"] = Parameter(np.random.randn(4 * hidden_size, hidden_size).astype(np.float32), True)
        self._params["B_hh_l0"] = Parameter(np.zeros((4 * hidden_size), dtype=np.float32), True)

        self.inp_gate_sigmoid = Sigmoid()
        self.forg_gate_sigmoid = Sigmoid()
        self.cell_gate_tanh = Tanh()
        self.outp_gate_sigmoid = Sigmoid()

    def forward(self, input_tensor, h0, c0):
        batch_size, seq_length, _ = input_tensor.shape
        h = h0.squeeze(0)
        c = c0.squeeze(0)
        outputs = []
        self.cache = []  # To store intermediate values for backward pass

        for t in range(seq_length):
            x_t = input_tensor[:, t, :]

            # Forget gate
            ft = self.forg_gate_sigmoid(
                np.dot(x_t, self._params["W_ih_l0"].data[self.hidden_size:2 * self.hidden_size, :].T) +
                np.dot(h, self._params["W_hh_l0"].data[self.hidden_size:2 * self.hidden_size, :].T) +
                self._params["B_ih_l0"].data[self.hidden_size:2 * self.hidden_size] +
                self._params["B_hh_l0"].data[self.hidden_size:2 * self.hidden_size]
            )

            # Input gate
            it = self.inp_gate_sigmoid(
                np.dot(x_t, self._params["W_ih_l0"].data[:self.hidden_size, :].T) +
                np.dot(h, self._params["W_hh_l0"].data[:self.hidden_size, :].T) +
                self._params["B_ih_l0"].data[:self.hidden_size] +
                self._params["B_hh_l0"].data[:self.hidden_size]
            )

            # Candidate cell state
            c_tilde = self.cell_gate_tanh(
                np.dot(x_t, self._params["W_ih_l0"].data[2 * self.hidden_size:3 * self.hidden_size, :].T) +
                np.dot(h, self._params["W_hh_l0"].data[2 * self.hidden_size:3 * self.hidden_size, :].T) +
                self._params["B_ih_l0"].data[2 * self.hidden_size:3 * self.hidden_size] +
                self._params["B_hh_l0"].data[2 * self.hidden_size:3 * self.hidden_size]
            )

            # Update cell state
            c = ft * c + it * c_tilde

            # Output gate
            ot = self.outp_gate_sigmoid(
                np.dot(x_t, self._params["W_ih_l0"].data[3 * self.hidden_size:, :].T) +
                np.dot(h, self._params["W_hh_l0"].data[3 * self.hidden_size:, :].T) +
                self._params["B_ih_l0"].data[3 * self.hidden_size:] +
                self._params["B_hh_l0"].data[3 * self.hidden_size:]
            )

            # Update hidden state
            h = ot * self.cell_gate_tanh(c)
            outputs.append(h)
            self.cache.append((x_t, h, c, ft, it, c_tilde, ot))

        outputs = np.stack(outputs, axis=1)  # Shape should be (batch_size, seq_length, hidden_size)
        return outputs, h[np.newaxis, :, :], c[np.newaxis, :, :]

    
    def backward(self, grad_output, grad_hn, grad_cn):
        grad_hn = grad_hn.squeeze(0)
        grad_cn = grad_cn.squeeze(0)

        dW_ih_l0 = np.zeros_like(self._params["W_ih_l0"].data)
        dB_ih_l0 = np.zeros_like(self._params["B_ih_l0"].data)
        dW_hh_l0 = np.zeros_like(self._params["W_hh_l0"].data)
        dB_hh_l0 = np.zeros_like(self._params["B_hh_l0"].data)

        dh_next = grad_hn
        dc_next = grad_cn
        dL_dx = np.zeros((grad_output.shape[0], grad_output.shape[1], self.input_size))

        for t in reversed(range(len(self.cache))):
            x_t, h, c, ft, it, c_tilde, ot = self.cache[t]

            # Calculate gradients for the output gate
            do = self.outp_gate_sigmoid.backward((grad_output[:, t, :] + dh_next) * self.cell_gate_tanh.forward(c))

            # Calculate gradients for the cell state
            dc = (grad_output[:, t, :] * ot * (1 - np.tanh(c) ** 2)) + dc_next + (dh_next * ot * (1 - np.tanh(c) ** 2))

            # Calculate gradients for the input gate
            di = self.inp_gate_sigmoid.backward(dc * c_tilde)

            # Calculate gradients for the forget gate
            df = self.forg_gate_sigmoid.backward(dc * c)

            # Calculate gradients for the candidate cell state
            dc_tilde = self.cell_gate_tanh.backward(dc * it)

            # Update gradients for the input weights
            dW_ih_l0[:self.hidden_size] += np.dot(di.T, x_t)
            dW_ih_l0[self.hidden_size:2*self.hidden_size] += np.dot(df.T, x_t)
            dW_ih_l0[2*self.hidden_size:3*self.hidden_size] += np.dot(dc_tilde.T, x_t)
            dW_ih_l0[3*self.hidden_size:] += np.dot(do.T, x_t)

            dB_ih_l0[:self.hidden_size] += np.sum(di, axis=0)
            dB_ih_l0[self.hidden_size:2*self.hidden_size] += np.sum(df, axis=0)
            dB_ih_l0[2*self.hidden_size:3*self.hidden_size] += np.sum(dc_tilde, axis=0)
            dB_ih_l0[3*self.hidden_size:] += np.sum(do, axis=0)

            # Update gradients for the hidden weights
            dW_hh_l0[:self.hidden_size] += np.dot(di.T, h)
            dW_hh_l0[self.hidden_size:2*self.hidden_size] += np.dot(df.T, h)
            dW_hh_l0[2*self.hidden_size:3*self.hidden_size] += np.dot(dc_tilde.T, h)
            dW_hh_l0[3*self.hidden_size:] += np.dot(do.T, h)

            dB_hh_l0[:self.hidden_size] += np.sum(di, axis=0)
            dB_hh_l0[self.hidden_size:2*self.hidden_size] += np.sum(df, axis=0)
            dB_hh_l0[2*self.hidden_size:3*self.hidden_size] += np.sum(dc_tilde, axis=0)
            dB_hh_l0[3*self.hidden_size:] += np.sum(do, axis=0)

            # Calculate gradient with respect to input x_t
            dx_t = np.dot(di, self._params["W_ih_l0"].data[:self.hidden_size, :]) + \
                np.dot(df, self._params["W_ih_l0"].data[self.hidden_size:2*self.hidden_size, :]) + \
                np.dot(dc_tilde, self._params["W_ih_l0"].data[2*self.hidden_size:3*self.hidden_size, :]) + \
                np.dot(do, self._params["W_ih_l0"].data[3*self.hidden_size:, :])
            dL_dx[:, t, :] = dx_t

            # Propagate gradients to the previous time step
            dh_next = np.dot(di, self._params["W_hh_l0"].data[:self.hidden_size, :].T) + \
                    np.dot(df, self._params["W_hh_l0"].data[self.hidden_size:2*self.hidden_size, :].T) + \
                    np.dot(dc_tilde, self._params["W_hh_l0"].data[2*self.hidden_size:3*self.hidden_size, :].T) + \
                    np.dot(do, self._params["W_hh_l0"].data[3*self.hidden_size:, :].T)

            dc_next = dc * ft

        self._params["W_ih_l0"].grad = dW_ih_l0
        self._params["B_ih_l0"].grad = dB_ih_l0
        self._params["W_hh_l0"].grad = dW_hh_l0
        self._params["B_hh_l0"].grad = dB_hh_l0

        return dL_dx, dh_next[np.newaxis, :, :], dc_next[np.newaxis, :, :]




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
        self.custom_lstm._params["W_ih_l0"].data[:] = self.torch_lstm.weight_ih_l0.detach().numpy()
        self.custom_lstm._params["W_hh_l0"].data[:] = self.torch_lstm.weight_hh_l0.detach().numpy()
        self.custom_lstm._params["B_ih_l0"].data[:] = self.torch_lstm.bias_ih_l0.detach().numpy()
        self.custom_lstm._params["B_hh_l0"].data[:] = self.torch_lstm.bias_hh_l0.detach().numpy()


        self.x = np.random.randn(self.batch_size, self.seq_length, self.input_size).astype(np.float32)
        self.h0 = np.random.randn(self.num_layers, self.batch_size, self.hidden_size).astype(np.float32)
        self.c0 = np.random.randn(self.num_layers, self.batch_size, self.hidden_size).astype(np.float32)

        self.x_torch = torch.from_numpy(self.x).float().requires_grad_(True)
        self.h0_torch = torch.from_numpy(self.h0).float().requires_grad_(True)
        self.c0_torch = torch.from_numpy(self.c0).float().requires_grad_(True)

    def test_forward_pass(self):
        custom_output, custom_hn, custom_cn = self.custom_lstm.forward(self.x, self.h0, self.c0)
        torch_output, (torch_hn, torch_cn) = self.torch_lstm(self.x_torch, (self.h0_torch, self.c0_torch))

        # Check if forward outputs match
        np.testing.assert_allclose(custom_output, torch_output.detach().numpy(), atol=1e-6)
        np.testing.assert_allclose(custom_hn, torch_hn.detach().numpy(), atol=1e-6)
        np.testing.assert_allclose(custom_cn, torch_cn.detach().numpy(), atol=1e-6)

    def test_backward_pass(self):
        # Forward pass through both layers to set input
        custom_output, custom_hn, custom_cn = self.custom_lstm.forward(self.x, self.h0, self.c0)
        torch_output, (torch_hn, torch_cn) = self.torch_lstm(self.x_torch, (self.h0_torch, self.c0_torch))

        # Create random gradient for backward pass
        grad_output = np.random.randn(*custom_output.shape).astype(np.float32)
        grad_hn = np.random.randn(*custom_hn.shape).astype(np.float32)
        grad_cn = np.random.randn(*custom_cn.shape).astype(np.float32)

        grad_output_torch = torch.from_numpy(grad_output).float()

        # Backward pass through custom layer
        self.custom_lstm.zero_grad()
        custom_grad_input, custom_dh0, custom_dc0 = self.custom_lstm.backward(grad_output, grad_hn, grad_cn)

        # Backward pass through PyTorch layer
        self.x_torch.retain_grad()
        torch_output.backward(grad_output_torch, retain_graph=True)
        torch_grad_input = self.x_torch.grad.numpy()

        # Print gradients for debugging
        print("Custom W_ih_l0 grad:\n", self.custom_lstm._params["W_ih_l0"].grad)
        print("Torch W_ih_l0 grad:\n", self.torch_lstm.weight_ih_l0.grad.numpy())
        print("Custom B_ih_l0 grad:\n", self.custom_lstm._params["B_ih_l0"].grad)
        print("Torch B_ih_l0 grad:\n", self.torch_lstm.bias_ih_l0.grad.numpy())
        print("Custom W_hh_l0 grad:\n", self.custom_lstm._params["W_hh_l0"].grad)
        print("Torch W_hh_l0 grad:\n", self.torch_lstm.weight_hh_l0.grad.numpy())
        print("Custom B_hh_l0 grad:\n", self.custom_lstm._params["B_hh_l0"].grad)
        print("Torch B_hh_l0 grad:\n", self.torch_lstm.bias_hh_l0.grad.numpy())

        # Check if gradients are the same
        self.assertTrue(np.allclose(self.custom_lstm._params["W_ih_l0"].grad, self.torch_lstm.weight_ih_l0.grad.numpy(), atol=1e-6), "Input weight gradients do not match!")
        self.assertTrue(np.allclose(self.custom_lstm._params["B_ih_l0"].grad, self.torch_lstm.bias_ih_l0.grad.numpy(), atol=1e-6), "Input bias gradients do not match!")
        self.assertTrue(np.allclose(self.custom_lstm._params["W_hh_l0"].grad, self.torch_lstm.weight_hh_l0.grad.numpy(), atol=1e-6), "Hidden weight gradients do not match!")
        self.assertTrue(np.allclose(self.custom_lstm._params["B_hh_l0"].grad, self.torch_lstm.bias_hh_l0.grad.numpy(), atol=1e-6), "Hidden bias gradients do not match!")

if __name__ == '__main__':
    unittest.main()