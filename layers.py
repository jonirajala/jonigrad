import numpy as np
from abc import ABC

class Module:
    def __init__(self):
        self._params = {}
    
    def step(self, lr):
        for _, val in self._params.items():
            val._step(lr)
    
    def zero_grad(self):
        for _, val in self._params.items():
            val._zero_grad()


class Parameter(ABC):
    def __init__(self, data, requires_grad=True):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(data, dtype=data.dtype)
     
    def _step(self, lr):
        if self.requires_grad:
            self.data -= self.grad * lr
    
    def _zero_grad(self):
        self.grad.fill(0)
        

class Linear(Module):
    def __init__(self, inp, out):
        super().__init__()
        self._params["W"] = Parameter(np.zeros((out, inp), dtype=np.float32), True) 
        self._params["B"] = Parameter(np.zeros((1, out), dtype=np.float32), True)

        self.__xavier_init()

    def __call__(self, x):
        self.x = x
        # print(x.shape, self.weights.T.shape, self.bias.shape)
        y = x @ self._params["W"].data.T + self._params["B"].data
        return y

    def backward(self, dL_dy):
        # Gradient of the loss with respect to the input of the layer
        # dL_dx = np.dot(self.weights.data.T, dL_dy)
        # print(dL_dy.shape, self.weights.data.T.shape)
        dL_dx = dL_dy @ self._params["W"].data
        
        # Gradient of the loss with respect to the weights
        # print(self.x.T.shape, dL_dy.shape)
        dL_dW = self.x.T @ dL_dy
        
        # Gradient of the loss with respect to the biases
        dL_db = np.sum(dL_dy, axis=0, keepdims=True)
        
        self._params["W"].grad = dL_dW.T
        self._params["B"].grad = dL_db.squeeze()

        return dL_dx



    def __xavier_init(self):
        fan_in, fan_out = self._params["W"].data.shape[1], self._params["W"].data.shape[0]
        variance = 2.0 / (fan_in + fan_out)
        self._params["W"].data = np.random.normal(0.0, np.sqrt(variance), self._params["W"].data.shape)

class Conv(Module):
    def __init__(self, n_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.n_channels = n_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self._params["F"] = Parameter(np.random.rand(out_channels, n_channels, kernel_size, kernel_size).astype(np.float32), True)
   
    
    # def __call__(self, x):
    #     self.x = x.astype(np.float32)

    #     N, H_in, W_in, C_in = x.shape
    #     H_out = int(np.floor((H_in - self.kernel_size + 2 * self.padding) / self.stride) + 1)
    #     W_out = int(np.floor((W_in - self.kernel_size + 2 * self.padding) / self.stride) + 1)
    #     # print(N, H_out, W_out, self.out_channels)
    #     out = np.zeros((N, H_out, W_out, self.out_channels), dtype=np.float32)

    #     self.out_shape = out.shape

    #     if self.padding != 0:
    #         x = np.pad(x, pad_width=((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)

    #     for n in range(N):
    #         for k in range(self.out_channels):
    #             for i in range(H_out):
    #                 for j in range(W_out):
    #                     mat = x[n, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size, :]
    #                     # print(out.shape, mat.shape, self.filters[k, :, :, :].shape)
    #                     out[n, i, j, k] = np.sum(mat * self._params["F"].data[k, :, :, :].transpose(1, 2, 0))
        
        # return out

    # def backward(self, dL_dy):
    #     N, H_in, W_in, C_in = self.x.shape
    #     H_out, W_out = self.out_shape[1], self.out_shape[2]

    #     if self.padding != 0:
    #         self.x = np.pad(self.x, pad_width=((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant', constant_values=0)

    #     dL_dx = np.zeros_like(self.x, dtype=np.float32)
    #     self.dL_dW = np.zeros_like(self._params["F"].data)

    #     for n in range(N):
    #         for k in range(self.out_channels):
    #             for i in range(H_out):
    #                 for j in range(W_out):
    #                     mat = self.x[n, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size, :]
    #                     self.dL_dW[k, :, :, :] += dL_dy[n, i, j, k] * mat.transpose(2, 0, 1)
    #                     dL_dx[n, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size, :] += dL_dy[n, i, j, k] * self._params["F"].data[k, :, :, :].transpose(1, 2, 0)

    #     if self.padding != 0:
    #         dL_dx = dL_dx[:, self.padding:-self.padding, self.padding:-self.padding, :]

    #     return dL_dx
    
    def __call__(self, x):
        self.x = x.astype(np.float32)

        N, H_in, W_in, C_in = x.shape
        H_out = int((H_in - self.kernel_size + 2 * self.padding) / self.stride + 1)
        W_out = int((W_in - self.kernel_size + 2 * self.padding) / self.stride + 1)

        if self.padding != 0:
            x = np.pad(x, pad_width=((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant', constant_values=0)

        # Im2col operation
        cols = np.lib.stride_tricks.as_strided(
            x, 
            shape=(N, H_out, W_out, self.kernel_size, self.kernel_size, C_in),
            strides=(x.strides[0], x.strides[1] * self.stride, x.strides[2] * self.stride, x.strides[1], x.strides[2], x.strides[3]),
            writeable=False
        ).reshape(N * H_out * W_out, -1)

        F_col = self._params["F"].data.reshape(self.out_channels, -1)

        out = np.dot(cols, F_col.T).reshape(N, H_out, W_out, self.out_channels)
        self.out_shape = out.shape

        return out

    def backward(self, dL_dy):
        N, H_in, W_in, C_in = self.x.shape
        H_out, W_out = self.out_shape[1], self.out_shape[2]

        if self.padding != 0:
            self.x = np.pad(self.x, pad_width=((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant', constant_values=0)

        dL_dx = np.zeros_like(self.x, dtype=np.float32)
        self.dL_dW = np.zeros_like(self._params["F"].data)

        cols = np.lib.stride_tricks.as_strided(
            self.x, 
            shape=(N, H_out, W_out, self.kernel_size, self.kernel_size, C_in),
            strides=(self.x.strides[0], self.x.strides[1] * self.stride, self.x.strides[2] * self.stride, self.x.strides[1], self.x.strides[2], self.x.strides[3]),
            writeable=False
        ).reshape(N * H_out * W_out, -1)

        dL_dy_reshaped = dL_dy.transpose(0, 3, 1, 2).reshape(-1, self.out_channels)

        # Gradient with respect to weights
        self.dL_dW = np.dot(dL_dy_reshaped.T, cols).reshape(self._params["F"].data.shape)

        # Gradient with respect to input
        F_reshaped = self._params["F"].data.reshape(self.out_channels, -1)
        dL_dcol = np.dot(dL_dy_reshaped, F_reshaped).reshape(N, H_out, W_out, self.kernel_size, self.kernel_size, C_in)

        for i in range(H_out):
            for j in range(W_out):
                dL_dx[:, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size, :] += dL_dcol[:, i, j, :, :, :]

        if self.padding != 0:
            dL_dx = dL_dx[:, self.padding:-self.padding, self.padding:-self.padding, :]

        return dL_dx

class MSELoss:
    def __call__(self, preds, targs):
        self.preds = preds
        self.targs = targs

        loss = np.mean(np.square(preds - targs), dtype=np.float32)
        return loss

    def backward(self):
        dL_dy = (2 / self.preds.size) * (self.preds - self.targs)
        self.dL_dy = dL_dy
        return dL_dy

class CrossEntropyLoss:
    def __call__(self, preds, targs):
        self.targs = targs
        
        # Compute softmax probabilities from predictions
        exp_logits = np.exp(preds - np.max(preds, axis=1, keepdims=True))
        y_pred = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Store the predicted probabilities for backward pass
        self.y_pred = y_pred
        
        # Ensure y_pred is clipped to avoid log(0) which is undefined
        y_pred = np.clip(y_pred, 1e-12, 1. - 1e-12)
        
        # If targs is given as class indices, convert to one-hot encoding
        if targs.ndim == 1:
            targs = np.eye(y_pred.shape[1])[targs]

        # Calculate cross-entropy loss
        loss = -np.sum(targs * np.log(y_pred)) / targs.shape[0]
        return loss.astype(np.float32)

    def backward(self):
        if self.targs.ndim == 1:
            targs_one_hot = np.eye(self.y_pred.shape[1])[self.targs]
        else:
            targs_one_hot = self.targs
        
        # Gradient of the loss w.r.t. the logits
        dL_dy = (self.y_pred - targs_one_hot) / targs_one_hot.shape[0]
        return dL_dy
    

class ReLU(Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        self.x = x
        self.y = np.maximum(0, x)
        return self.y
    
    
    def backward(self, dL_dy):
        if dL_dy.shape != self.x.shape:
            dL_dy = dL_dy.reshape(self.x.shape)
        dL_dx = dL_dy * (self.x > 0).astype(float)
        self.dL_dx = dL_dx
        return self.dL_dx
