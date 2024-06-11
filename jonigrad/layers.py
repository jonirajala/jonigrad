import numpy as np
from abc import ABC, abstractmethod


class Module:
    def __init__(self):
        self._training = True

    def __call__(self, *inputs):
        return self.forward(*inputs)

    def step(self, lr):
        if not hasattr(self, 'layers'):
            self.get_layers()
        for layer in self.layers:
            layer.step(lr)

    def zero_grad(self):
        if not hasattr(self, 'layers'):
            self.get_layers()
        for layer in self.layers:
            layer.zero_grad()

    def train(self):
        self._training = True
        self.get_layers()
        for layer in self.layers:
            layer.train()

    def eval(self):
        self._training = False
        for layer in self.layers:
            layer.eval()

    def clip_grad(self, threshold, batch_size):
        if not hasattr(self, 'layers'):
            self.get_layers()
        for layer in self.layers:
            layer.clip_grad(threshold, batch_size)

    def get_layers(self):
        layers = []
        for attr_name in dir(self):
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, Layer):
                layers.append(attr_value)
            elif isinstance(attr_value, Module) and attr_value != self:
                layers.extend(attr_value.get_layers())
            elif isinstance(attr_value, Sequential) and attr_value != self:
                layers.extend(attr_value.get_layers())
        self.layers = layers
        return layers
    
    def parameter_count(self):
        if not hasattr(self, 'layers'):
            self.get_layers()
        n = 0
        for layer in self.layers:
            n += layer.parameter_count()
        return n

    @abstractmethod
    def forward(self, *inputs):
        raise NotImplementedError

class Sequential:
    def __init__(self, layers=[]):
        self.seq_layers = layers

    def get_layers(self):
        layers = []
        for attr in self.seq_layers:
            if isinstance(attr, Layer):
                layers.append(attr)
            elif isinstance(attr, Module) and attr != self:
                layers.extend(attr.get_layers())
            elif isinstance(attr, Sequential) and attr != self:
                layers.extend(attr.get_layers())
        self.layers = layers
        return layers

    def __getitem__(self, idx):
        if idx >= len(self.seq_layers):
            raise IndexError(f'Index {idx} is out of bounds for the number of layers {self.seq_layers}')
        return self.seq_layers[idx]

class Layer:
    def __init__(self):
        self._params = {}
        self._training = True

    def __call__(self, *inputs):
        return self.forward(*inputs)

    def step(self, lr):
        for _, val in self._params.items():
            val._step(lr)

    def zero_grad(self):
        for _, val in self._params.items():
            val._zero_grad()

    def clip_grad(self, threshold, batch_size):
        for _, val in self._params.items():
            val._clip_grad(threshold, batch_size)

    def train(self):
        self._training = True

    def eval(self):
        self._training = False

    def parameter_count(self):
        n = 0
        for _, val in self._params.items():
            n += val._parameter_count()
        return n

    @abstractmethod
    def forward(self, *inputs):
        raise NotImplementedError


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

    def _clip_grad(self, threshold, batch_size):
        if self.requires_grad:
            gradient = self.grad / batch_size
            gradient_norm = np.linalg.norm(gradient)
            if gradient_norm > threshold:
                self.grad = (threshold / gradient_norm) * self.grad
    
    def _parameter_count(self):
        return self.grad.size


class Linear(Layer):
    def __init__(self, in_features, out_features):
        super().__init__()
        self._params["W"] = Parameter(
            np.zeros((out_features, in_features), dtype=np.float32), True
        )
        self._params["B"] = Parameter(
            np.zeros((1, out_features), dtype=np.float32), True
        )
        self.__xavier_init()

    def forward(self, x):
        self.x_shape = x.shape
        batch_size, *dims, in_features = x.shape
        x = x.reshape(-1, in_features)  # Flatten all but the last dimension
        self.x = x  # Store reshaped x for backward pass
        y = x @ self._params["W"].data.T + self._params["B"].data
        return y.reshape(batch_size, *dims, -1)

    def backward(self, dL_dy):
        batch_size, *dims, out_features = dL_dy.shape
        dL_dy = dL_dy.reshape(-1, out_features)  # Flatten all but the last dimension
        dL_dx = dL_dy @ self._params["W"].data
        dL_dW = dL_dy.T @ self.x
        dL_db = np.sum(dL_dy, axis=0, keepdims=True)

        self._params["W"].grad = dL_dW
        self._params["B"].grad = dL_db

        dL_dx = dL_dx.reshape(batch_size, *dims, -1)  # Reshape back to input shape
        return dL_dx

    def __xavier_init(self):
        fan_in, fan_out = self._params["W"].data.shape[1], self._params["W"].data.shape[0]
        variance = 2.0 / (fan_in + fan_out)
        self._params["W"].data = np.random.normal(0.0, np.sqrt(variance), self._params["W"].data.shape)



class Conv(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self._params["F"] = Parameter(
            np.random.rand(out_channels, in_channels, kernel_size, kernel_size).astype(
                np.float32
            ),
            True,
        )

    # def forward(self, x):
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

    def forward(self, x):
        self.x = x.astype(np.float32)

        N, C_in, H_in, W_in = x.shape
        H_out = int((H_in - self.kernel_size + 2 * self.padding) / self.stride + 1)
        W_out = int((W_in - self.kernel_size + 2 * self.padding) / self.stride + 1)

        if self.padding != 0:
            x = np.pad(
                x,
                pad_width=(
                    (0, 0),
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                ),
                mode="constant",
                constant_values=0,
            )

        # Im2col operation
        cols = (
            np.lib.stride_tricks.as_strided(
                x,
                shape=(N, C_in, self.kernel_size, self.kernel_size, H_out, W_out),
                strides=(
                    x.strides[0],
                    x.strides[1],
                    x.strides[2],
                    x.strides[3],
                    x.strides[2] * self.stride,
                    x.strides[3] * self.stride,
                ),
                writeable=False,
            )
            .transpose(0, 4, 5, 1, 2, 3)
            .reshape(N * H_out * W_out, -1)
        )

        F_col = self._params["F"].data.reshape(self.out_channels, -1)

        out = (
            np.dot(cols, F_col.T)
            .reshape(N, H_out, W_out, self.out_channels)
            .transpose(0, 3, 1, 2)
        )
        self.out_shape = out.shape

        return out

    def backward(self, dL_dy):
        N, C_in, H_in, W_in = self.x.shape
        H_out, W_out = self.out_shape[2], self.out_shape[3]

        if self.padding != 0:
            self.x = np.pad(
                self.x,
                pad_width=(
                    (0, 0),
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                ),
                mode="constant",
                constant_values=0,
            )

        dL_dx = np.zeros_like(self.x, dtype=np.float32)

        cols = (
            np.lib.stride_tricks.as_strided(
                self.x,
                shape=(N, C_in, self.kernel_size, self.kernel_size, H_out, W_out),
                strides=(
                    self.x.strides[0],
                    self.x.strides[1],
                    self.x.strides[2],
                    self.x.strides[3],
                    self.x.strides[2] * self.stride,
                    self.x.strides[3] * self.stride,
                ),
                writeable=False,
            )
            .transpose(0, 4, 5, 1, 2, 3)
            .reshape(N * H_out * W_out, -1)
        )

        dL_dy_reshaped = dL_dy.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

        # Gradient with respect to weights
        self._params["F"].grad = np.dot(dL_dy_reshaped.T, cols).reshape(
            self._params["F"].data.shape
        )

        # Gradient with respect to input
        F_reshaped = self._params["F"].data.reshape(self.out_channels, -1)
        dL_dcol = (
            np.dot(dL_dy_reshaped, F_reshaped)
            .reshape(N, H_out, W_out, C_in, self.kernel_size, self.kernel_size)
            .transpose(0, 3, 4, 5, 1, 2)
        )

        for i in range(H_out):
            for j in range(W_out):
                dL_dx[
                    :,
                    :,
                    i * self.stride : i * self.stride + self.kernel_size,
                    j * self.stride : j * self.stride + self.kernel_size,
                ] += dL_dcol[:, :, :, :, i, j]

        if self.padding != 0:
            dL_dx = dL_dx[
                :, :, self.padding : -self.padding, self.padding : -self.padding
            ]

        return dL_dx


class MSELoss:
    def __init__(self):
        super().__init__()

    def __call__(self, preds, targs):
        self.preds = preds
        self.targs = targs

        loss = np.mean(np.square(preds - targs), dtype=np.float32)
        return loss

    def backward(self):
        dL_dy = (2 / self.preds.size) * (self.preds - self.targs)
        self.dL_dy = dL_dy
        return dL_dy

class Softmax:
    def __init__(self):
        super().__init__()

    def __call__(self, x, dim=-1):
        exp_x = np.exp(x - np.max(x, axis=dim, keepdims=True))  # Subtract max for numerical stability
        self.y = exp_x / np.sum(exp_x, axis=dim, keepdims=True)
        self.y = self.y.astype(np.float32)
        return self.y

    def backward(self, dL_dy):
        # y = self.y.reshape(-1, 1)

        # # Compute the Jacobian matrix of the softmax function
        # jacobian_matrix = np.diagflat(y) - np.dot(y, y.T)
        
        # # Compute the gradient of the loss with respect to the input x
        # dL_dx = np.dot(jacobian_matrix, dL_dy)
        
        # return dL_dx
        y = self.y
        dL_dx = y * (dL_dy - np.sum(dL_dy * y, axis=-1, keepdims=True))
        return dL_dx


class CrossEntropyLoss:
    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index

    def __call__(self, preds, targs):
        self.targs = targs
        exp_logits = np.exp(preds - np.max(preds, axis=-1, keepdims=True))
        y_pred = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        y_pred = np.clip(y_pred, 1e-9, 1.0 - 1e-9)
        self.y_pred = y_pred

        if targs.ndim <= 2:
            targs_one_hot = np.eye(y_pred.shape[-1])[targs]
        else:
            targs_one_hot = targs

        mask = targs != self.ignore_index
        targs_one_hot[~mask] = 0
        y_pred[~mask] = 1e-9
        loss = -np.sum(targs_one_hot * np.log(y_pred)) / np.sum(mask)
        return loss.astype(np.float32)

    def backward(self):
        # Creating one-hot encoding if targets are not already in that form
        if self.targs.ndim <= 2:
            targs_one_hot = np.eye(self.y_pred.shape[-1])[self.targs]
        else:
            targs_one_hot = self.targs

        # Creating a mask for valid entries (not ignore_index)
        mask = self.targs != self.ignore_index
        targs_one_hot[~mask] = 0

        # Calculating the gradient of the loss with respect to the outputs
        dL_dy = self.y_pred - targs_one_hot

        # Applying the mask to zero out gradients for ignored indices
        dL_dy[~mask] = 0

        # Normalizing the gradient by the number of non-ignored samples
        dL_dy /= np.sum(mask)

        return dL_dy.astype(np.float32)


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        self.x = None

    def forward(self, x):
        self.x = x
        y = 1 / (1 + np.exp(-x))
        return y

    def backward(self, dL_dy):
        sigmoid_x = 1 / (1 + np.exp(-self.x))
        dL_dx = dL_dy * sigmoid_x * (1 - sigmoid_x)
        return dL_dx


class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.x = None

    def forward(self, x):
        self.x = x
        y = np.maximum(0, x)
        return y

    def backward(self, dL_dy):
        if dL_dy.shape != self.x.shape:
            dL_dy = dL_dy.reshape(self.x.shape)
        dL_dx = dL_dy * (self.x > 0).astype(float)
        self.dL_dx = dL_dx
        return self.dL_dx


class Tanh(Layer):
    def __init__(self):
        super().__init__()
        self.x = None

    def forward(self, x):
        self.x = x
        y = np.tanh(x)
        return y

    def backward(self, dL_dy):
        tanh_x = np.tanh(self.x)
        dL_dx = dL_dy * (1 - np.power(tanh_x, 2))
        return dL_dx


class LocalResponseNorm(Layer):
    def __init__(self, size, alpha=1e-4, beta=0.75, k=2.0):
        super().__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x):
        self.x = x
        self.norm = np.zeros_like(x)
        N, C, H, W = x.shape
        for i in range(C):
            start = max(0, i - self.size // 2)
            end = min(C, i + self.size // 2 + 1)
            self.norm[:, i, :, :] = np.sum(x[:, start:end, :, :] ** 2, axis=1)
        self.norm = (self.k + (self.alpha / self.size) * self.norm) ** self.beta
        return x / self.norm


    def backward(self, dL_dy):
        N, C, H, W = self.x.shape
        dx = np.zeros_like(self.x)

        for i in range(C):
            start = max(0, i - self.size // 2)
            end = min(C, i + self.size // 2 + 1)
            for j in range(start, end):
                if j != i:
                    dx[:, i, :, :] += (
                        -2
                        * self.alpha
                        * self.beta
                        / self.size
                        * self.x[:, i, :, :]
                        * self.x[:, j, :, :]
                        * dL_dy[:, j, :, :]
                        / np.power(self.norm[:, j, :, :], self.beta + 1)
                    )
                else:
                    dx[:, i, :, :] += dL_dy[:, i, :, :] / self.norm[:, i, :, :]
                    dx[:, i, :, :] += (
                        -2
                        * self.alpha
                        * self.beta
                        / self.size
                        * self.x[:, i, :, :]
                        * self.x[:, i, :, :]
                        * dL_dy[:, i, :, :]
                        / np.power(self.norm[:, i, :, :], self.beta + 1)
                    )
        return dx

    # N, C, H, W = self.x.shape
    # dx = np.zeros_like(self.x)

    # for i in range(C):
    #     start = max(0, i - self.size // 2)
    #     end = min(C, i + self.size // 2 + 1)

    #     for j in range(start, end):
    #         if j != i:
    #             dx[:, i, :, :] += (-2 * self.alpha * self.beta / self.size *
    #                                self.x[:, j, :, :] *
    #                                dL_dy[:, j, :, :] / (self.norm[:, j, :, :] ** (self.beta + 1)))
    #         else:
    #             dx[:, i, :, :] += dL_dy[:, i, :, :] / self.norm[:, i, :, :]
    #             dx[:, i, :, :] += (-2 * self.alpha * self.beta / self.size *
    #                                self.x[:, i, :, :] *
    #                                dL_dy[:, i, :, :] / (self.norm[:, i, :, :] ** (self.beta + 1)))
    # return dx


class MaxPool(Layer):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        out_H = (H - self.kernel_size) // self.stride + 1
        out_W = (W - self.kernel_size) // self.stride + 1

        out = np.zeros((N, C, out_H, out_W))
        for n in range(N):
            for k in range(C):
                for i in range(out_H):
                    for j in range(out_W):
                        window = x[
                            n,
                            k,
                            i * self.stride : i * self.stride + self.kernel_size,
                            j * self.stride : j * self.stride + self.kernel_size,
                        ]
                        max = np.amax(window)
                        out[n, k, i, j] = max

        return out

    def backward(self, dL_dy):
        N, C, H, W = self.x.shape
        dX = np.zeros_like(self.x)

        out_H = (H - self.kernel_size) // self.stride + 1
        out_W = (W - self.kernel_size) // self.stride + 1

        for n in range(N):
            for k in range(C):
                for i in range(out_H):
                    for j in range(out_W):
                        window = self.x[
                            n,
                            k,
                            i * self.stride : i * self.stride + self.kernel_size,
                            j * self.stride : j * self.stride + self.kernel_size,
                        ]
                        max_val = np.max(window)
                        for m in range(self.kernel_size):
                            for l in range(self.kernel_size):
                                if window[m, l] == max_val:
                                    dX[
                                        n, k, i * self.stride + m, j * self.stride + l
                                    ] += dL_dy[n, k, i, j]

        return dX


class AvgPool(Layer):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        out_H = (H - self.kernel_size) // self.stride + 1
        out_W = (W - self.kernel_size) // self.stride + 1

        out = np.zeros((N, C, out_H, out_W))
        for n in range(N):
            for k in range(C):
                for i in range(out_H):
                    for j in range(out_W):
                        window = x[
                            n,
                            k,
                            i * self.stride : i * self.stride + self.kernel_size,
                            j * self.stride : j * self.stride + self.kernel_size,
                        ]
                        avg = np.mean(window)
                        out[n, k, i, j] = avg

        return out

    def backward(self, dL_dy):
        N, C, H, W = self.x.shape
        dX = np.zeros_like(self.x)

        out_H = (H - self.kernel_size) // self.stride + 1
        out_W = (W - self.kernel_size) // self.stride + 1

        for n in range(N):
            for k in range(C):
                for i in range(out_H):
                    for j in range(out_W):
                        window = self.x[
                            n,
                            k,
                            i * self.stride : i * self.stride + self.kernel_size,
                            j * self.stride : j * self.stride + self.kernel_size,
                        ]
                        avg_grad = dL_dy[n, k, i, j] / (
                            self.kernel_size * self.kernel_size
                        )
                        for m in range(self.kernel_size):
                            for l in range(self.kernel_size):
                                dX[
                                    n, k, i * self.stride + m, j * self.stride + l
                                ] += avg_grad

        return dX


class Dropout(Layer):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = None

    def forward(self, x):
        if self._training:
            self.mask = np.random.binomial(1, 1 - self.p, size=x.shape).astype(
                np.float32
            )
            x = x * self.mask
            x = x / (1 - self.p)

        return x

    def backward(self, dL_dy):
        if self._training:
            dL_dx = dL_dy * self.mask
            dL_dx = dL_dx / (1 - self.p)
        else:
            dL_dx = dL_dy
        return dL_dx


class Flatten(Layer):
    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dL_dy):
        # Reshape the gradient to the shape of the input during the forward pass
        dl_dx = dL_dy.reshape(self.input_shape)
        return dl_dx

class LayerNorm(Layer):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self._params["G"] = Parameter(np.ones(normalized_shape), True)
        self._params["B"] = Parameter(np.zeros(normalized_shape), True)

    def forward(self, x):
        self.x = x
        # print("normshape", self.normalized_shape)
        axes = tuple(range(x.ndim-len(self.normalized_shape), x.ndim))
        self.mean = np.mean(x, axis=axes, keepdims=True)
        self.var = np.var(x, axis=axes, keepdims=True)
        self.x_normalized = (x - self.mean) / np.sqrt(self.var + self.eps)
        y = self._params["G"].data * self.x_normalized + self._params["B"].data
        return y
    
    def backward(self, dL_dy):
        axes_to_norm = tuple(range(dL_dy.ndim-len(self.normalized_shape), dL_dy.ndim))
        other_axes = tuple(range(0, dL_dy.ndim-len(self.normalized_shape)))
        # print("other axes", other_axes)
        # Gradients of gamma (G) and beta (B)
        # print("inputshape to lr", dL_dy.shape)
        dL_dG = np.sum(dL_dy * self.x_normalized, axis=other_axes, keepdims=True)
        dL_dB = np.sum(dL_dy, axis=other_axes, keepdims=True)

        # Gradient of the input
        dL_dx_normalized = dL_dy * self._params["G"].data
        dL_dvar = np.sum(dL_dx_normalized * (self.x - self.mean) * -0.5 * np.power(self.var + self.eps, -1.5), axis=axes_to_norm, keepdims=True)
        dL_dmean = np.sum(dL_dx_normalized * -1 / np.sqrt(self.var + self.eps), axis=axes_to_norm, keepdims=True) + dL_dvar * np.sum(-2 * (self.x - self.mean), axis=axes_to_norm, keepdims=True) / np.prod(self.normalized_shape)
        
        dL_dx = (dL_dx_normalized / np.sqrt(self.var + self.eps)) + (dL_dvar * 2 * (self.x - self.mean) / np.prod(self.normalized_shape)) + (dL_dmean / np.prod(self.normalized_shape))

        # print(dL_dG.squeeze().shape, self.normalized_shape)
        # assert list(dL_dG.squeeze().shape) == self.normalized_shape
        # assert list(dL_dB.squeeze().shape) == self.normalized_shape
        

        self._params["G"].grad = dL_dG.squeeze()
        self._params["B"].grad = dL_dB.squeeze()

        return dL_dx

class BatchNorm(Layer):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.num_features = num_features
        self.momentum = momentum
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self._params["G"] = Parameter(np.ones(num_features), True)
        self._params["B"] = Parameter(np.zeros(num_features), True)

    def forward(self, x):
        assert len(x.shape) == 4
        self.x = x  # Store x for backward pass
        if self._training:
            self.batch_mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            self.batch_var = np.var(x, axis=(0, 2, 3), keepdims=True)
            self.x_norm = (x - self.batch_mean) / np.sqrt(self.batch_var + self.eps)
            self.running_mean = (
                self.momentum * np.squeeze(self.batch_mean)
                + (1 - self.momentum) * self.running_mean
            )
            self.running_var = (
                self.momentum * np.squeeze(self.batch_var)
                + (1 - self.momentum) * self.running_var
            )
        else:
            self.x_norm = (x - self.running_mean.reshape(1, -1, 1, 1)) / np.sqrt(
                self.running_var.reshape(1, -1, 1, 1) + self.eps
            )

        out = self._params["G"].data.reshape(1, -1, 1, 1) * self.x_norm + self._params[
            "B"
        ].data.reshape(1, -1, 1, 1)
        return out

    def backward(self, dL_dy):
        N, C, H, W = dL_dy.shape
        batch_var_eps = self.batch_var + self.eps

        dL_dgamma = np.sum(dL_dy * self.x_norm, axis=(0, 2, 3))
        dL_dbeta = np.sum(dL_dy, axis=(0, 2, 3))

        dL_dx_norm = dL_dy * self._params["G"].data.reshape(1, -1, 1, 1)

        dL_dvar = np.sum(
            dL_dx_norm
            * (self.x - self.batch_mean)
            * -0.5
            * np.power(batch_var_eps, -1.5),
            axis=(0, 2, 3),
            keepdims=True,
        )

        dL_dmean = np.sum(
            dL_dx_norm * -1 / np.sqrt(batch_var_eps), axis=(0, 2, 3), keepdims=True
        ) + dL_dvar * np.sum(
            -2 * (self.x - self.batch_mean), axis=(0, 2, 3), keepdims=True
        ) / (
            N * H * W
        )

        dL_dx = (
            (dL_dx_norm / np.sqrt(batch_var_eps))
            + (dL_dvar * 2 * (self.x - self.batch_mean) / (N * H * W))
            + (dL_dmean / (N * H * W))
        )

        self.dL_dgamma = dL_dgamma
        self.dL_dbeta = dL_dbeta

        return dL_dx


class LSTM(Layer):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False, dropout=0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout if num_layers > 1 else 0

        for layer in range(num_layers):
            input_dim = input_size if layer == 0 else hidden_size * (2 if bidirectional else 1)
            bound_ih = np.sqrt(1 / input_dim)
            bound_hh = np.sqrt(1 / hidden_size)
            self._params[f"W_ih_l{layer}"] = Parameter(
                np.random.uniform(-bound_ih, bound_ih, (4 * hidden_size, input_dim)).astype(np.float32)
            )
            self._params[f"B_ih_l{layer}"] = Parameter(
                np.zeros((4 * hidden_size), dtype=np.float32)
            )
            self._params[f"W_hh_l{layer}"] = Parameter(
                np.random.uniform(-bound_hh, bound_hh, (4 * hidden_size, hidden_size)).astype(np.float32)
            )
            self._params[f"B_hh_l{layer}"] = Parameter(
                np.zeros((4 * hidden_size), dtype=np.float32)
            )
            if bidirectional:
                self._params[f"W_ih_reverse_l{layer}"] = Parameter(
                    np.random.uniform(-bound_ih, bound_ih, (4 * hidden_size, input_dim)).astype(np.float32)
                )
                self._params[f"B_ih_reverse_l{layer}"] = Parameter(
                    np.zeros((4 * hidden_size), dtype=np.float32)
                )
                self._params[f"W_hh_reverse_l{layer}"] = Parameter(
                    np.random.uniform(-bound_hh, bound_hh, (4 * hidden_size, hidden_size)).astype(np.float32)
                )
                self._params[f"B_hh_reverse_l{layer}"] = Parameter(
                    np.zeros((4 * hidden_size), dtype=np.float32)
                )

    def init_hidden(self, batch_size):
        num_directions = 2 if self.bidirectional else 1
        h0 = np.zeros((self.num_layers * num_directions, batch_size, self.hidden_size), dtype=np.float32)
        c0 = np.zeros((self.num_layers * num_directions, batch_size, self.hidden_size), dtype=np.float32)
        return h0, c0
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_backward(self, d_output, output):
        return d_output * output * (1 - output)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_backward(self, d_output, output):
        return d_output * (1 - np.square(output))
    
    def forward_layer(self, x_t, h_prev, c_prev, W_ih, W_hh, B_ih, B_hh):
        gates = np.dot(x_t, W_ih.T) + B_ih + np.dot(h_prev, W_hh.T) + B_hh

        i_t = self.sigmoid(gates[:, :self.hidden_size])
        f_t = self.sigmoid(gates[:, self.hidden_size:2*self.hidden_size])
        g_t = self.tanh(gates[:, 2*self.hidden_size:3*self.hidden_size])
        o_t = self.sigmoid(gates[:, 3*self.hidden_size:])

        c_new = f_t * c_prev + i_t * g_t
        h_new = o_t * self.tanh(c_new)

        return h_new, c_new

    def forward(self, input_tensor, h0=None, c0=None):
        batch_size, seq_length, _ = input_tensor.shape
        if h0 is None or c0 is None:
            h0, c0 = self.init_hidden(batch_size)

        h = h0.copy()
        c = c0.copy()
        outputs = []
        self.cache = []

        for t in range(seq_length):
            x_t = input_tensor[:, t, :]

            layer_h = []
            layer_c = []
            x_t_cache = []
            for layer in range(self.num_layers):
                h_prev = h[layer]
                c_prev = c[layer]

                W_ih = self._params[f"W_ih_l{layer}"].data
                W_hh = self._params[f"W_hh_l{layer}"].data
                B_ih = self._params[f"B_ih_l{layer}"].data
                B_hh = self._params[f"B_hh_l{layer}"].data

                h_new, c_new = self.forward_layer(x_t, h_prev, c_prev, W_ih, W_hh, B_ih, B_hh)
                layer_h.append(h_new)
                layer_c.append(c_new)

                x_t_cache.append(x_t)  # Cache x_t before updating it
                x_t = h_new

                if self.bidirectional:
                    W_ih_reverse = self._params[f"W_ih_reverse_l{layer}"].data
                    W_hh_reverse = self._params[f"W_hh_reverse_l{layer}"].data
                    B_ih_reverse = self._params[f"B_ih_reverse_l{layer}"].data
                    B_hh_reverse = self._params[f"B_hh_reverse_l{layer}"].data

                    h_reverse, c_reverse = self.forward_layer(
                        input_tensor[:, seq_length - t - 1, :], h[layer], c[layer], W_ih_reverse, W_hh_reverse, B_ih_reverse, B_hh_reverse
                    )
                    h_new = np.concatenate((h_new, h_reverse), axis=-1)
                    c_new = np.concatenate((c_new, c_reverse), axis=-1)

                    layer_h[-1] = h_new
                    layer_c[-1] = c_new

                    x_t_cache[-1] = x_t  # Cache the updated x_t

            h = np.stack(layer_h, axis=0)
            c = np.stack(layer_c, axis=0)
            outputs.append(h[-1])
            self.cache.append((x_t_cache, h.copy(), c.copy()))  # Cache the list of x_t for each layer

        outputs = np.stack(outputs, axis=1)
        return outputs, h, c
    


    def backward(self, grad_output, dh_next=None, dc_next=None):
        batch_size, seq_length, _ = grad_output.shape

        dW_ih = {layer: np.zeros_like(self._params[f"W_ih_l{layer}"].data) for layer in range(self.num_layers)}
        dW_hh = {layer: np.zeros_like(self._params[f"W_hh_l{layer}"].data) for layer in range(self.num_layers)}
        dB_ih = {layer: np.zeros_like(self._params[f"B_ih_l{layer}"].data) for layer in range(self.num_layers)}
        dB_hh = {layer: np.zeros_like(self._params[f"B_hh_l{layer}"].data) for layer in range(self.num_layers)}

        if self.bidirectional:
            dW_ih_reverse = {layer: np.zeros_like(self._params[f"W_ih_reverse_l{layer}"].data) for layer in range(self.num_layers)}
            dW_hh_reverse = {layer: np.zeros_like(self._params[f"W_hh_reverse_l{layer}"].data) for layer in range(self.num_layers)}
            dB_ih_reverse = {layer: np.zeros_like(self._params[f"B_ih_reverse_l{layer}"].data) for layer in range(self.num_layers)}
            dB_hh_reverse = {layer: np.zeros_like(self._params[f"B_hh_reverse_l{layer}"].data) for layer in range(self.num_layers)}

        dh_next = np.zeros((self.num_layers, batch_size, self.hidden_size)) if dh_next is None else dh_next
        dc_next = np.zeros((self.num_layers, batch_size, self.hidden_size)) if dc_next is None else dc_next

        if self.bidirectional:
            dh_next_reverse = np.zeros_like(dh_next)
            dc_next_reverse = np.zeros_like(dc_next)

        grad_input = np.zeros((batch_size, seq_length, self.input_size))

        for t in reversed(range(seq_length)):
            x_t_cache, h, c = self.cache[t]

            for layer in reversed(range(self.num_layers)):
                if self.bidirectional:
                    dh = grad_output[:, t, :self.hidden_size] + dh_next[layer]
                else:
                    dh = grad_output[:, t, :] + dh_next[layer]
                dc = dc_next[layer]

                h_prev = h[layer - 1] if layer > 0 else np.zeros((batch_size, self.hidden_size))
                x_t = x_t_cache[layer]

                W_ih = self._params[f"W_ih_l{layer}"].data
                W_hh = self._params[f"W_hh_l{layer}"].data

                gates = np.dot(x_t, W_ih.T) + np.dot(h_prev, W_hh.T) + self._params[f"B_ih_l{layer}"].data + self._params[f"B_hh_l{layer}"].data

                i_t = self.sigmoid(gates[:, :self.hidden_size])
                f_t = self.sigmoid(gates[:, self.hidden_size:2*self.hidden_size])
                g_t = self.tanh(gates[:, 2*self.hidden_size:3*self.hidden_size])
                o_t = self.sigmoid(gates[:, 3*self.hidden_size:])

                do = dh * np.tanh(c[layer])
                dc = dc + (dh * o_t * (1 - np.tanh(c[layer]) ** 2))
                di = dc * g_t
                df = dc * (c[layer - 1] if layer > 0 else 0)
                dg = dc * i_t

                di_input = self.sigmoid_backward(di, i_t)
                df_input = self.sigmoid_backward(df, f_t)
                dg_input = self.tanh_backward(dg, g_t)
                do_input = self.sigmoid_backward(do, o_t)

                dgate = np.concatenate((di_input, df_input, dg_input, do_input), axis=1)

                dW_ih[layer] += np.dot(dgate.T, x_t)
                dW_hh[layer] += np.dot(dgate.T, h_prev)
                dB_ih[layer] += np.sum(dgate, axis=0)
                dB_hh[layer] += np.sum(dgate, axis=0)

                dh_next[layer] = np.dot(dgate, W_hh)
                dc_next[layer] = dc * f_t

                if self.bidirectional:
                    dh_reverse = grad_output[:, seq_length - t - 1, self.hidden_size:] + dh_next_reverse[layer]
                    dc_reverse = dc_next_reverse[layer]

                    h_prev_reverse = h[layer - 1][::-1] if layer > 0 else np.zeros((batch_size, self.hidden_size))
                    x_t_reverse = x_t[::-1]

                    W_ih_reverse = self._params[f"W_ih_reverse_l{layer}"].data
                    W_hh_reverse = self._params[f"W_hh_reverse_l{layer}"].data

                    gates_reverse = np.dot(x_t_reverse, W_ih_reverse.T) + np.dot(h_prev_reverse, W_hh_reverse.T) + self._params[f"B_ih_reverse_l{layer}"].data + self._params[f"B_hh_reverse_l{layer}"].data

                    i_t_reverse = self.sigmoid(gates_reverse[:, :self.hidden_size])
                    f_t_reverse = self.sigmoid(gates_reverse[:, self.hidden_size:2*self.hidden_size])
                    g_t_reverse = self.tanh(gates_reverse[:, 2*self.hidden_size:3*self.hidden_size])
                    o_t_reverse = self.sigmoid(gates_reverse[:, 3*self.hidden_size:])

                    do_reverse = dh_reverse * np.tanh(c[layer][::-1])
                    dc_reverse = dc_reverse + (dh_reverse * o_t_reverse * (1 - np.tanh(c[layer][::-1]) ** 2))
                    di_reverse = dc_reverse * g_t_reverse
                    df_reverse = dc_reverse * (c[layer - 1][::-1] if layer > 0 else 0)
                    dg_reverse = dc_reverse * i_t_reverse

                    di_input_reverse = self.sigmoid_backward(di_reverse, i_t_reverse)
                    df_input_reverse = self.sigmoid_backward(df_reverse, f_t_reverse)
                    dg_input_reverse = self.tanh_backward(dg_reverse, g_t_reverse)
                    do_input_reverse = self.sigmoid_backward(do_reverse, o_t_reverse)

                    dgate_reverse = np.concatenate((di_input_reverse, df_input_reverse, dg_input_reverse, do_input_reverse), axis=1)

                    dW_ih_reverse[layer] += np.dot(dgate_reverse.T, x_t_reverse)
                    dW_hh_reverse[layer] += np.dot(dgate_reverse.T, h_prev_reverse)
                    dB_ih_reverse[layer] += np.sum(dgate_reverse, axis=0)
                    dB_hh_reverse[layer] += np.sum(dgate_reverse, axis=0)

                    dh_next_reverse[layer] = np.dot(dgate_reverse, W_hh_reverse)
                    dc_next_reverse[layer] = dc_reverse * f_t_reverse

                    grad_input[:, seq_length - t - 1, :] += np.dot(dgate_reverse, W_ih_reverse.T)

            grad_input[:, t, :] += np.dot(dgate, W_ih)

        for layer in range(self.num_layers):
            self._params[f"W_ih_l{layer}"].grad = dW_ih[layer]
            self._params[f"W_hh_l{layer}"].grad = dW_hh[layer]
            self._params[f"B_ih_l{layer}"].grad = dB_ih[layer]
            self._params[f"B_hh_l{layer}"].grad = dB_hh[layer]

            if self.bidirectional:
                self._params[f"W_ih_reverse_l{layer}"].grad = dW_ih_reverse[layer]
                self._params[f"W_hh_reverse_l{layer}"].grad = dW_hh_reverse[layer]
                self._params[f"B_ih_reverse_l{layer}"].grad = dB_ih_reverse[layer]
                self._params[f"B_hh_reverse_l{layer}"].grad = dB_hh_reverse[layer]

        return grad_input, dh_next, dc_next
    
    def backward_layer(self, d_h_new, d_c_new, x_t, h_prev, c_prev, W_ih, W_hh, i_t, f_t, g_t, o_t, c_new):
        d_o_t = d_h_new * self.tanh(c_new)
        d_c_new += d_h_new * o_t * (1 - self.tanh(c_new) ** 2)

        d_f_t = d_c_new * c_prev
        d_i_t = d_c_new * g_t
        d_g_t = d_c_new * i_t

        d_i_t = self.sigmoid_backward(d_i_t, i_t)
        d_f_t = self.sigmoid_backward(d_f_t, f_t)
        d_g_t = self.tanh_backward(d_g_t, g_t)
        d_o_t = self.sigmoid_backward(d_o_t, o_t)

        d_gates = np.hstack([d_i_t, d_f_t, d_g_t, d_o_t])

        d_x_t = np.dot(d_gates, W_ih)
        d_h_prev = np.dot(d_gates, W_hh)
        d_c_prev = d_c_new * f_t

        

        d_W_ih = np.dot(d_gates.T, x_t)
        d_W_hh = np.dot(d_gates.T, h_prev)
        d_B_ih = np.sum(d_gates, axis=0)
        d_B_hh = np.sum(d_gates, axis=0)

        return d_x_t, d_h_prev, d_c_prev, d_W_ih, d_W_hh, d_B_ih, d_B_hh


    
    def backward(self, d_output):
        batch_size, seq_length, _ = d_output.shape
        d_x = np.zeros((batch_size, seq_length, self.input_size), dtype=np.float32)
        d_h = np.zeros_like(self.cache[0][1])
        d_c = np.zeros_like(self.cache[0][2])

        for t in reversed(range(seq_length)):
            x_t_cache, h_cache, c_cache = self.cache[t]

            for layer in reversed(range(self.num_layers)):
                W_ih = self._params[f"W_ih_l{layer}"].data
                W_hh = self._params[f"W_hh_l{layer}"].data

                h_prev = h_cache[layer]
                c_prev = c_cache[layer]
                i_t = self.sigmoid(np.dot(x_t_cache[layer], W_ih.T) + self._params[f"B_ih_l{layer}"].data + np.dot(h_prev, W_hh.T) + self._params[f"B_hh_l{layer}"].data)[:, :self.hidden_size]
                f_t = self.sigmoid(np.dot(x_t_cache[layer], W_ih.T) + self._params[f"B_ih_l{layer}"].data + np.dot(h_prev, W_hh.T) + self._params[f"B_hh_l{layer}"].data)[:, self.hidden_size:2*self.hidden_size]
                g_t = self.tanh(np.dot(x_t_cache[layer], W_ih.T) + self._params[f"B_ih_l{layer}"].data + np.dot(h_prev, W_hh.T) + self._params[f"B_hh_l{layer}"].data)[:, 2*self.hidden_size:3*self.hidden_size]
                o_t = self.sigmoid(np.dot(x_t_cache[layer], W_ih.T) + self._params[f"B_ih_l{layer}"].data + np.dot(h_prev, W_hh.T) + self._params[f"B_hh_l{layer}"].data)[:, 3*self.hidden_size:]
                c_new = f_t * c_prev + i_t * g_t

                d_h_new = d_output[:, t, :] + d_h[layer]
                d_c_new = d_c[layer]

                d_x_t, d_h_prev, d_c_prev, d_W_ih, d_W_hh, d_B_ih, d_B_hh = self.backward_layer(
                    d_h_new, d_c_new, x_t_cache[layer], h_prev, c_prev,
                    W_ih, W_hh, i_t, f_t, g_t, o_t, c_new
                )

                self._params[f"W_ih_l{layer}"].grad += d_W_ih
                self._params[f"W_hh_l{layer}"].grad += d_W_hh
                self._params[f"B_ih_l{layer}"].grad += d_B_ih
                self._params[f"B_hh_l{layer}"].grad += d_B_hh

                d_h[layer] = d_h_prev
                d_c[layer] = d_c_prev

                if layer == 0:
                    d_x[:, t, :] = d_x_t
                else:
                    d_output[:, t, :] = d_x_t

        return d_x, d_h, d_c
    
    
class Embedding(Layer):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self._params["E"] = Parameter(np.random.rand(vocab_size, emb_dim).astype(np.float32), True)

    def forward(self, indices):
        self.last_indices = indices
        return self._params["E"].data[indices]

    def backward(self, dL_dy):
        E_grad = self._params["E"].grad
        indices = self.last_indices

        np.add.at(E_grad, indices, dL_dy)

        # dL_dx can be zero or some placeholder as gradients are not propagated through indices
        dL_dx = np.zeros_like(indices, dtype=float)

        return dL_dx
