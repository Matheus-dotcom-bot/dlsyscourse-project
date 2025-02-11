"""The module.
"""
from functools import reduce
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(init.kaiming_uniform(
            fan_in=in_features,
            fan_out=out_features,
            device=device,
            dtype=dtype,
            requires_grad=True
        ))

        self.bias = None
        if bias:
            self.bias = init.kaiming_uniform(
                fan_in=out_features,
                fan_out=1,
                device=device,
                dtype=dtype,
                requires_grad=True
            )
            self.bias = Parameter(self.bias.reshape((1, out_features)))

    def forward(self, X: Tensor) -> Tensor:
        ret = X @ self.weight
        if self.bias:
            ret += self.bias.broadcast_to(ret.shape)
        return ret


class Flatten(Module):
    def forward(self, X):
        N, *dims = X.shape
        return X.reshape((N, np.prod(dims)))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.tanh(x)


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return (1 + ops.exp(-x))**-1


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        return reduce(lambda a, f: f(a), self.modules, x)


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        n, classes = logits.shape
        y_one_hot = init.one_hot(classes, y, device=y.device)
        return (ops.logsumexp(logits, 1) - (logits * y_one_hot).sum(1)).sum() / n


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        M, N = x.shape
        assert self.dim == N
        m = self.momentum
        if self.training:
            mean = (x.sum(axes=0) / M)
            self.running_mean = (1 - m) * self.running_mean + m * mean.data
            mean = mean.reshape((1, N)).broadcast_to(x.shape)

            var = ((x - mean)**2).sum(axes=0) / M
            self.running_var = (1 - m) * self.running_var + m * var.data
            var = var.reshape((1, N)).broadcast_to(x.shape)
        else:
            mean = self.running_mean.reshape((1, N)).broadcast_to(x.shape)
            var = self.running_var.reshape((1, N)).broadcast_to(x.shape)
        x = (x - mean) / (var + self.eps)**0.5
        weight = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
        bias = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
        return weight * x + bias


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        M, N = x.shape
        assert self.dim == N
        mean = (x.sum(axes=1) / N).reshape((M, 1)).broadcast_to(x.shape)
        var = (((x - mean)**2).sum(axes=1) / N).reshape((M, 1)).broadcast_to(x.shape)
        x = ((x - mean) / (var + self.eps)**0.5)
        return x * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mask = init.randb(
                *x.shape,
                p=1-self.p,
                dtype='float32',
                device=x.device,
            )
            x = mask * x / (1 - self.p)
        return x


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.weight = Parameter(init.kaiming_uniform(
            fan_in=in_channels * kernel_size * kernel_size,
            fan_out=out_channels * kernel_size * kernel_size,
            shape=(kernel_size, kernel_size, in_channels, out_channels),
            device=device,
            dtype=dtype,
            requires_grad=True
        ))

        self.bias = None
        if bias:
            self.bias = Parameter(init.kaiming_uniform(
                fan_in=6 * self.in_channels * self.kernel_size**2,
                fan_out=None,
                shape=[self.out_channels],
                device=device,
                dtype=dtype,
                requires_grad=True
            ))


    def forward(self, x: Tensor) -> Tensor:
        ret = ops.conv(
            x.transpose((1, 2)).transpose((2, 3)),
            self.weight,
            stride=self.stride,
            padding=self.kernel_size // 2
        )
        if self.bias:
            ret += self.bias.reshape((1, 1, 1, self.out_channels)).broadcast_to(ret.shape)
        return ret.transpose((1, 3)).transpose((2, 3))


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.device = device
        self.dtype = dtype

        PARAMS = dict(
            fan_in=6 * hidden_size,
            fan_out=None,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        shape = input_size, hidden_size
        self.W_ih = Parameter(init.kaiming_uniform(shape=shape, **PARAMS))

        shape = hidden_size, hidden_size
        self.W_hh = Parameter(init.kaiming_uniform(shape=shape, **PARAMS))

        shape = hidden_size,
        self.bias_ih = Parameter(init.kaiming_uniform(shape=shape, **PARAMS))
        self.bias_hh = Parameter(init.kaiming_uniform(shape=shape, **PARAMS))


    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        bs, _ = X.shape
        shape = bs, self.hidden_size
        h = h or init.zeros(*shape, device=self.device, dtype=self.dtype)

        X_new = X @ self.W_ih + h @ self.W_hh
        if self.bias:
            add_dim = 1, self.hidden_size
            bias_ih = self.bias_ih.reshape(add_dim).broadcast_to(shape)
            bias_hh = self.bias_hh.reshape(add_dim).broadcast_to(shape)
            X_new += bias_ih + bias_hh

        h_out = getattr(ops, self.nonlinearity)(X_new)
        return h_out


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        def rnn_cell(input_size):
            return RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype)

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.device = device
        self.dtype = dtype

        self.rnn_cells = [rnn_cell(input_size)] + [rnn_cell(hidden_size) for _ in range(num_layers - 1)]



    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        _, bs, _ = X.shape
        shape = self.num_layers, bs, self.hidden_size
        h0 = h0 or init.zeros(*shape, device=self.device, dtype=self.dtype)

        X_new = ops.split(X, axis=0)
        h_n = []
        for rnn_cell, h in zip(self.rnn_cells, ops.split(h0, axis=0)):
            X_new = [(h := rnn_cell(X_t, h)) for X_t in X_new]
            h_n.append(h)

        output = ops.stack(X_new, axis=0)
        h_n = ops.stack(h_n, axis=0)
        return output, h_n


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device
        self.dtype = dtype

        PARAMS = dict(
            fan_in=6 * hidden_size,
            fan_out=None,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )

        shape = input_size, 4 * hidden_size
        self.W_ih = Parameter(init.kaiming_uniform(shape=shape, **PARAMS))

        shape = hidden_size, 4 * hidden_size
        self.W_hh = Parameter(init.kaiming_uniform(shape=shape, **PARAMS))

        shape = 4 * hidden_size,
        self.bias_ih = Parameter(init.kaiming_uniform(shape=shape, **PARAMS))
        self.bias_hh = Parameter(init.kaiming_uniform(shape=shape, **PARAMS))


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        bs, _ = X.shape
        shape = 2, bs, self.hidden_size
        h = h or ops.split(init.zeros(*shape, device=self.device, dtype=self.dtype), 0)
        h0, c0 = h

        X_new = X @ self.W_ih + h0 @ self.W_hh
        if self.bias:
            add_dim = 1, 4 * self.hidden_size
            shape = bs, 4 * self.hidden_size
            bias_ih = self.bias_ih.reshape(add_dim).broadcast_to(shape)
            bias_hh = self.bias_hh.reshape(add_dim).broadcast_to(shape)
            X_new += bias_ih + bias_hh

        i, f, g, o = ops.split(X_new.reshape((bs, 4, self.hidden_size)), axis=1)
        i, f, g, o = Sigmoid()(i), Sigmoid()(f), Tanh()(g), Sigmoid()(o)

        c_out = f * c0 + i * g
        h_out = o * Tanh()(c_out)
        return h_out, c_out


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        def lstm_cell(input_size):
            return LSTMCell(input_size, hidden_size, bias, device, dtype)

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.device = device
        self.dtype = dtype

        self.lstm_cells = [lstm_cell(input_size)] + [lstm_cell(hidden_size) for _ in range(num_layers - 1)]

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            c_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        _, bs, _ = X.shape
        shape = 2, self.num_layers, bs, self.hidden_size
        h0, c0 = h or ops.split(init.zeros(*shape, device=self.device, dtype=self.dtype), 0)

        h0 = ops.split(h0, axis=0)
        c0 = ops.split(c0, axis=0)
        X = ops.split(X, axis=0)
        h_n = []
        c_n = []
        for lstm_cell, h, c in zip(self.lstm_cells, h0, c0):
            X_next = []
            for X_t in X:
                h, c = lstm_cell(X_t, (h, c))
                X_next.append(h)
            X = X_next
            h_n.append(h)
            c_n.append(c)

        output = ops.stack(X, 0)
        h_n = ops.stack(h_n, 0)
        c_n = ops.stack(c_n, 0)
        return output, (h_n, c_n)


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.weight = Parameter(init.randn(
            num_embeddings,
            embedding_dim,
            mean=0.0,
            std=1.0,
            device=device,
            dtype=dtype,
        ))


    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        seq_len, bs = x.shape
        one_hot = init.one_hot(
            self.num_embeddings,
            x.reshape((seq_len * bs,)),
            device=self.device,
            dtype=self.dtype,
        )
        return (one_hot @ self.weight).reshape((seq_len, bs, self.embedding_dim))


class GRUCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A gated recurrent unit (GRU) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 3*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 3*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (3*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (3*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device
        self.dtype = dtype

        PARAMS = dict(
            fan_in=6 * hidden_size,
            fan_out=None,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )

        shape = input_size, 3 * hidden_size
        self.W_ih = Parameter(init.kaiming_uniform(shape=shape, **PARAMS))

        shape = hidden_size, 3 * hidden_size
        self.W_hh = Parameter(init.kaiming_uniform(shape=shape, **PARAMS))

        shape = 3 * hidden_size,
        self.bias_ih = Parameter(init.kaiming_uniform(shape=shape, **PARAMS))
        self.bias_hh = Parameter(init.kaiming_uniform(shape=shape, **PARAMS))

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        # r = σ(W_ir @ x + b_ir + W_hr * h + b_hr)
        # z = σ(W_iz @ x + b_iz + W_hz * h + b_hz)
        # n = tanh(W_in * x + b_in + r * (Whn * h + bhn))
        # h' = (1 - z) * n + z * h
        # https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html
        bs, _ = X.shape
        shape = bs, self.hidden_size
        h = h or init.zeros(*shape, device=self.device, dtype=self.dtype)

        X_new = X @ self.W_ih
        h_new = h @ self.W_hh
        if self.bias:
            add_dim = 1, 3 * self.hidden_size
            shape = bs, 3 * self.hidden_size
            X_new += self.bias_ih.reshape(add_dim).broadcast_to(shape)
            h_new += self.bias_hh.reshape(add_dim).broadcast_to(shape)

        xr, xz, xn = ops.split(X_new.reshape((bs, 3, self.hidden_size)), axis=1)
        hr, hz, hn = ops.split(h_new.reshape((bs, 3, self.hidden_size)), axis=1)

        r = Sigmoid()(hr + xr)  # reset gates
        z = Sigmoid()(hz + xz)  # update gates
        n = Tanh()(r * hn + xn)  # new gates (candidate for replacing h)

        h_out = (1 - z) * n + z * h
        return h_out


class GRU(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        """
        Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        gru_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 3*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 3*hidden_size).
        gru_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 3*hidden_size).
        gru_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (3*hidden_size,).
        gru_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (3*hidden_size,).
        """
        def gru_cell(input_size):
            return GRUCell(input_size, hidden_size, bias, device, dtype)

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.device = device
        self.dtype = dtype

        self.gru_cells = [gru_cell(input_size)] + [gru_cell(hidden_size) for _ in range(num_layers - 1)]

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the
            input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros
            if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output
            features (h_t) from the last layer of the GRU, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden
            state for each element in the batch.
        """
        _, bs, _ = X.shape
        shape = self.num_layers, bs, self.hidden_size
        h0 = h0 or init.zeros(*shape, device=self.device, dtype=self.dtype)

        X_new = ops.split(X, axis=0)
        h_n = []
        for gru_cell, h in zip(self.gru_cells, ops.split(h0, axis=0)):
            X_new = [(h := gru_cell(X_t, h)) for X_t in X_new]
            h_n.append(h)

        output = ops.stack(X_new, axis=0)
        h_n = ops.stack(h_n, axis=0)
        return output, h_n
