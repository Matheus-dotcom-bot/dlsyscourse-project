"""Optimization module"""
import needle as ndl
import numpy as np

class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = [0] * len(params)
        self.weight_decay = weight_decay

    def step(self):
        for i, w in enumerate(self.params):
            if w.grad is None:
                continue
            grad = w.grad.data + self.weight_decay * w.data
            self.u[i] = self.momentum * self.u[i] + (1 - self.momentum) * grad
            w.data = w.data - self.lr * self.u[i].data

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        total_norm = np.linalg.norm(np.array([np.linalg.norm(p.grad.detach().numpy()).reshape((1,)) for p in self.params]))
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = min((np.asscalar(clip_coef), 1.0))
        for p in self.params:
            p.grad = p.grad.detach() * clip_coef_clamped


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = [0] * len(params)
        self.v = [0] * len(params)

    def step(self):
        self.t += 1
        for i, w in enumerate(self.params):
            if w.grad is None:
                continue
            grad = w.grad.data + self.weight_decay * w.data
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2

            m̂ = self.m[i] / (1 - self.beta1**self.t)
            v̂ = self.v[i] / (1 - self.beta2**self.t)
            w.data = w.data - self.lr * m̂ / (v̂**0.5 + self.eps)
