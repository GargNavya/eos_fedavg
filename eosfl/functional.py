import torch
from torch.func import functional_call, grad, jvp


class FlatModel:
    """Model as a function of one flat parameter vector w."""

    def __init__(self, model):
        self.model = model
        named = list(model.named_parameters())
        self.names = [n for n, _ in named]
        self.shapes = [p.shape for _, p in named]
        self.numels = [p.numel() for _, p in named]
        self.buffers = dict(model.named_buffers())
        self.dim = sum(self.numels)
        self._grad = grad(self.loss)

    def init_vector(self):
        return torch.cat([p.detach().reshape(-1) for _, p in self.model.named_parameters()]).clone()

    def params(self, w):
        chunks = torch.split(w, self.numels)
        return {n: c.view(s) for n, c, s in zip(self.names, chunks, self.shapes)}

    def forward(self, w, X):
        return functional_call(self.model, {**self.params(w), **self.buffers}, (X,))

    def loss(self, w, X, Y):
        return 0.5 * ((self.forward(w, X) - Y) ** 2).sum(1).mean()

    def grad(self, w, X, Y):
        return self._grad(w, X, Y)

    def hvp(self, w, X, Y, v):
        return jvp(lambda u: self._grad(u, X, Y), (w,), (v,))[1]

    @torch.no_grad()
    def evaluate(self, w, X, Y, y):
        if len(y) == 0:
            return float("nan"), float("nan")
        out = self.forward(w, X)
        loss = 0.5 * ((out - Y) ** 2).sum(1).mean()
        acc = (out.argmax(1) == y).float().mean()
        return loss.item(), acc.item()
