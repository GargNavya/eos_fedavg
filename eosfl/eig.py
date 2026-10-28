import numpy as np
import torch
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh


def eigsh_torch(matvec, like, k=1, which="LA", v0=None, tol=1e-3):
    """Lanczos (ARPACK) on a symmetric operator given as a torch matvec."""
    npdt = np.float64 if like.dtype == torch.float64 else np.float32
    dim = like.numel()

    def mv(x):
        xt = torch.from_numpy(np.ascontiguousarray(x, dtype=npdt).reshape(-1)).to(like.device)
        return matvec(xt).detach().cpu().numpy().astype(npdt, copy=False)

    op = LinearOperator((dim, dim), matvec=mv, dtype=npdt)
    try:
        vals, vecs = eigsh(op, k=k, which=which, v0=v0, tol=tol)
    except ArpackNoConvergence as e:
        vals, vecs = e.eigenvalues, e.eigenvectors
    order = np.argsort(vals)
    return vals[order], vecs[:, order]


class EigTracker:
    """Extreme eigenvalues with warm starts kept per key."""

    def __init__(self, tol=1e-3):
        self.tol = tol
        self.v0 = {}

    def top(self, key, matvec, like):
        vals, vecs = eigsh_torch(matvec, like, 1, "LA", self.v0.get(key), self.tol)
        if len(vals) == 0:
            return float("nan")
        self.v0[key] = vecs[:, -1].copy()
        return float(vals[-1])

    def extremes(self, key, matvec, like):
        vals, vecs = eigsh_torch(matvec, like, 2, "BE", self.v0.get(key), self.tol)
        if len(vals) < 2:
            return float("nan"), float("nan")
        self.v0[key] = vecs.sum(1)
        return float(vals[0]), float(vals[-1])
