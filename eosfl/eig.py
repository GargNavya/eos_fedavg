import numpy as np
import torch
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigs, eigsh


def _operator(matvec, like, real_input=False):
    npdt = np.float64 if like.dtype == torch.float64 else np.float32
    dim = like.numel()

    def mv(x):
        x = np.real(x) if real_input else x
        xt = torch.from_numpy(np.ascontiguousarray(x, dtype=npdt).reshape(-1)).to(like.device)
        return matvec(xt).detach().cpu().numpy().astype(npdt, copy=False)

    return LinearOperator((dim, dim), matvec=mv, dtype=npdt)


def eigsh_torch(matvec, like, k=1, which="LA", v0=None, tol=1e-3, maxiter=None):
    """Lanczos (ARPACK) on a symmetric operator given as a torch matvec. Sorted ascending."""
    try:
        vals, vecs = eigsh(_operator(matvec, like), k=k, which=which, v0=v0, tol=tol, maxiter=maxiter)
    except ArpackNoConvergence as e:
        vals, vecs = e.eigenvalues, e.eigenvectors
    order = np.argsort(vals)
    return vals[order], vecs[:, order]


def eigs_torch(matvec, like, k=2, which="LR", v0=None, tol=1e-3, maxiter=None):
    """ARPACK on a non-symmetric operator. Complex eigenvalues sorted by real part."""
    try:
        vals, vecs = eigs(_operator(matvec, like, real_input=True), k=k, which=which, v0=v0, tol=tol, maxiter=maxiter)
    except ArpackNoConvergence as e:
        vals, vecs = e.eigenvalues, e.eigenvectors
    order = np.argsort(vals.real)
    return vals[order], vecs[:, order]


class EigTracker:
    """Extreme eigenvalues with warm starts kept per key.

    maxiter caps ARPACK restarts per solve (each restart costs ~ncv = 20 matvecs);
    None = scipy default. A solve that does not converge returns NaN instead of hanging.
    """

    def __init__(self, tol=1e-3, maxiter=None):
        self.tol = tol
        self.maxiter = maxiter
        self.v0 = {}
        self.n_mv = 0

    def _count(self, matvec):
        def mv(v):
            self.n_mv += 1
            return matvec(v)
        return mv

    def top(self, key, matvec, like):
        vals, vecs = eigsh_torch(self._count(matvec), like, 1, "LA", self.v0.get(key), self.tol, self.maxiter)
        if len(vals) == 0:
            return float("nan")
        self.v0[key] = vecs[:, -1].copy()
        return float(vals[-1])

    def extremes(self, key, matvec, like):
        """Symmetric operator: (smallest, largest) eigenvalue."""
        vals, vecs = eigsh_torch(self._count(matvec), like, 2, "BE", self.v0.get(key), self.tol, self.maxiter)
        if len(vals) < 2:
            return float("nan"), float("nan")
        self.v0[key] = vecs.sum(1)
        return float(vals[0]), float(vals[-1])

    def _one_end(self, key, matvec, like, which):
        vals, vecs = eigs_torch(self._count(matvec), like, 2, which, self.v0.get(key), self.tol, self.maxiter)
        if len(vals) == 0:
            return complex("nan")
        i = -1 if which == "LR" else 0
        v = vecs[:, i]
        self.v0[key] = (np.real(v) + np.imag(v)).copy()
        return complex(vals[i])

    def top_real(self, key, matvec, like):
        """Non-symmetric operator: (real, imag) of the eigenvalue with the largest real part."""
        z = self._one_end(key, matvec, like, "LR")
        return z.real, z.imag

    def ends_nonsym(self, key, matvec, like):
        """Non-symmetric operator: eigenvalues with the smallest and the largest real part (complex)."""
        return (self._one_end(key + "_SR", matvec, like, "SR"),
                self._one_end(key + "_LR", matvec, like, "LR"))


def growth(taus):
    """Per-round growth factor implied by eigenvalues tau of T = I - J: max |1 - tau|."""
    vals = [abs(1 - complex(t)) for t in taus if np.isfinite(complex(t).real)]
    return max(vals) if vals else float("nan")