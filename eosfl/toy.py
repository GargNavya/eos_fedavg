"""Exact check of the round-map prediction on 2D quadratic clients (float64, CPU)."""
import numpy as np


def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def make_clients(M=3, het=0.0, lam_top=10.0, lam_low=1.0, seed=0):
    """het = 0 gives identical clients; larger het spreads curvature and rotates eigvecs."""
    rng = np.random.default_rng(seed)
    spread = np.linspace(-1, 1, M)
    Hs = []
    for s in spread:
        Q = rot(0.5 * het * s)
        Hs.append(Q @ np.diag([lam_top * (1 + het * s), lam_low]) @ Q.T)
    cs = rng.standard_normal((M, 2))
    p = np.full(M, 1.0 / M)
    return np.array(Hs), cs, p


def round_matrix(Hs, p, lr, K, server_lr, mu=0.0):
    """R for a batch of learning rates: shape (L, d, d)."""
    lr = np.atleast_1d(lr)[:, None, None]
    d = Hs.shape[-1]
    I = np.eye(d)
    A = sum(pi * np.linalg.matrix_power(I - lr * (H + mu * I), K) for H, pi in zip(Hs, p))
    return I - server_lr * (I - A)


def spectral_radius(R):
    return np.abs(np.linalg.eigvalsh(R)).max(-1)


def simulate_diverges(Hs, cs, p, lrs, K, server_lr, mu=0.0, rounds=3000, blowup=1e6, seed=1):
    """Run FedAvg on the quadratics for all lrs at once. Returns bool array (diverged)."""
    rng = np.random.default_rng(seed)
    lrs = np.asarray(lrs)[:, None]
    w = np.tile(rng.standard_normal(2), (len(lrs), 1))
    dead = np.zeros(len(lrs), dtype=bool)
    with np.errstate(all="ignore"):
        for _ in range(rounds):
            delta = np.zeros_like(w)
            for H, c, pi in zip(Hs, cs, p):
                u = w.copy()
                for _ in range(K):
                    u = u - lrs * ((u - c) @ H + mu * (u - w))
                delta += pi * (u - w)
            w = w + server_lr * delta
            bad = ~np.isfinite(w).all(1) | (np.linalg.norm(w, axis=1) > blowup)
            dead |= bad
            w[dead] = 0.0
    return dead


def first_true(mask, grid):
    i = np.argmax(mask)
    return grid[i] if mask[i] else np.nan


def threshold(Hs, cs, p, K, server_lr, mu=0.0, n_grid=1000, rounds=3000):
    lam_glob = np.linalg.eigvalsh(np.tensordot(p, Hs, 1)).max()
    lam_cli = np.linalg.eigvalsh(Hs).max()
    grid = np.geomspace(0.05 / lam_cli, 10.0 / lam_glob, n_grid)
    pred = first_true(spectral_radius(round_matrix(Hs, p, grid, K, server_lr, mu)) > 1 + 1e-9, grid)
    emp = first_true(simulate_diverges(Hs, cs, p, grid, K, server_lr, mu, rounds), grid)
    return dict(lam_glob=lam_glob, lam_client_max=lam_cli, lr_pred=pred, lr_emp=emp,
                rel_err=abs(emp - pred) / pred,
                glob_norm=pred * lam_glob / 2, client_norm=pred * lam_cli / 2,
                glob_norm_slr=pred * server_lr * lam_glob / 2)
