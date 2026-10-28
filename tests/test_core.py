import json
import math
import os

import numpy as np
import pytest
import torch

from eosfl.config import load_config
from eosfl.data import dirichlet_split
from eosfl.eig import EigTracker
from eosfl.fedavg import fedavg_round, train
from eosfl.functional import FlatModel
from eosfl.roundmap import hessian_mv, roundmap_T_mv
from eosfl.toy import make_clients, threshold

torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(__file__)


def tiny():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 5), torch.nn.Tanh(), torch.nn.Linear(5, 3))
    fm = FlatModel(model)
    X = torch.randn(30, 4)
    Y = torch.nn.functional.one_hot(torch.randint(0, 3, (30,)), 3).double()
    return fm, fm.init_vector(), X, Y


def full_hessian(fm, w, X, Y):
    return torch.func.hessian(fm.loss)(w, X, Y)


def test_hvp_matches_hessian():
    fm, w, X, Y = tiny()
    H = full_hessian(fm, w, X, Y)
    v = torch.randn_like(w)
    assert torch.allclose(fm.hvp(w, X, Y, v), H @ v, atol=1e-10)


def test_sharpness_matches_eigh():
    fm, w, X, Y = tiny()
    H = full_hessian(fm, w, X, Y)
    top = EigTracker(1e-10).top("g", hessian_mv(fm, w, X, Y), w)
    assert top == pytest.approx(torch.linalg.eigvalsh(H).max().item(), rel=1e-6)


@pytest.mark.parametrize("K,slr,mu", [(1, 1.0, 0.0), (3, 0.7, 0.0), (4, 1.3, 0.5)])
def test_roundmap_matches_explicit(K, slr, mu):
    fm, w, X, Y = tiny()
    clients = [(X[:10], Y[:10]), (X[10:], Y[10:])]
    p = [10 / 30, 20 / 30]
    lr = 0.3
    I = torch.eye(w.numel())
    A = sum(pi * torch.linalg.matrix_power(I - lr * (full_hessian(fm, w, Xi, Yi) + mu * I), K)
            for (Xi, Yi), pi in zip(clients, p))
    ev = torch.linalg.eigvalsh(slr * (I - A))
    tmin, tmax = EigTracker(1e-10).extremes("T", roundmap_T_mv(fm, w, clients, p, lr, K, slr, mu), w)
    assert tmin == pytest.approx(ev.min().item(), abs=1e-6)
    assert tmax == pytest.approx(ev.max().item(), abs=1e-6)


def test_roundmap_K1_is_lr_times_global_hessian():
    fm, w, X, Y = tiny()
    clients, p = [(X[:10], Y[:10]), (X[10:], Y[10:])], [1 / 3, 2 / 3]
    lr, slr = 0.3, 0.8
    lam = torch.linalg.eigvalsh(full_hessian(fm, w, X, Y)).max().item()
    _, tmax = EigTracker(1e-10).extremes("T", roundmap_T_mv(fm, w, clients, p, lr, 1, slr), w)
    assert tmax == pytest.approx(slr * lr * lam, rel=1e-6)


def test_fedavg_round_K1_is_gd():
    fm, w, X, Y = tiny()
    clients, p = [(X[:10], Y[:10]), (X[10:], Y[10:])], [1 / 3, 2 / 3]
    w1, _ = fedavg_round(fm, w, clients, p, lr=0.1, K=1, server_lr=1.0)
    assert torch.allclose(w1, w - 0.1 * fm.grad(w, X, Y), atol=1e-12)


def test_dirichlet_split_is_partition():
    y = np.random.default_rng(0).integers(0, 10, 2000)
    for alpha in [math.inf, 1.0, 0.1]:
        parts = dirichlet_split(y, 10, alpha, np.random.default_rng(1), min_size=5)
        allidx = np.sort(np.concatenate(parts))
        assert np.array_equal(allidx, np.arange(len(y)))


@pytest.mark.parametrize("K", [1, 2, 5, 10])
def test_toy_identical_clients_edge_is_2_over_lr(K):
    Hs, cs, p = make_clients(het=0.0)
    out = threshold(Hs, cs, p, K, 1.0, n_grid=400, rounds=1500)
    assert out["glob_norm"] == pytest.approx(1.0, rel=0.02)
    assert out["rel_err"] < 0.02


def test_toy_K1_server_lr():
    Hs, cs, p = make_clients(het=0.6)
    out = threshold(Hs, cs, p, 1, 0.5, n_grid=400, rounds=1500)
    assert out["glob_norm_slr"] == pytest.approx(1.0, rel=0.02)


def test_debug_run_end_to_end(tmp_path):
    cfg = load_config(os.path.join(HERE, "..", "configs", "debug.yaml"),
                      [f"results_dir={tmp_path}", "max_steps=60", "dtype=float64"])
    res = train(cfg, verbose=False)
    assert res["status"] in ("budget", "converged")
    with open(os.path.join(cfg.run_dir(), "metrics.jsonl")) as f:
        rows = [json.loads(line) for line in f]
    assert any("tau_max" in r for r in rows)
    assert any("client_sharp" in r for r in rows)
