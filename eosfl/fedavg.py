import json
import math
import os
import time

import numpy as np
import torch

from .data import dirichlet_split, load_dataset
from .eig import EigTracker
from .functional import FlatModel
from .models import build_model
from .roundmap import hessian_mv, roundmap_T_mv


def fedavg_round(fm, w, clients, p, lr, K, server_lr, mu=0.0):
    delta = torch.zeros_like(w)
    for (X, Y), pi in zip(clients, p):
        u = w
        for _ in range(K):
            g = fm.grad(u, X, Y)
            if mu:
                g = g + mu * (u - w)
            u = u - lr * g
        delta = delta + pi * (u - w)
    return w + server_lr * delta, delta


class JsonlLogger:
    def __init__(self, path):
        self.f = open(path, "w")

    def write(self, row):
        self.f.write(json.dumps(row) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def _dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def setup(cfg):
    """Build everything a run needs. Reusable from notebooks."""
    use_cuda = cfg.device.startswith("cuda") and torch.cuda.is_available()
    device = torch.device(cfg.device if use_cuda else "cpu")
    dtype = getattr(torch, cfg.dtype)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(cfg.seed)

    data = load_dataset(cfg)
    split_rng = np.random.default_rng(cfg.data_seed + 1)
    parts = dirichlet_split(data["ytr"], cfg.num_clients, cfg.alpha, split_rng, cfg.min_client_size)

    def T(a, dt=dtype):
        return torch.as_tensor(a, dtype=dt, device=device)

    X, Y, y = T(data["Xtr"]), T(data["Ytr"]), T(data["ytr"], torch.long)
    Xte, Yte, yte = T(data["Xte"]), T(data["Yte"]), T(data["yte"], torch.long)
    clients = []
    for idx in parts:
        i = T(idx, torch.long)
        clients.append((X[i], Y[i]))
    p = [len(idx) / len(y) for idx in parts]

    model = build_model(cfg, data["in_shape"], data["num_classes"]).to(device=device, dtype=dtype)
    fm = FlatModel(model)
    split_info = {
        "sizes": [len(i) for i in parts],
        "label_counts": [np.bincount(data["ytr"][i], minlength=data["num_classes"]).tolist() for i in parts],
    }
    return dict(fm=fm, w=fm.init_vector(), X=X, Y=Y, y=y, Xte=Xte, Yte=Yte, yte=yte,
                clients=clients, p=p, split_info=split_info, device=device)


def measure(cfg, s, w, eig, row, which):
    """Fill `row` with the measurements named in `which`."""
    fm, X, Y = s["fm"], s["X"], s["Y"]
    if "test" in which:
        row["test_loss"], row["test_acc"] = fm.evaluate(w, s["Xte"], s["Yte"], s["yte"])
    if "sharp" in which:
        row["sharpness"] = eig.top("global", hessian_mv(fm, w, X, Y), w)
    if "client" in which:
        cs = [eig.top(f"c{i}", hessian_mv(fm, w, Xi, Yi), w) for i, (Xi, Yi) in enumerate(s["clients"])]
        row["client_sharp"] = cs
        row["client_sharp_max"] = max(cs)
    if "roundmap" in which:
        mv = roundmap_T_mv(fm, w, s["clients"], s["p"], cfg.lr, cfg.local_steps, cfg.server_lr, cfg.prox_mu)
        row["tau_min"], row["tau_max"] = eig.extremes("T", mv, w)
    return row


def train(cfg, verbose=True):
    t0 = time.time()
    run_dir = cfg.run_dir()
    os.makedirs(run_dir, exist_ok=True)
    s = setup(cfg)
    fm, w, X, Y, y = s["fm"], s["w"], s["X"], s["Y"], s["y"]
    _dump(cfg.to_dict(), os.path.join(run_dir, "config.json"))
    _dump(s["split_info"], os.path.join(run_dir, "split.json"))
    log = JsonlLogger(os.path.join(run_dir, "metrics.jsonl"))
    eig = EigTracker(cfg.eig_tol)
    K = cfg.local_steps

    def due(every, r, last):
        return every > 0 and (r % every == 0 or last)

    r, upd, status = 0, float("nan"), "budget"
    while True:
        steps = r * K
        loss, acc = fm.evaluate(w, X, Y, y)
        row = {"round": r, "steps": steps, "train_loss": loss, "train_acc": acc, "update_norm": upd}
        if not math.isfinite(loss) or loss > cfg.diverge_loss:
            status = "diverged"
            log.write(row)
            break
        hit = cfg.target_loss > 0 and loss <= cfg.target_loss
        last = hit or r >= cfg.max_rounds or (cfg.max_steps > 0 and steps >= cfg.max_steps)
        which = [k for k, e in [("test", cfg.log_every), ("sharp", cfg.sharp_every),
                                ("client", cfg.client_sharp_every), ("roundmap", cfg.roundmap_every)]
                 if due(e, r, last)]
        if which:
            log.write(measure(cfg, s, w, eig, row, which))
        if verbose and (r % cfg.print_every == 0 or last):
            extra = f" sharp={row['sharpness']:.2f} (2/lr={2 / cfg.lr:.2f})" if "sharpness" in row else ""
            extra += f" tau_max/2={row['tau_max'] / 2:.3f}" if "tau_max" in row else ""
            print(f"[{cfg.run_name()}] r={r} steps={steps} loss={loss:.4f} acc={acc:.3f}{extra}", flush=True)
        if last:
            status = "converged" if hit else "budget"
            break
        w, delta = fedavg_round(fm, w, s["clients"], s["p"], cfg.lr, K, cfg.server_lr, cfg.prox_mu)
        upd = delta.norm().item()
        r += 1

    log.close()
    result = {"status": status, "rounds": r, "steps": r * K, "final_train_loss": loss,
              "seconds": round(time.time() - t0, 1)}
    _dump(result, os.path.join(run_dir, "status.json"))
    if verbose:
        print(f"[{cfg.run_name()}] done: {result}", flush=True)
    return result
