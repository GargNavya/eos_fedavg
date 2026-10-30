"""Recompute eigenvector alignment for finished runs from their saved final_w.pt (no retraining).

python scripts/align.py $RESULTS_DIR/fedavg_align/*/ --k 5
"""
import argparse
import json
import os

import torch

from eosfl.alignment import alignment
from eosfl.config import load_config
from eosfl.fedavg import setup

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tol", type=float, default=1e-4)
    a = ap.parse_args()
    for d in a.run_dirs:
        wpath = os.path.join(d, "final_w.pt")
        if not os.path.exists(wpath):
            print("skip (no final_w.pt):", d)
            continue
        with open(os.path.join(d, "config.json")) as f:
            cfg = load_config(overrides=json.load(f))
        s = setup(cfg)
        w = torch.load(wpath).to(device=s["w"].device, dtype=s["w"].dtype)
        al = alignment(s["fm"], w, s["X"], s["Y"], s["clients"], s["p"], a.k, a.tol)
        al.update(lr=cfg.lr, source="final_w.pt")
        with open(os.path.join(d, "alignment.json"), "w") as f:
            json.dump(al, f, indent=1)
        print(f"{os.path.basename(os.path.normpath(d))}: λ_g·η/2={al['lam_global'][0] * cfg.lr / 2:.3f} "
              f"Σpλ·η/2={al['weyl_sum'] * cfg.lr / 2:.3f}")