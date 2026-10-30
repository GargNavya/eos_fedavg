import json
import os

import numpy as np
import pandas as pd


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def add_derived(m, K):
    """Add growth factors and per-step edge indices to a metrics frame (works on older runs too).

    g      = max |1 - tau| over the extreme eigenvalues of T  (growth per round)
    c_eff  = (1 + g^(1/K)) / 2                                 (per-local-step edge index; 1 = edge)
    """
    if {"tau_min", "tau_max"} <= set(m):  # recompute: NaN unless both ends solved
        m["g_frozen"] = np.maximum((1 - m.tau_min).abs(), (1 - m.tau_max).abs())
    if {"tau_path_min", "tau_path"} <= set(m):
        im = lambda c: m[c].fillna(0) if c in m else 0.0
        lo = m.tau_path_min + 1j * im("tau_path_min_im")
        hi = m.tau_path + 1j * im("tau_path_im")
        m["g_path"] = np.maximum(np.abs(1 - lo), np.abs(1 - hi))  # NaN propagates
    for src, dst in [("g_frozen", "ceff_frozen"), ("g_path", "ceff_path")]:
        if src in m:
            m[dst] = (1 + m[src] ** (1.0 / K)) / 2
    return m


def load_metrics(run_dir, K=None):
    with open(os.path.join(run_dir, "metrics.jsonl")) as f:
        m = pd.DataFrame([json.loads(line) for line in f])
    if K is None:
        cfg = _read_json(os.path.join(run_dir, "config.json")) or {}
        K = cfg.get("local_steps", 1)
    return add_derived(m, K)


def equilibrium(series, frac=0.3, steps=None):
    """Median over the last `frac` of training. With `steps`, the window is the last `frac` of the
    run's local steps (so an early lone measurement never counts as equilibrium); without it,
    the last `frac` of the non-missing values."""
    if steps is not None:
        s = series[steps >= (1 - frac) * steps.max()].dropna()
    else:
        s = series.dropna()
        s = s.iloc[int(len(s) * (1 - frac)):]
    return float(s.median()) if len(s) else np.nan


EQ_COLS = ["sharpness", "client_sharp_max", "tau_max", "tau_min", "tau_path", "tau_path_min",
           "g_frozen", "g_path", "ceff_frozen", "ceff_path", "train_loss"]


def load_exp(exp_dir, frac=0.3):
    """One row per run: config + status + equilibrium values (normalized so edge = 1)."""
    rows = []
    for name in sorted(os.listdir(exp_dir)):
        d = os.path.join(exp_dir, name)
        cfg = _read_json(os.path.join(d, "config.json"))
        if cfg is None or not os.path.exists(os.path.join(d, "metrics.jsonl")):
            continue
        st = _read_json(os.path.join(d, "status.json")) or {"status": "running"}
        m = load_metrics(d, cfg["local_steps"])
        row = {**cfg, **st, "run": name, "run_dir": d}
        for col in EQ_COLS:
            row[f"eq_{col}"] = equilibrium(m[col], frac, m["steps"]) if col in m else np.nan
        row["eq_sharp_norm"] = row["eq_sharpness"] * cfg["lr"] / 2
        row["eq_client_norm"] = row["eq_client_sharp_max"] * cfg["lr"] / 2
        row["eq_edge"] = row["eq_tau_max"] / 2
        row["eq_path_edge"] = row["eq_tau_path"] / 2
        rows.append(row)
    return pd.DataFrame(rows)