import json
import os

import numpy as np
import pandas as pd


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_metrics(run_dir):
    with open(os.path.join(run_dir, "metrics.jsonl")) as f:
        return pd.DataFrame([json.loads(line) for line in f])


def equilibrium(series, frac=0.3):
    """Median over the last `frac` of the non-missing values."""
    s = series.dropna()
    s = s.iloc[int(len(s) * (1 - frac)):]
    return float(s.median()) if len(s) else np.nan


def load_exp(exp_dir, frac=0.3):
    """One row per run: config + status + equilibrium values (normalized so edge = 1)."""
    rows = []
    for name in sorted(os.listdir(exp_dir)):
        d = os.path.join(exp_dir, name)
        cfg = _read_json(os.path.join(d, "config.json"))
        if cfg is None or not os.path.exists(os.path.join(d, "metrics.jsonl")):
            continue
        st = _read_json(os.path.join(d, "status.json")) or {"status": "running"}
        m = load_metrics(d)
        row = {**cfg, **st, "run": name, "run_dir": d}
        for col in ["sharpness", "client_sharp_max", "tau_max", "tau_min", "tau_path", "train_loss"]:
            row[f"eq_{col}"] = equilibrium(m[col], frac) if col in m else np.nan
        row["eq_sharp_norm"] = row["eq_sharpness"] * cfg["lr"] / 2
        row["eq_client_norm"] = row["eq_client_sharp_max"] * cfg["lr"] / 2
        row["eq_edge"] = row["eq_tau_max"] / 2
        row["eq_path_edge"] = row["eq_tau_path"] / 2
        rows.append(row)
    return pd.DataFrame(rows)