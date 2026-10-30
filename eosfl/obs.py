"""Data access for the analysis notebooks.

Every loader prefers the full raw results ($RESULTS_DIR/<exp>/, written by train) and falls
back to the small copies kept in observations/<exp>/ (written by scripts/keep.py). Both paths
return the same columns, so the notebooks run on any machine that has the repo.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from .analysis import add_derived, equilibrium

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBS = os.path.join(ROOT, "observations")
CFG_KEYS = ["num_clients", "alpha", "local_steps", "lr", "server_lr", "prox_mu", "seed", "data_seed"]
EQ_COLS = ["sharpness", "client_sharp_max", "tau_min", "tau_max", "g_frozen", "ceff_frozen",
           "tau_path", "tau_path_min", "g_path", "ceff_path", "train_loss"]


def results_dir(exp):
    return os.path.join(os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results")), exp)


def obs_dir(exp, names=()):
    """observations/<exp>, or a numbered variant such as observations/03_<exp>."""
    for n in (exp, *names):
        d = os.path.join(OBS, n)
        if os.path.isdir(d):
            return d
    if os.path.isdir(OBS):
        for n in sorted(os.listdir(OBS)):
            head, _, rest = n.partition("_")
            if head.isdigit() and rest in (exp, *names) and os.path.isdir(os.path.join(OBS, n)):
                return os.path.join(OBS, n)
    return None


def _has_raw(exp):
    return bool(glob.glob(os.path.join(results_dir(exp), "*", "metrics.jsonl")))


def source(exp):
    if _has_raw(exp):
        return "raw", results_dir(exp)
    d = obs_dir(exp)
    if d and os.path.exists(os.path.join(d, "trajectories.csv")):
        return "obs", d
    raise FileNotFoundError(
        f"No data for '{exp}': neither {results_dir(exp)} nor observations/{exp}/trajectories.csv. "
        f"Run the experiment, or `python scripts/keep.py {exp}` where the results live.")


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _parse_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str) and x.strip():
        return [float(v) for v in x.split()]
    return np.nan


def load_traj(exp):
    """Long table: one row per logged round, with run config columns and derived g / c_eff."""
    kind, d = source(exp)
    frames = []
    if kind == "raw":
        for run_dir in sorted(glob.glob(os.path.join(d, "*", "metrics.jsonl"))):
            run_dir = os.path.dirname(run_dir)
            cfg = _read_json(os.path.join(run_dir, "config.json")) or {}
            with open(os.path.join(run_dir, "metrics.jsonl")) as f:
                m = pd.DataFrame([json.loads(line) for line in f if line.strip()])
            m.insert(0, "run", os.path.basename(run_dir))
            for k in CFG_KEYS:
                m[k] = cfg.get(k)
            frames.append(m)
        t = pd.concat(frames, ignore_index=True)
    else:
        t = pd.read_csv(os.path.join(d, "trajectories.csv"))
        runs = pd.read_csv(os.path.join(d, "runs.csv"))
        t = t.merge(runs[["run"] + [k for k in CFG_KEYS if k in runs]], on="run", how="left")
    if "client_sharp" in t:
        t["client_sharp"] = t["client_sharp"].map(_parse_list)
    t["alpha"] = t["alpha"].astype(float)
    parts = [add_derived(g.copy(), int(g["local_steps"].iloc[0])) for _, g in t.groupby("run", sort=False)]
    return pd.concat(parts, ignore_index=True)


def load_runs(exp, frac=0.3):
    """One row per run: config, status, final metrics, equilibrium (median of last `frac`) values."""
    kind, d = source(exp)
    t = load_traj(exp)
    if kind == "raw":
        status = {}
        for p in glob.glob(os.path.join(d, "*", "status.json")):
            status[os.path.basename(os.path.dirname(p))] = _read_json(p)
        st = pd.DataFrame([{"run": k, **v} for k, v in status.items()])
    else:
        st = pd.read_csv(os.path.join(d, "runs.csv"))
        st = st[[c for c in st.columns if c not in CFG_KEYS]]
    rows = []
    for run, g in t.groupby("run", sort=True):
        row = {"run": run, **{k: g[k].iloc[0] for k in CFG_KEYS if k in g}}
        for c in EQ_COLS:
            row[f"eq_{c}"] = equilibrium(g[c], frac, g["steps"]) if c in g else np.nan
        for c in ["train_loss", "train_acc", "test_loss", "test_acc"]:
            row[f"final_{c}"] = g[c].dropna().iloc[-1] if c in g and g[c].notna().any() else np.nan
        rows.append(row)
    r = pd.DataFrame(rows)
    if len(st):
        r = r.merge(st[[c for c in st.columns if not c.startswith("final_")]], on="run", how="left")
    r["eq_sharp_norm"] = r["eq_sharpness"] * r["lr"] / 2
    r["eq_client_norm"] = r["eq_client_sharp_max"] * r["lr"] / 2
    return r


def load_splits(exp):
    kind, d = source(exp)
    if kind == "obs":
        return _read_json(os.path.join(d, "splits.json")) or {}
    out = {}
    for run_dir in sorted(glob.glob(os.path.join(d, "*"))):
        cfg, sp = _read_json(os.path.join(run_dir, "config.json")), _read_json(os.path.join(run_dir, "split.json"))
        if cfg and sp:
            out.setdefault(f"alpha={cfg['alpha']} M={cfg['num_clients']} data_seed={cfg['data_seed']}", sp)
    return out


def load_alignment(exp):
    """{run_name: alignment dict} from alignment.json files (raw) or observations/<exp>/alignment.json."""
    kind, d = source(exp)
    if kind == "obs":
        return _read_json(os.path.join(d, "alignment.json")) or {}
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*", "alignment.json"))):
        out[os.path.basename(os.path.dirname(p))] = _read_json(p)
    return out


def load_toy():
    for p in [os.path.join(results_dir("toy"), "toy.csv")] + \
             [os.path.join(OBS, n, "toy.csv") for n in ("01_toy", "toy")]:
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError("toy.csv not found: run `python scripts/toy_check.py` (and keep.py toy --name 01_toy).")