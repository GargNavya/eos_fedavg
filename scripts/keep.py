"""Copy the small, worth-keeping outputs of one experiment into observations/<exp>/.

python scripts/keep.py fedavg_main
python scripts/keep.py toy --name 01_toy

Writes to observations/<name>/:
  *.png, *.csv        plots and tables from $RESULTS_DIR/<exp>/_plots (or the exp folder itself, e.g. toy)
  runs.csv            one row per run: status, timings, key config, final train/test metrics
  trajectories.csv    every run's measured quantities, thinned to <= --points rows per run
  splits.json         client sizes and label counts, one entry per (alpha, num_clients, data_seed)
Raw metrics.jsonl and logs are never copied.
"""
import argparse
import csv
import glob
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEEP_EXT = (".png", ".csv")
CFG_KEYS = ["num_clients", "alpha", "local_steps", "lr", "server_lr", "prox_mu", "seed", "data_seed"]
TRAJ_COLS = ["round", "steps", "train_loss", "train_acc", "test_loss", "test_acc", "update_norm",
             "sharpness", "client_sharp_max", "client_sharp", "tau_min", "tau_max", "g_frozen",
             "tau_path", "tau_path_im", "tau_path_min", "tau_path_min_im", "g_path"]
MEASURED = ("sharpness", "client_sharp_max", "tau_max", "tau_path", "g_path")


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def read_metrics(run_dir):
    path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def thin(rows, n):
    """Keep every row with a sharpness/round-map measurement, thinned evenly to at most n, plus the last row."""
    keep = [r for r in rows if any(k in r for k in MEASURED)] or rows
    if len(keep) > n:
        step = len(keep) / n
        keep = [keep[int(i * step)] for i in range(n)]
    if rows and (not keep or keep[-1] is not rows[-1]):
        keep.append(rows[-1])
    return keep


def last_value(rows, key):
    for r in reversed(rows):
        if r.get(key) is not None:
            return r[key]
    return None


def write_csv(path, rows):
    if not rows:
        return 0
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", help="experiment folder name under $RESULTS_DIR")
    ap.add_argument("--name", help="folder name under observations/ (default: exp)")
    ap.add_argument("--points", type=int, default=200, help="max trajectory rows per run")
    ap.add_argument("--results", default=os.environ.get("RESULTS_DIR", "results"))
    a = ap.parse_args()

    exp_dir = os.path.join(a.results, a.exp)
    if not os.path.isdir(exp_dir):
        raise SystemExit(f"not found: {exp_dir}")
    dest = os.path.join(ROOT, "observations", a.name or a.exp)
    os.makedirs(dest, exist_ok=True)
    report = []

    src = os.path.join(exp_dir, "_plots")
    if not os.path.isdir(src):
        src = exp_dir
    for f in sorted(os.listdir(src)):
        p = os.path.join(src, f)
        if os.path.isfile(p) and f.endswith(KEEP_EXT):
            shutil.copy2(p, os.path.join(dest, f))
            report.append(f)

    runs, traj, splits = [], [], {}
    for run_dir in sorted(d for d in glob.glob(os.path.join(exp_dir, "*")) if os.path.isdir(d)):
        cfg = read_json(os.path.join(run_dir, "config.json"))
        if cfg is None:
            continue
        name = os.path.basename(run_dir)
        st = read_json(os.path.join(run_dir, "status.json")) or {"status": "unfinished"}
        rows = read_metrics(run_dir)
        runs.append({"run": name, **st, **{k: cfg.get(k) for k in CFG_KEYS},
                     **{f"final_{k}": last_value(rows, k) for k in ("train_loss", "train_acc", "test_loss", "test_acc")}})
        for r in thin(rows, a.points):
            t = {"run": name, **{k: r[k] for k in TRAJ_COLS if k in r}}
            if "client_sharp" in t:
                t["client_sharp"] = " ".join(f"{c:.4g}" for c in t["client_sharp"])
            traj.append(t)
        split = read_json(os.path.join(run_dir, "split.json"))
        key = f"alpha={cfg.get('alpha')} M={cfg.get('num_clients')} data_seed={cfg.get('data_seed')}"
        if split and key not in splits:
            splits[key] = split

    n = write_csv(os.path.join(dest, "runs.csv"), runs)
    if n:
        report.append(f"runs.csv ({n} runs)")
    n = write_csv(os.path.join(dest, "trajectories.csv"), traj)
    if n:
        report.append(f"trajectories.csv ({n} rows)")
    if splits:
        with open(os.path.join(dest, "splits.json"), "w") as f:
            json.dump(splits, f, indent=1)
        report.append(f"splits.json ({len(splits)} splits)")

    print(f"-> {os.path.relpath(dest, ROOT)}")
    for r in report:
        print("  ", r)
    if not report:
        print("   nothing to copy")
    size = sum(os.path.getsize(os.path.join(dest, f)) for f in os.listdir(dest))
    print(f"   total {size / 1e6:.2f} MB")