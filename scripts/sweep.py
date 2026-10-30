"""Grid sweep across GPUs. Resumable: runs with status.json are skipped.

python scripts/sweep.py sweeps/main.yaml --gpus 0,1,2,3 --jobs-per-gpu 4
python scripts/sweep.py sweeps/main.yaml --dry-run
"""
import argparse
import itertools
import os
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import yaml

from eosfl.config import fmt_value, load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def expand(spec_path):
    with open(spec_path) as f:
        spec = yaml.safe_load(f)
    base = os.path.join(os.path.dirname(spec_path), spec["base"])
    fixed = spec.get("set", {}) or {}
    grid = spec.get("grid", {}) or {}
    cases = spec.get("cases") or [{}]  # list of override dicts, crossed with the grid
    keys = list(grid)
    jobs = []
    for case in cases:
        for combo in itertools.product(*(grid[k] for k in keys)):
            over = {**fixed, **case, **dict(zip(keys, combo))}
            cfg = load_config(base, over)
            args = [f"{k}={fmt_value(v)}" for k, v in over.items()]
            jobs.append((cfg, base, args))
    return jobs


def prepare_data(cfg):
    if cfg.dataset == "cifar10":
        from torchvision.datasets import CIFAR10
        CIFAR10(cfg.data_root(), train=True, download=True)
        CIFAR10(cfg.data_root(), train=False, download=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--jobs-per-gpu", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rerun", action="store_true", help="ignore existing status.json")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads per job")
    a = ap.parse_args()

    jobs = expand(a.spec)
    todo = [j for j in jobs if a.rerun or not os.path.exists(os.path.join(j[0].run_dir(), "status.json"))]
    print(f"{len(jobs)} runs in grid, {len(todo)} to do")
    if a.dry_run:
        for cfg, _, args in todo:
            print(" ", cfg.run_dir(), " ".join(args))
        return
    if not todo:
        return
    prepare_data(todo[0][0])

    slots = queue.Queue()
    for g in a.gpus.split(","):
        for _ in range(a.jobs_per_gpu):
            slots.put(g.strip())

    def run(job):
        cfg, base, args = job
        gpu = slots.get()
        try:
            os.makedirs(cfg.run_dir(), exist_ok=True)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
            for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                env[var] = str(a.threads)
            cmd = [sys.executable, os.path.join(ROOT, "scripts", "run.py"), "--config", base, "--set", *args]
            with open(os.path.join(cfg.run_dir(), "stdout.log"), "w") as out:
                rc = subprocess.call(cmd, env=env, stdout=out, stderr=subprocess.STDOUT)
            print(f"[gpu {gpu}] rc={rc} {cfg.run_name()}", flush=True)
            return rc
        finally:
            slots.put(gpu)

    with ThreadPoolExecutor(max_workers=slots.qsize()) as ex:
        rcs = list(ex.map(run, todo))
    print(f"done: {sum(rc == 0 for rc in rcs)}/{len(rcs)} ok")


if __name__ == "__main__":
    main()