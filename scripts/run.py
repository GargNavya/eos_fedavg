"""Single run.

python scripts/run.py --config configs/fedavg.yaml --set local_steps=5 alpha=0.1 lr=0.02
"""
import argparse

from eosfl.config import load_config
from eosfl.fedavg import train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    train(load_config(args.config, args.set))
