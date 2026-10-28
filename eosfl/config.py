import math
import os
from dataclasses import asdict, dataclass, fields

import yaml


@dataclass
class Config:
    # bookkeeping
    exp: str = "default"
    name: str = ""
    seed: int = 0
    data_seed: int = 0
    device: str = "cuda"
    dtype: str = "float32"
    results_dir: str = ""
    # data
    dataset: str = "cifar10"  # cifar10 | synthetic
    data_dir: str = ""
    n_train: int = 5000  # -1 = all
    n_test: int = 1000  # -1 = all, 0 = none
    num_clients: int = 10
    alpha: float = math.inf  # Dirichlet concentration; inf = IID
    min_client_size: int = 10
    # model
    model: str = "mlp"  # mlp | cnn
    width: int = 200
    depth: int = 2
    activation: str = "tanh"
    # optimization
    lr: float = 0.04
    server_lr: float = 1.0
    local_steps: int = 1
    prox_mu: float = 0.0
    max_rounds: int = 100000
    max_steps: int = 10000  # cap on rounds * local_steps; 0 = off
    target_loss: float = 0.0  # stop below this; 0 = off
    diverge_loss: float = 1e3
    # tracking (every N rounds; 0 = off)
    log_every: int = 1
    sharp_every: int = 10
    client_sharp_every: int = 50
    roundmap_every: int = 0
    eig_tol: float = 1e-3
    print_every: int = 100

    def to_dict(self):
        return asdict(self)

    def run_name(self):
        parts = [self.name] if self.name else []
        parts += [
            f"M{self.num_clients}",
            f"a{self.alpha:g}",
            f"K{self.local_steps}",
            f"lr{self.lr:g}",
            f"slr{self.server_lr:g}",
        ]
        if self.prox_mu:
            parts.append(f"mu{self.prox_mu:g}")
        parts.append(f"s{self.seed}")
        return "_".join(parts)

    def results_root(self):
        return self.results_dir or os.environ.get("RESULTS_DIR", "results")

    def run_dir(self):
        return os.path.join(self.results_root(), self.exp, self.run_name())

    def data_root(self):
        return self.data_dir or os.environ.get("DATA_DIR", "data")


_TYPES = {f.name: f.type for f in fields(Config)}


def _coerce(key, val):
    if key not in _TYPES:
        raise KeyError(f"unknown config key: {key}")
    t = _TYPES[key]
    if t is bool:
        return val if isinstance(val, bool) else str(val).lower() in ("1", "true", "yes")
    if t is int:
        return int(val)
    if t is float:
        return float(val)
    if t is str:
        return str(val)
    return val


def read_yaml(path):
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    base = d.pop("base", None)
    if base:
        merged = read_yaml(os.path.join(os.path.dirname(path), base))
        merged.update(d)
        return merged
    return d


def parse_overrides(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def fmt_value(v):
    if isinstance(v, float) and math.isinf(v):
        return ".inf" if v > 0 else "-.inf"
    return str(v)


def load_config(path=None, overrides=None):
    d = read_yaml(path) if path else {}
    d.update(overrides if isinstance(overrides, dict) else parse_overrides(overrides))
    return Config(**{k: _coerce(k, v) for k, v in d.items()})
