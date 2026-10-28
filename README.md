# eos_fedavg

Where is the Edge of Stability for FedAvg? Predicting the stability threshold of federated training via the linearized round map.

## Setup (airavat)

```bash
source /22tdata/priya/oml_nav_rlmerge/fedavg_analysis/env.sh
cd $M/eos_fedavg
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
pytest -q
```

## Layout

```
eosfl/            library
  config.py       Config dataclass, YAML loading, CLI overrides, run naming
  data.py         CIFAR-10 / synthetic data, balanced subset, Dirichlet split
  models.py       MLP, CNN
  functional.py   FlatModel: loss, grad, Hessian-vector product on a flat vector
  eig.py          Lanczos (ARPACK) eigenvalues with warm starts
  roundmap.py     linearized FedAvg round operator T = I - R
  fedavg.py       training loop + measurements
  toy.py          exact 2D quadratic check
  analysis.py     load runs, equilibrium values
scripts/          entry points (run, sweep, toy_check, plot)
configs/          run configs (base -> centralized / fedavg -> debug)
sweeps/           grids for sweep.py
tests/            correctness checks
```

Outputs go to `$RESULTS_DIR/<exp>/<run_name>/`: `config.json`, `split.json`, `metrics.jsonl`, `status.json`, `stdout.log`.

## Usage

```bash
# toy check (CPU, ~1 min)
python scripts/toy_check.py

# smoke test (CPU, seconds)
python scripts/run.py --config configs/debug.yaml

# single run, any key overridable
nvidia-smi                                   # pick a free GPU
CUDA_VISIBLE_DEVICES=1 python scripts/run.py --config configs/centralized.yaml --set lr=0.04

# sweeps (resumable; preview with --dry-run)
python scripts/sweep.py sweeps/centralized.yaml --gpus 1 --jobs-per-gpu 3
python scripts/sweep.py sweeps/main.yaml --gpus 0,1,2,3 --jobs-per-gpu 4

# plots + summary.csv into $RESULTS_DIR/<exp>/_plots
python scripts/plot.py fedavg_main
```

## Measured quantities

| column | meaning | edge at |
|---|---|---|
| `sharpness` | top eigenvalue of global Hessian | `sharpness · lr / 2 = 1` |
| `client_sharp`, `client_sharp_max` | per-client top eigenvalues | `· lr / 2 = 1` |
| `tau_max`, `tau_min` | extreme eigenvalues of `T = server_lr (I − Σ p_i (I − lr(H_i + μI))^K)` | stable iff all eigenvalues in (0, 2): `tau_max / 2 = 1`, `tau_min = 0` |

`ρ(R)` itself is always ≥ ~1 because flat directions of the loss give eigenvalue ≈ 1, so we track `T = I − R` instead.
