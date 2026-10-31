# Where Is the Edge for FedAvg?

Edge of Stability (EoS) in Federated Averaging, and a stability criterion from the linearized round map.

Full-batch GD trains at the edge: sharpness λ_max(H) rises to about 2/η and stays there. This repo asks where the edge lies in FedAvg, and how the local steps K, data heterogeneity α and the server step η_s move it.

Throughout, **S = η·λ_max / 2** (normalized sharpness). S = 1 means exactly 2/η.

## 1. Idea in one paragraph

Freeze every client Hessian Hᵢ at the server point. One FedAvg round is then linear:

```
A = Σ pᵢ (I − η Hᵢ)^K          pᵢ = client data share
R = I − η_s (I − A)            round map
T = I − R = η_s (I − A)        what we track
```

The round is stable iff every eigenvalue τ of T lies in (0, 2). We track T, not ρ(R), because flat directions give ρ(R) ≳ 1 always. Growth per round is g = max|1 − τ|; growth per local step is c_eff = (1 + g^{1/K}) / 2. For GD (K = 1, η_s = 1), c_eff = S.

Loss: MSE on one-hot targets, F_i(w) = mean over samples of ½‖f(x; w) − y‖²; global F = Σ pᵢ Fᵢ. Model: tanh MLP, 2 hidden layers of width 200, on a CIFAR-10 subset.

## 2. Setup

Needs Python ≥ 3.9 and an NVIDIA GPU for the network experiments (CPU is enough for the quadratic check, tests and the smoke run).

```bash
git clone <this repo> && cd eos_fedavg
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt        # torch 2.6.0 (CUDA 12.4 wheels), installs eosfl as editable package
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
pytest -q                              # correctness tests, CPU
```

For a different CUDA version, edit the `--extra-index-url` line in `requirements.txt`.

Optional environment variables:

```bash
export DATA_DIR=/path/to/data          # CIFAR-10 download location (default ./data)
export RESULTS_DIR=/path/to/results    # run outputs (default ./results); keep outside git
```

Pick a free GPU with `nvidia-smi` and select it with `CUDA_VISIBLE_DEVICES`.

## 3. Layout

```
eosfl/            library
  config.py       Config dataclass, YAML loading with `base:` inheritance, CLI overrides, run naming
  data.py         CIFAR-10 / synthetic data, balanced subset, Dirichlet(α) client split
  models.py       tanh MLP, small CNN
  functional.py   FlatModel: loss, gradient, Hessian-vector product on a flat parameter vector
  eig.py          Lanczos / ARPACK eigenvalues with warm starts
  roundmap.py     frozen round map T and path-linearized round map
  alignment.py    top-k eigenspaces of global and client Hessians, overlaps, misalignment
  fedavg.py       training loop and measurements
  toy.py          exact 2-D quadratic check
  analysis.py     load runs, equilibrium values
  obs.py          helpers used by the notebooks
scripts/          entry points: run, sweep, toy_check, plot, keep, align
configs/          run configs (base -> centralized / fedavg -> debug)
sweeps/           grids for sweep.py
notebooks/        analysis notebooks 01-09 (see section 7)
observations/     small results kept in git (created by keep.py)
report/           LaTeX report and slides
tests/            correctness checks
```

Each run writes to `$RESULTS_DIR/<exp>/<run_name>/`: `config.json`, `split.json`, `metrics.jsonl`, `status.json`, `stdout.log` (and `alignment.json`, `final_w.pt` when asked).

## 4. Running things

### Smoke tests

```bash
python scripts/toy_check.py                        # quadratic check, CPU, a few minutes
python scripts/run.py --config configs/debug.yaml  # tiny end-to-end run, CPU, under a minute
```

### One run

Any config key can be overridden with `--set key=value`.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run.py --config configs/fedavg.yaml \
    --set alpha=0.1 local_steps=5 lr=0.02 exp=my_test
```

### A sweep

Grids live in `sweeps/*.yaml`: `base` (config), `set` (fixed overrides), `grid` (crossed values), `cases` (extra per-case overrides, crossed with the grid). Sweeps are resumable: runs with a `status.json` are skipped, so re-running the same command restarts crashed runs.

```bash
python scripts/sweep.py sweeps/main.yaml --dry-run                      # list the runs
python scripts/sweep.py sweeps/main.yaml --gpus 0,1 --jobs-per-gpu 4   # run them
python scripts/sweep.py sweeps/main.yaml --rerun                        # ignore existing results
```

Watch progress (the sweep terminal is silent until a run ends):

```bash
tail -n 3 $RESULTS_DIR/<exp>/*/stdout.log
```

Sweep jobs import the code fresh, so editing `eosfl/` during a sweep changes runs that start later.

### Plots and summary

```bash
python scripts/plot.py <exp>              # plots + summary.csv into $RESULTS_DIR/<exp>/_plots
python scripts/plot.py <exp> --frac 0.3   # equilibrium = median over the last 30% of measurements
```

### Keep the small outputs (for git)

Raw results never go in git. `keep.py` copies plots, `runs.csv`, `trajectories.csv`, `summary.csv` and `splits.json` into `observations/<exp>/`.

```bash
python scripts/keep.py <exp> [--name folder] [--points 200]
git add observations/<exp> && git commit -m "keep <exp>"
```

### Eigenvector alignment (recompute later)

For runs trained with `save_final: true`:

```bash
python scripts/align.py $RESULTS_DIR/fedavg_align/<run_name> [--k 5] [--tol 1e-4]
```

### Notebooks

```bash
pip install jupyter
jupyter notebook notebooks/
```

Notebooks read `$RESULTS_DIR` (or the kept `observations/`) and rebuild the tables and plots of the report.

## 5. Measured quantities

All eigenvalues are reported as `· η/2`, so the classical edge is 1. Measurement frequencies (`sharp_every`, `client_sharp_every`, `roundmap_every`, `roundmap_path_every`) are in **rounds**; 0 = off.

| column | meaning |
|---|---|
| `sharpness` | λ_max of the global Hessian |
| `client_sharp`, `client_sharp_max` | per-client top eigenvalues, and the largest |
| `tau_max`, `tau_min` | τ_max/2 and τ_min of the frozen round map T |
| `tau_path`, `tau_path_im` | eigenvalue of the path-linearized map with largest real part |

## 6. Pitfalls

- **Measurement dominates run time.** Training takes about 6 min per 10k-step run for every K. The round map costs K·N Hessian-vector products per product, and ARPACK needs many products when K is large. Time one run and read `status.json` (`t_train`, `t_measure`, `n_matvec`) before planning a sweep.
- **Frozen round map at K = 20 is very slow** (hours per solve, up to 8 h). Set `roundmap_every=0` there, or cap the solver with `eig_maxiter`.
- **Precision.** TF32 is disabled. GPU runs are float32; use float64 only on CPU.
- **Equilibrium** is the median of the last 30% of measurements. Sharpness still drifts slowly at 10k steps.
- **Training length** is counted in local steps (rounds × K).

## 7. Experiments and their parameters

Common setup: CIFAR-10 subset (5000 train / 1000 test, class balanced, standardized), MSE loss, tanh MLP 2 × 200, N = 10 clients (Dirichlet(α), α = ∞ means IID), full-batch local GD, all clients every round, float32.

| # | Experiment | Command | Parameters |
|---|---|---|---|
| 1 | Quadratic clients (round-map check) | `python scripts/toy_check.py` | 3 clients, 2-D quadratics. K ∈ {1,2,3,5,10,20}, heterogeneity h ∈ {0, 0.3, 0.6, 0.9}, η_s ∈ {0.5, 1, 1.5}. Round-map threshold vs simulated divergence. CPU. Notebook 01 |
| 2 | Centralized GD | `sweeps/centralized.yaml` | 1 client, K = 1, η ∈ {0.02, 0.04, 0.08}, 10⁴ steps. Notebook 02 |
| 3 | Single FedAvg run | `run.py --config configs/fedavg.yaml --set alpha=0.1 local_steps=5 lr=0.02 max_steps=2000` | α = 0.1, K = 5, η = 0.02, η_s = 1, 2000 local steps, round map every 50 rounds. Notebook 03 |
| 4 | Main sweep | `sweeps/main.yaml` | α ∈ {∞, 1, 0.1}, K ∈ {1, 2, 5, 10, 20}, η ∈ {0.02, 0.04}, seeds {0, 1}, η_s = 1, 10⁴ local steps. Global sharpness every 10 rounds, client sharpness every 50, round map every 100 (off for K = 20 at η = 0.04). Notebooks 04, 05 |
| 5 | Real vs frozen round map | `sweeps/path.yaml` | α ∈ {∞, 0.1}, K ∈ {1, 2, 5, 10}, η = 0.02, seed 0, 5000 steps, about 6 round-map measurements per run, solver capped at 6 restarts. Notebook 06 |
| 6 | Eigenvector alignment | `sweeps/alignment.yaml` | α ∈ {∞, 0.1}, K ∈ {5, 10}, η = 0.02, seed 0, 5000 steps, round maps off, top-5 eigenspaces at the end. Notebook 07 |
| 7 | Server step | `sweeps/server_lr.yaml` | IID, η = 0.02, seed 0, 10⁴ steps, η_s ∈ {0.5, 1.5, 2} × K ∈ {1, 2, 5}, round maps off. The η_s = 1 reference comes from experiment 4. Notebook 08 |
| – | Empirical laws | notebook 09 | Pools experiments 2, 4, 5, 6, 7. Fits L1–L4 (section 9) |

Notes on reproducing experiment 4 exactly:
- `sweeps/main.yaml` lists the full grid (η up to 0.08, seeds up to 2). Edit `grid:` to `lr: [0.02, 0.04]` and `seed: [0, 1]` for the grid we ran.
- For K = 20 at η = 0.04 run with `roundmap_every=0`. `sweeps/main_seed2.yaml` is a cheaper variant for seed 2 with sparse measurements and a solver cap.
- `sweeps/fedprox.yaml` (FedProx edge prediction) was written but not run.

Compute (sum of run times; jobs shared GPUs): experiment 2: 2.0 GPU h; 3: 0.08; 4: 43; 5: 1.9; 6: 0.3; 7: 1.4. Total about 49 GPU h on RTX A6000 GPUs.

## 8. Hypotheses and what held

| | Hypothesis | Verdict | Result in short |
|---|---|---|---|
| H1 | Training self-tunes so the frozen round map sits at its edge, τ_max/2 ≈ 1 | Rejected as stated; holds per local step | τ_max/2 reaches 1.5 and g reaches 25 at K = 20, yet every run is stable. The growth per local step is c_eff = 1.077 ± 0.010 over 55 runs, equal to GD's own overshoot (S = 1.07–1.10). Along the real round, odd K keeps the overshoot, even K keeps about 30% of it |
| H2 | IID clients: S ≈ 1 for every K | Supported | S is flat in K (1.061–1.063 for K ≥ 2, 1.079 at K = 1), at GD's level of about 1.07, not exactly 1. Exact on quadratics. Explains why K barely matters for IID |
| H3 | Non-IID: global S below 1, sharpest client at its edge, K effect saturates | Supported qualitatively, refined | Global S falls with heterogeneity (0.98 at α = 0.1 for K ≥ 5). Exact identity λ_max = λ̄ − M: the client average λ̄ ≈ 1.10 is the same for all α, and the misalignment M of sharp directions grows (0.02 IID, 0.09–0.10 at α = 0.1). The sharpest client sits 11–15% above its edge, not at it. K matters mostly between 1 and 2 and saturates by K ≈ 10 |
| H4 | Server step: edge at 1/η_s for K = 1, fades with K, with an even/odd pattern | Supported | K = 1 ratios 1.965 / 0.670 / 0.506 vs predicted 2 / 0.667 / 0.5. Even K ignores η_s. Odd K follows the closed form within 2–3%, except at the cusp (K = 5, η_s = 2): 0.78 vs 0.5 |

Caveats: one architecture and dataset, mostly one η, two seeds (IID server-step runs: one seed), one Dirichlet split per α. Seed 2 of the main sweep is not run.

## 9. Empirical laws (notebook 09)

Values are `· η/2`.

- **L1.** Per-step round-map growth equals GD's overshoot: c_eff = c = 1.077 ± 0.010.
- **L1a.** IID edge in closed form: the smallest x = ηλ with |1 − η_s(1 − (1 − x)^K)| = (2c − 1)^K. Mean error 1.9% over 29 IID runs, against 7.2% for c = 1.
- **L2.** λ_max ≈ c₀ + A/K − M, with c₀ ≈ 1.086 for every α and M = 0.026 / 0.038 / 0.120 for IID / α = 1 / α = 0.1.
- **L3.** Sharpest client ≈ c_max + B/K, with B = 0.04 / 0.09 / 0.33.
- **L4.** Path overshoot c_eff − 1 is about 0.9 (c − 1) for odd K and about 0.3 (c − 1) for even K.

## References

Cohen et al. 2021 (EoS in GD); Damian et al. 2023 (self-stabilization); Lewkowycz et al. 2020 (catapult); McMahan et al. 2017 (FedAvg); Reddi et al. 2021 (server optimizers); Hsu et al. 2019 (Dirichlet split); arXiv 2608.25873 (EoS in federated optimization). Full list in `report/report.pdf`.