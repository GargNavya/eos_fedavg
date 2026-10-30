"""Standard plots for one experiment folder.

python scripts/plot.py fedavg_main
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eosfl.analysis import load_exp, load_metrics


def trajectories(df, out, lr=None):
    if lr is None:
        lr = sorted(df.lr.unique())[0]
    sub = df[(df.lr == lr) & (df.seed == df.seed.min()) & (df.server_lr == 1.0)]
    alphas = sorted(sub.alpha.unique(), reverse=True)
    if not alphas:
        return
    fig, axes = plt.subplots(1, len(alphas), figsize=(5 * len(alphas), 4), sharey=True, squeeze=False)
    for ax, a in zip(axes[0], alphas):
        for _, r in sub[sub.alpha == a].sort_values("local_steps").iterrows():
            m = load_metrics(r.run_dir)
            if "sharpness" in m:
                s = m.dropna(subset=["sharpness"])
                line, = ax.plot(s.steps, s.sharpness * lr / 2, label=f"K={r.local_steps}")
                if "client_sharp_max" in m:
                    c = m.dropna(subset=["client_sharp_max"])
                    ax.plot(c.steps, c.client_sharp_max * lr / 2, "--", color=line.get_color(), alpha=0.5)
        ax.axhline(1, color="k", lw=0.8)
        ax.set_title(f"alpha={a:g}, lr={lr:g}")
        ax.set_xlabel("local steps")
    axes[0, 0].set_ylabel("sharpness · lr / 2  (dashed: max client)")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"traj_lr{lr:g}.png"), dpi=150)
    plt.close(fig)


def equilibrium_vs_K(df, out):
    done = df[df.status != "diverged"]
    done = done[done.server_lr == 1.0]
    cols = [("eq_sharp_norm", "global sharpness · lr/2"),
            ("eq_client_norm", "max client sharpness · lr/2"),
            ("eq_ceff_frozen", "frozen round map, per-step c_eff"),
            ("eq_ceff_path", "path round map, per-step c_eff")]
    cols = [(c, t) for c, t in cols if done[c].notna().any()]
    if not cols:
        return
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4), squeeze=False)
    for ax, (c, title) in zip(axes[0], cols):
        for a, g in done.groupby("alpha"):
            st = g.groupby("local_steps")[c].agg(["mean", "std"]).reset_index()
            ax.errorbar(st.local_steps, st["mean"], yerr=st["std"], marker="o", capsize=3, label=f"alpha={a:g}")
        ax.axhline(1, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("K (local steps)")
        ax.set_title(title)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "equilibrium_vs_K.png"), dpi=150)
    plt.close(fig)


def edge_trajectories(df, out):
    sub = df[df.eq_ceff_path.notna() & (df.seed == df.seed.min())]
    if sub.empty:
        return
    alphas = sorted(sub.alpha.unique(), reverse=True)
    fig, axes = plt.subplots(1, len(alphas), figsize=(5 * len(alphas), 4), sharey=True, squeeze=False)
    for ax, a in zip(axes[0], alphas):
        for _, r in sub[sub.alpha == a].sort_values(["lr", "local_steps"]).iterrows():
            m = load_metrics(r.run_dir)
            f = m.dropna(subset=["ceff_frozen"])
            line, = ax.plot(f.steps, f.ceff_frozen, "o-", label=f"K={r.local_steps} lr={r.lr:g}")
            g = m.dropna(subset=["ceff_path"])
            ax.plot(g.steps, g.ceff_path, "s--", color=line.get_color())
        ax.axhline(1, color="k", lw=0.8)
        ax.set_title(f"alpha={a:g}")
        ax.set_xlabel("local steps")
    axes[0, 0].set_ylabel("per-step c_eff (solid: frozen, dashed: path)")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "edge_traj.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--results", default=os.environ.get("RESULTS_DIR", "results"))
    ap.add_argument("--frac", type=float, default=0.3, help="tail fraction used for equilibrium")
    a = ap.parse_args()
    exp_dir = os.path.join(a.results, a.exp)
    out = os.path.join(exp_dir, "_plots")
    os.makedirs(out, exist_ok=True)
    df = load_exp(exp_dir, a.frac)
    df.drop(columns=["run_dir"]).to_csv(os.path.join(out, "summary.csv"), index=False)
    for lr in sorted(df.lr.unique()):
        trajectories(df, out, lr)
    equilibrium_vs_K(df, out)
    edge_trajectories(df, out)
    print(df.groupby(["alpha", "local_steps"])[["eq_sharp_norm", "eq_client_norm", "eq_ceff_frozen", "eq_ceff_path"]].mean().round(3))
    print(f"saved to {out}")