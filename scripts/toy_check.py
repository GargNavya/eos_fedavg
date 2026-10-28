"""Predicted vs simulated divergence threshold on 2D quadratic clients.

python scripts/toy_check.py
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from eosfl.toy import make_clients, threshold

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ks", default="1,2,3,5,10,20")
    ap.add_argument("--hets", default="0,0.3,0.6,0.9")
    ap.add_argument("--slrs", default="0.5,1,1.5")
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--out", default=os.path.join(os.environ.get("RESULTS_DIR", "results"), "toy"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for het in map(float, a.hets.split(",")):
        Hs, cs, p = make_clients(het=het)
        for slr in map(float, a.slrs.split(",")):
            for K in map(int, a.Ks.split(",")):
                rows.append({"het": het, "server_lr": slr, "K": K,
                             **threshold(Hs, cs, p, K, slr, rounds=a.rounds)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "toy.csv"), index=False)
    pd.set_option("display.width", 200)
    print(df[["het", "server_lr", "K", "lr_pred", "lr_emp", "rel_err",
              "glob_norm", "client_norm", "glob_norm_slr"]].round(4).to_string(index=False))
    print(f"\nmax rel_err (pred vs sim): {df.rel_err.max():.4f}")

    slrs = sorted(df.server_lr.unique())
    fig, axes = plt.subplots(1, len(slrs), figsize=(5 * len(slrs), 4), sharey=True)
    for ax, slr in zip(axes if len(slrs) > 1 else [axes], slrs):
        for het, g in df[df.server_lr == slr].groupby("het"):
            line, = ax.plot(g.K, g.glob_norm, "o-", label=f"het={het:g} global")
            ax.plot(g.K, g.client_norm, "s--", color=line.get_color(), alpha=0.6)
        ax.axhline(1, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("K (local steps)")
        ax.set_title(f"server_lr={slr:g}")
    axes_list = axes if len(slrs) > 1 else [axes]
    axes_list[0].set_ylabel("lr* · λ / 2   (solid: global, dashed: max client)")
    axes_list[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "toy.png"), dpi=150)
    print(f"saved to {a.out}")
