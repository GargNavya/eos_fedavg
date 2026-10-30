"""Eigenvector alignment between the global Hessian and the client Hessians.

Global loss = Σ pᵢ · client loss (pᵢ = data share), so H = Σ pᵢ Hᵢ. For the global top eigenvector v:
    λ_global = vᵀ H v = Σ pᵢ (vᵀ Hᵢ v)          (identity; checks the solver)
    vᵀ Hᵢ v ≤ λᵢ  ⇒  λ_global ≤ Σ pᵢ λᵢ          (Weyl; the gap = misalignment cost)
Subspace overlap of two k-dim orthonormal bases U, V: ||Uᵀ V||_F² / k ∈ [0, 1]
(1 = same subspace). Used instead of single-vector cosines because top eigenvalues can be near-degenerate.
"""
import numpy as np
import torch

from .eig import eigsh_torch
from .roundmap import hessian_mv


def topk(matvec, like, k, tol=1e-4, maxiter=None):
    """Top-k eigenpairs (descending) of a symmetric operator. Returns (values, orthonormal columns)."""
    vals, vecs = eigsh_torch(matvec, like, k, "LA", None, tol, maxiter)
    order = np.argsort(vals)[::-1]
    V = torch.as_tensor(np.ascontiguousarray(vecs[:, order]), dtype=like.dtype, device=like.device)
    if V.shape[1]:
        V, _ = torch.linalg.qr(V)  # re-orthonormalize in working precision
    return vals[order], V


def subspace_overlap(U, V):
    k = min(U.shape[1], V.shape[1])
    if k == 0:
        return float("nan")
    return float((U.T @ V).pow(2).sum() / k)


def alignment(fm, w, X, Y, clients, p, k=5, tol=1e-4, maxiter=None):
    lam_g, Vg = topk(hessian_mv(fm, w, X, Y), w, k, tol, maxiter)
    out_clients, bases = [], []
    for (Xi, Yi), pi in zip(clients, p):
        mv = hessian_mv(fm, w, Xi, Yi)
        lam_i, Vi = topk(mv, w, k, tol, maxiter)
        rayleigh = [float(Vg[:, j] @ mv(Vg[:, j])) for j in range(Vg.shape[1])]
        out_clients.append({
            "p": float(pi),
            "lam": [float(x) for x in lam_i],
            "rayleigh_global_vecs": rayleigh,          # v_gjᵀ Hᵢ v_gj for the global top-k vectors
            "overlap_with_global": subspace_overlap(Vg, Vi),
            "cos2_top": float((Vg[:, 0] @ Vi[:, 0]) ** 2) if Vi.shape[1] and Vg.shape[1] else float("nan"),
        })
        bases.append(Vi)
    M = len(bases)
    pair = [[subspace_overlap(bases[i], bases[j]) for j in range(M)] for i in range(M)]
    lam_top = float(lam_g[0]) if len(lam_g) else float("nan")
    from_rayleigh = float(sum(c["p"] * c["rayleigh_global_vecs"][0] for c in out_clients)) if len(lam_g) else float("nan")
    weyl = float(sum(c["p"] * c["lam"][0] for c in out_clients if c["lam"]))
    return {
        "k": k,
        "lam_global": [float(x) for x in lam_g],
        "lam_global_from_rayleigh": from_rayleigh,     # should equal lam_global[0]
        "weyl_sum": weyl,                              # Σ pᵢ λᵢ
        "misalignment_cost": weyl - lam_top,           # Σ pᵢ (λᵢ − v_gᵀ Hᵢ v_g)
        "clients": out_clients,
        "pairwise_overlap": pair,
    }