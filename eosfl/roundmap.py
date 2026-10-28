"""Linearized FedAvg round.

Near w, client i's K local steps act on a perturbation v as (I - lr (H_i + mu I))^K v,
with H_i frozen at the round-start point w. Aggregation gives
    A = sum_i p_i (I - lr (H_i + mu I))^K,   R = I - server_lr (I - A).
We work with T = I - R = server_lr (I - A), the round analogue of lr * H:
    stable  <=>  every eigenvalue of T lies in (0, 2)
    edge index = tau_max / 2  (equals lr * sharpness / 2 for plain GD)
    tau_min < 0 flags non-oscillatory blow-up (even K, a client past its own edge).
"""
import torch


def hessian_mv(fm, w, X, Y):
    return lambda v: fm.hvp(w, X, Y, v)


def local_map_mv(fm, w, X, Y, lr, K, mu=0.0):
    def mv(v):
        u = v
        for _ in range(K):
            u = u - lr * (fm.hvp(w, X, Y, u) + mu * u)
        return u
    return mv


def roundmap_T_mv(fm, w, clients, p, lr, K, server_lr, mu=0.0):
    maps = [local_map_mv(fm, w, X, Y, lr, K, mu) for X, Y in clients]

    def mv(v):
        acc = torch.zeros_like(v)
        for m, pi in zip(maps, p):
            acc = acc + pi * m(v)
        return server_lr * (v - acc)
    return mv
