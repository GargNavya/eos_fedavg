import math

import numpy as np


def balanced_subset(y, n, rng):
    if n == 0:
        return np.array([], dtype=int)
    if n < 0 or n >= len(y):
        return np.arange(len(y))
    classes = np.unique(y)
    per = n // len(classes)
    idx = [rng.choice(np.where(y == c)[0], per, replace=False) for c in classes]
    return np.sort(np.concatenate(idx))


def one_hot(y, C):
    Y = np.zeros((len(y), C))
    Y[np.arange(len(y)), y] = 1.0
    return Y


def _cifar10(cfg, rng):
    from torchvision.datasets import CIFAR10

    root = cfg.data_root()
    tr = CIFAR10(root, train=True, download=True)
    te = CIFAR10(root, train=False, download=True)
    ytr, yte = np.asarray(tr.targets), np.asarray(te.targets)
    itr, ite = balanced_subset(ytr, cfg.n_train, rng), balanced_subset(yte, cfg.n_test, rng)
    Xtr = tr.data[itr].astype(np.float64) / 255.0
    Xte = te.data[ite].astype(np.float64) / 255.0
    mean, std = Xtr.mean(axis=(0, 1, 2)), Xtr.std(axis=(0, 1, 2))
    Xtr = ((Xtr - mean) / std).transpose(0, 3, 1, 2)
    Xte = ((Xte - mean) / std).transpose(0, 3, 1, 2) if len(ite) else Xte.reshape(0, 3, 32, 32)
    return Xtr, ytr[itr], Xte, yte[ite], 10


def _synthetic(cfg, rng):
    C, shape = 10, (1, 8, 8)
    d = int(np.prod(shape))
    teacher = rng.standard_normal((d, C))

    def gen(n):
        X = rng.standard_normal((max(n, 0), d))
        return X.reshape(-1, *shape), (X @ teacher).argmax(1)

    Xtr, ytr = gen(cfg.n_train if cfg.n_train > 0 else 1000)
    Xte, yte = gen(cfg.n_test if cfg.n_test >= 0 else 1000)
    return Xtr, ytr, Xte, yte, C


def load_dataset(cfg):
    rng = np.random.default_rng(cfg.data_seed)
    loaders = {"cifar10": _cifar10, "synthetic": _synthetic}
    if cfg.dataset not in loaders:
        raise ValueError(f"unknown dataset: {cfg.dataset}")
    Xtr, ytr, Xte, yte, C = loaders[cfg.dataset](cfg, rng)
    return dict(
        Xtr=Xtr, ytr=ytr, Ytr=one_hot(ytr, C),
        Xte=Xte, yte=yte, Yte=one_hot(yte, C),
        in_shape=tuple(Xtr.shape[1:]), num_classes=C,
    )


def dirichlet_split(y, M, alpha, rng, min_size=1, max_tries=1000):
    n = len(y)
    if M == 1:
        return [np.arange(n)]
    if math.isinf(alpha):
        return [np.sort(a) for a in np.array_split(rng.permutation(n), M)]
    classes = np.unique(y)
    for _ in range(max_tries):
        parts = [[] for _ in range(M)]
        for c in classes:
            idx = rng.permutation(np.where(y == c)[0])
            props = rng.dirichlet(alpha * np.ones(M))
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for i, chunk in enumerate(np.split(idx, cuts)):
                parts[i].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return [np.sort(np.asarray(p, dtype=int)) for p in parts]
    raise RuntimeError("min_client_size not met; lower it or raise alpha")
