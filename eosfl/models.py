import math

import torch.nn as nn

ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "gelu": nn.GELU}


def mlp(in_shape, num_classes, width, depth, act):
    layers, d = [nn.Flatten()], math.prod(in_shape)
    for _ in range(depth):
        layers += [nn.Linear(d, width), act()]
        d = width
    layers.append(nn.Linear(d, num_classes))
    return nn.Sequential(*layers)


def cnn(in_shape, num_classes, width, depth, act):
    c, h, w = in_shape
    layers = []
    for _ in range(depth):
        layers += [nn.Conv2d(c, width, 3, padding=1), act(), nn.MaxPool2d(2)]
        c, h, w = width, h // 2, w // 2
    layers += [nn.Flatten(), nn.Linear(c * h * w, num_classes)]
    return nn.Sequential(*layers)


MODELS = {"mlp": mlp, "cnn": cnn}


def build_model(cfg, in_shape, num_classes):
    if cfg.model not in MODELS:
        raise ValueError(f"unknown model: {cfg.model}")
    return MODELS[cfg.model](in_shape, num_classes, cfg.width, cfg.depth, ACTIVATIONS[cfg.activation])
