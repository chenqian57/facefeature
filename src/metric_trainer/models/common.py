from torch import nn


def get_activation(name="silu"):
    if name == "silu":
        module = nn.SiLU
    elif name == "relu":
        module = nn.ReLU
    elif name == "lrelu":
        module = nn.LeakyReLU
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


def get_norm(name="bn"):
    if name == "bn":
        module = nn.BatchNorm2d
    elif name == "ln":
        module = nn.LayerNorm
    elif name == "gn":
        module = nn.GroupNorm
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module
