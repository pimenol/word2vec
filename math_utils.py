import numpy as np


def sigmoid(x):
    out = np.empty_like(x)
    pos = x >= 0
    neg = ~pos
    z = np.exp(-x[pos])
    out[pos] = 1.0 / (1.0 + z)
    z = np.exp(x[neg])
    out[neg] = z / (1.0 + z)
    return out


def log_sigmoid(x):
    return -np.logaddexp(np.float32(0.0) if x.dtype == np.float32 else 0.0, -x)
