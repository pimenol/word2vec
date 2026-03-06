import numpy as np


def generate_pairs_vectorized(tokens, max_window, rng):
    n = len(tokens)
    if n < 2:
        return np.empty((0, 2), dtype=np.int32)

    all_c, all_x = [], []
    for d in range(1, max_window + 1):
        m = n - d
        if m <= 0:
            break
        keep_prob = (max_window - d + 1) / max_window
        mask = rng.random(m, dtype=np.float32) < keep_prob
        idx = np.arange(m, dtype=np.int32)
        c = tokens[idx[mask]]
        x = tokens[idx[mask] + d]
        all_c.append(c); all_x.append(x)
        all_c.append(x); all_x.append(c)   # symmetric

    if not all_c:
        return np.empty((0, 2), dtype=np.int32)
    return np.column_stack([np.concatenate(all_c), np.concatenate(all_x)])


def generate_pairs_weighted(tokens, max_window, rng):
    """Like generate_pairs_vectorized but returns (center, context, distance).

    Returns ndarray of shape (N, 3) where column 2 is the distance d in [1, W].
    Used for position-dependent context weighting (1/d).
    """
    n = len(tokens)
    if n < 2:
        return np.empty((0, 3), dtype=np.int32)

    all_c, all_x, all_d = [], [], []
    for d in range(1, max_window + 1):
        m = n - d
        if m <= 0:
            break
        keep_prob = (max_window - d + 1) / max_window
        mask = rng.random(m, dtype=np.float32) < keep_prob
        idx = np.arange(m, dtype=np.int32)
        c = tokens[idx[mask]]
        x = tokens[idx[mask] + d]
        dist = np.full(len(c), d, dtype=np.int32)
        all_c.append(c); all_x.append(x); all_d.append(dist)
        all_c.append(x); all_x.append(c); all_d.append(dist)  # symmetric

    if not all_c:
        return np.empty((0, 3), dtype=np.int32)
    return np.column_stack([np.concatenate(all_c), np.concatenate(all_x),
                            np.concatenate(all_d)])


def generate_pairs_afws(tokens, freqs_f32, max_w, min_w, alpha, rng):
    n = len(tokens)
    if n < 2:
        return np.empty((0, 2), dtype=np.int32)

    f_max = freqs_f32.max()
    tf = freqs_f32[tokens]
    wmax = min_w + (max_w - min_w) * np.power(
        np.clip(1.0 - tf / f_max, 0, 1), alpha)

    all_c, all_x = [], []
    for d in range(1, max_w + 1):
        m = n - d
        if m <= 0:
            break
        idx = np.arange(m, dtype=np.int32)
        cw = wmax[idx]
        eligible = d <= cw
        kp = np.maximum(0.0, (cw - d + 1) / cw).astype(np.float32)
        mask = eligible & (rng.random(m, dtype=np.float32) < kp)
        c = tokens[idx[mask]]
        x = tokens[idx[mask] + d]
        all_c.append(c); all_x.append(x)
        all_c.append(x); all_x.append(c)

    if not all_c:
        return np.empty((0, 2), dtype=np.int32)
    return np.column_stack([np.concatenate(all_c), np.concatenate(all_x)])
