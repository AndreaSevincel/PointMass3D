"""Simple trajectory quality metrics."""

import numpy as np


def path_length(path):
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def mean_sq_accel(path):
    """Mean squared second difference — lower is smoother."""
    acc = path[2:] - 2.0 * path[1:-1] + path[:-2]
    return float(np.mean(np.sum(acc**2, axis=1)))


def min_clearance(env, path, resolution=0.005):
    """Minimum robot clearance along the densely-interpolated path."""
    path = np.asarray(path, dtype=float)
    best = np.inf
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / resolution)) + 1)
        pts = np.linspace(a, b, n)
        best = min(best, float(env.clearance(pts).min()))
    return best
