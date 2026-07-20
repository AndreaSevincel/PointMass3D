#bidirectional RRT with a greedy connect heuristic, plus smoothing and resampling


import time

import numpy as np


class _Tree:
    def __init__(self, root, capacity):
        self.nodes = np.empty((capacity, 3))
        self.nodes[0] = root
        self.parents = np.full(capacity, -1, dtype=int)
        self.size = 1

    def nearest(self, q):
        q = np.asarray(q, dtype=float)
        diff = self.nodes[: self.size] - q
        d2 = np.einsum("ij,ij->i", diff, diff)
        return int(np.argmin(d2))

    def add(self, q, parent):
        self.nodes[self.size] = q
        self.parents[self.size] = parent
        self.size += 1
        return self.size - 1

    def path_to_root(self, idx):
        path = []
        while idx != -1:
            path.append(self.nodes[idx].copy())
            idx = self.parents[idx]
        return path  # node -> .. -> root


def _extend(env, tree, q_target, step, margin, resolution):
    i_near = tree.nearest(q_target)
    q_near = tree.nodes[i_near]
    d = np.linalg.norm(q_target - q_near)
    if d < 1e-9:
        return "reached", i_near
    if d <= step:
        q_new = q_target
    else:
        q_new = q_near + (q_target - q_near) * (step / d)
    if not env.segment_free(q_near, q_new, margin, resolution):
        return "trapped", None
    idx = tree.add(q_new, i_near)
    return ("reached" if d <= step else "advanced"), idx


def rrt_connect(
    env,
    start,
    goal,
    step=0.10,
    max_iters=10000,
    margin=0.0,
    resolution=0.01,
    rng=None,
):
    #Returns (path, info) where path has shape (M, 3) or is None on failure.
    rng = np.random.default_rng(rng)
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    if env.in_collision(start, margin) or env.in_collision(goal, margin):
        raise ValueError("start or goal is in collision")

    t0 = time.perf_counter()
    capacity = max_iters + 2
    tree_a, tree_b = _Tree(start, capacity), _Tree(goal, capacity)
    a_is_start = True

    for it in range(max_iters):
        q_rand = rng.uniform(env.lo, env.hi, size=3)
        status, idx_a = _extend(env, tree_a, q_rand, step, margin, resolution)
        if status != "trapped":
            q_new = tree_a.nodes[idx_a]
            while True:
                status_b, idx_b = _extend(env, tree_b, q_new, step, margin, resolution)
                if status_b != "advanced":
                    break
            if status_b == "reached":
                path_a = tree_a.path_to_root(idx_a)[::-1]  # root -> connect point
                path_b = tree_b.path_to_root(idx_b)  # connect point -> root
                # The connect point is present in both trees; drop one copy.
                path = path_a + path_b[1:] if len(path_b) > 1 else path_a
                if not a_is_start:
                    path = path[::-1]
                info = {
                    "iterations": it + 1,
                    "n_nodes": tree_a.size + tree_b.size,
                    "time": time.perf_counter() - t0,
                }
                return np.array(path), info
        tree_a, tree_b = tree_b, tree_a
        a_is_start = not a_is_start

    return None, {
        "iterations": max_iters,
        "n_nodes": tree_a.size + tree_b.size,
        "time": time.perf_counter() - t0,
    }


def shortcut(env, path, iters=200, margin=0.0, resolution=0.01, rng=None):
    """Random shortcut smoothing: repeatedly try replacing a sub-path by a
    straight segment between two random waypoints."""
    rng = np.random.default_rng(rng)
    path = [np.asarray(p, dtype=float) for p in path]
    for _ in range(iters):
        if len(path) < 3:
            break
        i = rng.integers(0, len(path) - 2)
        j = rng.integers(i + 2, len(path))
        if env.segment_free(path[i], path[j], margin, resolution):
            path = path[: i + 1] + path[j:]
    return np.array(path)


def resample_path(path, n):
    """Resample a piecewise-linear path to n waypoints, equally spaced in
    arc length. Keeps the endpoints exactly."""
    path = np.asarray(path, dtype=float)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-12:
        return np.tile(path[0], (n, 1))
    targets = np.linspace(0.0, s[-1], n)
    out = np.empty((n, 3))
    for k in range(3):
        out[:, k] = np.interp(targets, s, path[:, k])
    out[0], out[-1] = path[0], path[-1]
    return out
