#RRT-Connect in SE(3), plus shortcutting and resampling.

#Same algorithm as the point-mass domain, with three things that change once
#the configuration space is SE(3) rather than R^3:

#  * the metric mixes a length and an angle, so the angle is converted to the
#    distance the body's farthest point sweeps (angle * extent). Without that
#    the weight is an arbitrary constant and the tree grows anisotropically.
#  * steering interpolates geodesically in rotation (slerp), not linearly in
#    any coordinate chart.
#  * local collision checks resolve rotation at the same swept resolution.

import time

import numpy as np

from .rotation import geodesic_angle, interpolate, rand_rotation, slerp


class _Tree:
    def __init__(self, p0, R0):
        self.pos = [np.asarray(p0, dtype=float)]
        self.rot = [np.asarray(R0, dtype=float)]
        self.parent = [-1]

    def nearest(self, p, R, w):
        d = [np.linalg.norm(p - self.pos[i]) + w * geodesic_angle(self.rot[i], R)
             for i in range(len(self.pos))]
        return int(np.argmin(d))

    def add(self, p, R, parent):
        self.pos.append(np.asarray(p, dtype=float))
        self.rot.append(np.asarray(R, dtype=float))
        self.parent.append(parent)
        return len(self.pos) - 1

    def path_to_root(self, idx):
        out = []
        while idx != -1:
            out.append((self.pos[idx], self.rot[idx]))
            idx = self.parent[idx]
        return out[::-1]


def _steer(pa, Ra, pb, Rb, step, w):
    #Move from a towards b by at most `step` of mixed distance.
    d = np.linalg.norm(pb - pa) + w * geodesic_angle(Ra, Rb)
    if d <= step or d < 1e-12:
        return np.asarray(pb, dtype=float), np.asarray(Rb, dtype=float)
    u = step / d
    return (1 - u) * pa + u * pb, slerp(Ra, Rb, u)


def _extend(env, tree, p_t, R_t, step, w, margin, resolution):
    i = tree.nearest(p_t, R_t, w)
    p_new, R_new = _steer(tree.pos[i], tree.rot[i], p_t, R_t, step, w)
    if not env.segment_free(tree.pos[i], tree.rot[i], p_new, R_new, margin, resolution):
        return None, "trapped"
    j = tree.add(p_new, R_new, i)
    reached = (np.linalg.norm(p_new - p_t) + w * geodesic_angle(R_new, R_t)) < 1e-9
    return j, ("reached" if reached else "advanced")


def rrt_connect_se3(env, start, goal, step=0.12, margin=0.0, resolution=0.02,
                    max_iters=6000, rng=None, timeout=None):
    """start/goal are (position, rotation) pairs. Returns (positions, rotations)
    or (None, None) on failure, plus an info dict."""
    rng = np.random.default_rng(rng)
    (ps, Rs), (pg, Rg) = start, goal
    #rotation is weighted by the body extent so an angle and a length are
    #compared in the same units: the distance the farthest body point moves
    w = env.body.extent
    t0 = time.perf_counter()

    if not env.pose_free(ps, Rs, margin) or not env.pose_free(pg, Rg, margin):
        return None, None, {"success": False, "reason": "endpoint in collision",
                            "time": time.perf_counter() - t0, "iters": 0}

    ta, tb = _Tree(ps, Rs), _Tree(pg, Rg)
    m = env.body.extent
    for it in range(max_iters):
        if timeout is not None and time.perf_counter() - t0 > timeout:
            return None, None, {"success": False, "reason": "timeout",
                                "time": time.perf_counter() - t0, "iters": it}
        p_rand = rng.uniform(env.lo + m, env.hi - m, size=3)
        R_rand = rand_rotation(rng)

        j, status = _extend(env, ta, p_rand, R_rand, step, w, margin, resolution)
        if status != "trapped":
            #connect the other tree all the way to the new node
            while True:
                k, st = _extend(env, tb, ta.pos[j], ta.rot[j], step, w, margin, resolution)
                if st != "advanced":
                    break
            if st == "reached":
                pa = ta.path_to_root(j)
                pb = tb.path_to_root(k)[::-1]
                seq = pa + pb[1:]
                pos = np.array([s[0] for s in seq])
                rot = np.array([s[1] for s in seq])
                #tree b grows from the goal, so if the swap count is odd the
                #concatenation runs goal->start and must be reversed
                if not np.allclose(pos[0], ps):
                    pos, rot = pos[::-1].copy(), rot[::-1].copy()
                return pos, rot, {"success": True, "iters": it,
                                  "time": time.perf_counter() - t0,
                                  "nodes": len(ta.pos) + len(tb.pos)}
        ta, tb = tb, ta

    return None, None, {"success": False, "reason": "max_iters",
                        "time": time.perf_counter() - t0, "iters": max_iters}


def shortcut_se3(env, pos, rot, iters=200, margin=0.0, resolution=0.02, rng=None):
    #Randomised shortcutting on the mixed metric. Waypoints are dropped only
    #when the straight-and-geodesic connection between the survivors is free.
    rng = np.random.default_rng(rng)
    pos = [p.copy() for p in pos]
    rot = [r.copy() for r in rot]
    for _ in range(iters):
        if len(pos) <= 2:
            break
        i, j = sorted(rng.integers(0, len(pos), size=2))
        if j - i < 2:
            continue
        if env.segment_free(pos[i], rot[i], pos[j], rot[j], margin, resolution):
            pos = pos[:i + 1] + pos[j:]
            rot = rot[:i + 1] + rot[j:]
    return np.array(pos), np.array(rot)


def resample_se3(pos, rot, n):
    #Resample to exactly n poses, uniformly in mixed arc length so that
    #waypoint density is even in the space the planner actually measures.
    pos = np.asarray(pos, dtype=float)
    rot = np.asarray(rot, dtype=float)
    if len(pos) == 1:
        return np.repeat(pos, n, axis=0), np.repeat(rot[None, 0], n, axis=0)

    seg = np.array([
        np.linalg.norm(pos[i + 1] - pos[i]) + geodesic_angle(rot[i], rot[i + 1])
        for i in range(len(pos) - 1)
    ])
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total < 1e-12:
        return np.repeat(pos[None, 0], n, axis=0), np.repeat(rot[None, 0], n, axis=0)

    targets = np.linspace(0.0, total, n)
    out_p, out_R = [], []
    for t in targets:
        k = int(np.clip(np.searchsorted(cum, t) - 1, 0, len(seg) - 1))
        u = 0.0 if seg[k] < 1e-12 else (t - cum[k]) / seg[k]
        out_p.append((1 - u) * pos[k] + u * pos[k + 1])
        out_R.append(slerp(rot[k], rot[k + 1], u))
    return np.array(out_p), np.array(out_R)


def plan_se3(env, start, goal, n_waypoints=64, rng=None, timeout=None,
             shortcut_iters=200):
    """Full expert pipeline: RRT-Connect, shortcut, resample."""
    pos, rot, info = rrt_connect_se3(env, start, goal, rng=rng, timeout=timeout)
    if pos is None:
        return None, None, info
    pos, rot = shortcut_se3(env, pos, rot, iters=shortcut_iters, rng=rng)
    pos, rot = resample_se3(pos, rot, n_waypoints)
    info["success"] = env.path_free(pos, rot)
    return pos, rot, info
