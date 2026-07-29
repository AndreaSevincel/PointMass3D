"""CHOMP

The trajectory is a sequence of N waypoints with fixed endpoints; the
optimizer updates the N-2 interior points. The objective is

    U(xi) = F_obs(xi) + lambda * F_smooth(xi)

where F_smooth is the sum of squared finite-difference velocities and F_obs
integrates a hinge-like obstacle cost c(d) of the SDF clearance d along the
trajectory, weighted by speed (arc length) so the cost is invariant to the
time parametrization. Updates are preconditioned by A^{-1} (the smoothness
metric), which propagates each waypoint's gradient smoothly along the whole
trajectory — the signature "covariant" CHOMP step.
"""

import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def _obstacle_cost(d, eps):
    """CHOMP obstacle cost and its derivative as a function of clearance d."""
    c = np.where(
        d < 0.0,
        -d + 0.5 * eps,
        np.where(d < eps, (d - eps) ** 2 / (2.0 * eps), 0.0),
    )
    cp = np.where(d < 0.0, -1.0, np.where(d < eps, (d - eps) / eps, 0.0))
    return c, cp


def chomp(
    env,
    start,
    goal,
    n_waypoints=64,
    init_path=None,
    iters=500,
    eta=100.0,
    lambda_smooth=1.0,
    eps_obs=0.10,
    tol=1e-5,
    patience=30,
    min_rel_improve=1e-3,
):
    """Optimize a trajectory between start and goal.

    init_path: optional (n_waypoints, 3) initial trajectory (e.g. a resampled
    RRT-Connect path). Defaults to the straight line.

    With the fixed step size used here, the step-norm `tol` criterion rarely
    triggers in practice (the smoothness cost keeps creeping down by a
    negligible amount indefinitely). Once a collision-free iterate has been
    found, stop after `patience` consecutive iterations without a smoothness
    improvement of at least a `min_rel_improve` fraction of the current best
    — empirically >95% of the achievable smoothing happens well before the
    `iters` budget is spent.

    Returns (path, info); path always has shape (n_waypoints, 3). info["success"]
    reports whether the returned path is collision-free (dense check).
    """
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    N = n_waypoints
    M = N - 2  # number of free (interior) waypoints

    if init_path is None:
        traj = np.linspace(start, goal, N)
    else:
        traj = np.asarray(init_path, dtype=float).copy()
        assert traj.shape == (N, 3)
    xi = traj[1:-1].copy()

    # Smoothness: F = 0.5 * sum ||x_{i+1} - x_i||^2 (unit timestep).
    # Gradient w.r.t. interior points is A xi + b with A = tridiag(-1, 2, -1).
    A = 2.0 * np.eye(M) - np.eye(M, k=1) - np.eye(M, k=-1)
    b = np.zeros((M, 3))
    b[0] -= start
    b[-1] -= goal
    A_chol = cho_factor(A)

    t0 = time.perf_counter()
    best_xi, best_cost = None, np.inf
    stale = 0

    for it in range(iters):
        traj = np.vstack([start, xi, goal])

        # --- obstacle functional gradient at interior points
        d = env.clearance(traj)
        c, cp = _obstacle_cost(d, eps_obs)
        g = env.clearance_grad(traj)

        vel = 0.5 * (traj[2:] - traj[:-2])  # central difference
        acc = traj[2:] - 2.0 * traj[1:-1] + traj[:-2]
        speed = np.maximum(np.linalg.norm(vel, axis=1, keepdims=True), 1e-8)
        vhat = vel / speed

        def project(u):  # (I - vhat vhat^T) u, rowwise
            return u - vhat * np.sum(vhat * u, axis=1, keepdims=True)

        kappa = project(acc) / speed**2
        grad_c = cp[1:-1, None] * g[1:-1]
        grad_obs = speed * (project(grad_c) - c[1:-1, None] * kappa)

        # --- covariant update
        grad = grad_obs + lambda_smooth * (A @ xi + b)
        step = cho_solve(A_chol, grad) / eta
        xi = xi - step

        # Track the best collision-free iterate (by smoothness cost).
        d_now = env.clearance(np.vstack([start, xi, goal]))
        if np.all(d_now > 0.0):
            smooth_cost = 0.5 * np.sum(np.diff(np.vstack([start, xi, goal]), axis=0) ** 2)
            if best_xi is None or best_cost - smooth_cost > min_rel_improve * best_cost:
                stale = 0
            else:
                stale += 1
            if smooth_cost < best_cost:
                best_cost = smooth_cost
                best_xi = xi.copy()
        elif best_xi is not None:
            stale += 1

        if np.linalg.norm(step) < tol:
            break
        if best_xi is not None and stale >= patience:
            break

    if best_xi is not None:
        xi = best_xi
    path = np.vstack([start, xi, goal])
    info = {
        "iterations": it + 1,
        "time": time.perf_counter() - t0,
        "success": bool(env.path_free(path)),
    }
    return path, info
