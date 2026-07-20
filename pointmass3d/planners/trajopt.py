import time

import numpy as np


def _exact_cost(env, xi, start, goal, mu, d_safe):
    traj = np.vstack([start, xi, goal])
    smooth = 0.5 * np.sum(np.diff(traj, axis=0) ** 2)
    viol = np.maximum(d_safe - env.clearance(traj[1:-1]), 0.0)
    return smooth + mu * np.sum(viol**2)


def trajopt(
    env,
    start,
    goal,
    n_waypoints=64,
    init_path=None,
    d_safe=0.02,
    mu0=100.0,
    mu_scale=10.0,
    penalty_iters=4,
    sqp_iters=50,
    ctol=1e-3,
    step_tol=1e-5,
):
    """Optimize a trajectory between start and goal.

    Returns (path, info); info["success"] reports a dense collision check.
    """
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    N = n_waypoints
    M = N - 2

    if init_path is None:
        traj = np.linspace(start, goal, N)
    else:
        traj = np.asarray(init_path, dtype=float).copy()
        assert traj.shape == (N, 3)
    xi = traj[1:-1].copy()

    # Quadratic smoothness on the flattened interior trajectory (M*3,).
    A = 2.0 * np.eye(M) - np.eye(M, k=1) - np.eye(M, k=-1)
    A3 = np.kron(A, np.eye(3))
    b = np.zeros((M, 3))
    b[0] -= start
    b[-1] -= goal
    b_flat = b.reshape(-1)
    I3M = np.eye(3 * M)

    t0 = time.perf_counter()
    mu = mu0
    total_sqp = 0

    for _ in range(penalty_iters):
        rho = 1.0  # proximal trust-region weight

        for _ in range(sqp_iters):
            total_sqp += 1
            x0 = xi.copy()
            d0 = env.clearance(x0)
            g = env.clearance_grad(x0)
            # Affine model of the violation: v_i(x) = r_i - g_i . x_i
            r = d_safe - d0 + np.sum(g * x0, axis=1)

            def model_cost(x, with_prox=False):
                xf = x.reshape(-1)
                m = 0.5 * xf @ A3 @ xf + b_flat @ xf
                v = np.maximum(r - np.sum(g * x, axis=1), 0.0)
                m += mu * np.sum(v**2)
                if with_prox:
                    m += rho * np.sum((x - x0) ** 2)
                return m

            def solve_subproblem(rho_):
                # Active-set iteration: with a fixed set of violated
                # waypoints the problem is quadratic -> one linear solve.
                active = (r - np.sum(g * x0, axis=1)) > 0.0
                x = x0
                for _ in range(10):
                    H = A3 + 2.0 * rho_ * I3M
                    rhs = -b_flat + 2.0 * rho_ * x0.reshape(-1)
                    for i in np.where(active)[0]:
                        sl = slice(3 * i, 3 * i + 3)
                        H[sl, sl] += 2.0 * mu * np.outer(g[i], g[i])
                        rhs[sl] += 2.0 * mu * r[i] * g[i]
                    x = np.linalg.solve(H, rhs).reshape(M, 3)
                    new_active = (r - np.sum(g * x, axis=1)) > 0.0
                    if np.array_equal(new_active, active):
                        break
                    active = new_active
                return x

            f0 = _exact_cost(env, x0, start, goal, mu, d_safe)
            m0 = model_cost(x0)  # equals f0 up to a constant that cancels in pred
            accepted = False
            for _ in range(12):  # trust-region adjustment retries
                x1 = solve_subproblem(rho)
                pred = m0 - model_cost(x1)
                f1 = _exact_cost(env, x1, start, goal, mu, d_safe)
                actual = f0 - f1
                if pred <= 1e-12:
                    break  # model can't improve: converged for this mu
                ratio = actual / pred
                if ratio > 0.25:
                    accepted = True
                    if ratio > 0.75:
                        rho *= 0.5
                    break
                rho *= 4.0  # poor model agreement: shrink the step
            if not accepted:
                break
            step_norm = np.linalg.norm(x1 - xi)
            xi = x1
            if step_norm < step_tol:
                break

        violation = np.max(d_safe - env.clearance(xi))
        if violation <= ctol:
            break
        mu *= mu_scale

    path = np.vstack([start, xi, goal])
    info = {
        "sqp_iterations": total_sqp,
        "penalty_mu": mu,
        "time": time.perf_counter() - t0,
        "success": bool(env.path_free(path)),
    }
    return path, info
