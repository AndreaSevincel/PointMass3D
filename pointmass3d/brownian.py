import numpy as np


def brownian(start, goal, n_waypoints=64, sigma_prior=0.1, n_samples=None, rng=None):
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    rng = np.random.default_rng(rng)

    N = n_waypoints
    gamma = np.linspace(0.0, 1.0, N)[:, None]  #(N, 1)
    mu = (1.0 - gamma) * start + gamma * goal  #(N, 3)
    std = sigma_prior * np.sqrt(gamma * (1.0 - gamma))  #(N, 1)

    batch = 1 if n_samples is None else n_samples
    eps = rng.standard_normal((batch, N, 3))
    traj = mu[None] + std[None] * eps
    return traj[0] if n_samples is None else traj
