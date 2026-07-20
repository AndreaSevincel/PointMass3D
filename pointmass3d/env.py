"""
PointMass3D environment, a spherical point-mass robot moving in a box
workspace populated with sphere and axis-aligned box obstacles

collision checking done through a signed distance field (SDF)
"""

import numpy as np


class SphereObstacle:
    def __init__(self, center, radius):
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)

    def sdf(self, points):
        points = np.asarray(points, dtype=float)
        return np.linalg.norm(points - self.center, axis=-1) - self.radius


class BoxObstacle:
    def __init__(self, center, half_extents):
        self.center = np.asarray(center, dtype=float)
        self.half_extents = np.asarray(half_extents, dtype=float)

    def sdf(self, points):
        points = np.asarray(points, dtype=float)
        q = np.abs(points - self.center) - self.half_extents
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        inside = np.minimum(q.max(axis=-1), 0.0)
        return outside + inside


class PointMass3DEnv:
    def __init__(self, obstacles=(), lo=-1.0, hi=1.0, robot_radius=0.03):
        self.obstacles = list(obstacles)
        self.lo = float(lo)
        self.hi = float(hi)
        self.robot_radius = float(robot_radius)

    def sdf(self, points):
        points = np.asarray(points, dtype=float)
        # Distance to the workspace walls, positive inside the workspace box.
        d = np.minimum(points - self.lo, self.hi - points).min(axis=-1)
        for obs in self.obstacles:
            d = np.minimum(d, obs.sdf(points))
        return d

    def clearance(self, points):
        return self.sdf(points) - self.robot_radius

    def clearance_grad(self, points, eps=1e-5):
        points = np.asarray(points, dtype=float)
        grad = np.empty_like(points)
        for k in range(3):
            dp = np.zeros(3)
            dp[k] = eps
            grad[..., k] = (self.sdf(points + dp) - self.sdf(points - dp)) / (2 * eps)
        return grad

    # collision tests
    def in_collision(self, points, margin=0.0):
        return self.clearance(points) <= margin

    def segment_free(self, a, b, margin=0.0, resolution=0.01):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / resolution)) + 1)
        pts = np.linspace(a, b, n)
        return not np.any(self.in_collision(pts, margin))

    def path_free(self, path, margin=0.0, resolution=0.01):
        path = np.asarray(path, dtype=float)
        return all(
            self.segment_free(path[i], path[i + 1], margin, resolution)
            for i in range(len(path) - 1)
        )

    #sampling
    def sample_free(self, rng=None, margin=0.02, max_tries=1000):
        rng = np.random.default_rng(rng)
        for _ in range(max_tries):
            q = rng.uniform(self.lo, self.hi, size=3)
            if self.clearance(q) > margin:
                return q
        raise RuntimeError("could not sample a collision-free configuration")


def make_random_env(
    n_spheres=20,
    n_boxes=20,
    seed=None,
    robot_radius=0.03,
    sphere_radius=(0.10, 0.25),
    box_half_extent=(0.08, 0.20),
    center_range=0.75,
):
    rng = np.random.default_rng(seed)
    obstacles = []
    for _ in range(n_spheres):
        c = rng.uniform(-center_range, center_range, size=3)
        r = rng.uniform(*sphere_radius)
        obstacles.append(SphereObstacle(c, r))
    for _ in range(n_boxes):
        c = rng.uniform(-center_range, center_range, size=3)
        h = rng.uniform(*box_half_extent, size=3)
        obstacles.append(BoxObstacle(c, h))
    return PointMass3DEnv(obstacles, robot_radius=robot_radius)


def sample_start_goal(env, rng=None, margin=0.05, min_dist=1.5, max_tries=1000):
    """Sample a well-separated collision-free (start, goal) pair."""
    rng = np.random.default_rng(rng)
    for _ in range(max_tries):
        start = env.sample_free(rng, margin)
        goal = env.sample_free(rng, margin)
        if np.linalg.norm(goal - start) >= min_dist:
            return start, goal
    raise RuntimeError("could not sample a start/goal pair")
