#The rigid body, and collision checking for it.

#The robot is a union of spheres rigidly attached to a body frame. That choice
#is deliberate: collision against ANY obstacle whose SDF is known reduces to
#evaluating that SDF at the transformed sphere centres and subtracting the
#sphere radius, so the exact analytic SDF the point-mass domain already has is
#reused unchanged, for a body that now has an orientation. It is also the
#representation CHOMP uses for real manipulators.

#The shape is an L, not a rod. A rod is rotationally symmetric about its axis,
#so one of its three orientation degrees of freedom would not affect the swept
#volume and the domain would quietly be SE(3)/SO(2) rather than SE(3). The L
#has trivial symmetry group, so all three rotational DOF are observable.

import numpy as np

from .rotation import interpolate

#body-frame sphere centres: a 3-sphere arm along +x and a 2-sphere arm along +y
L_SHAPE = np.array([
    [0.00, 0.00, 0.0],
    [0.06, 0.00, 0.0],
    [0.12, 0.00, 0.0],
    [0.00, 0.06, 0.0],
    [0.00, 0.12, 0.0],
])
SPHERE_RADIUS = 0.03


class RigidBody:
    def __init__(self, centers=L_SHAPE, radius=SPHERE_RADIUS):
        self.centers = np.asarray(centers, dtype=float)
        self.radius = float(radius)
        #centre the body on its centroid so that the pose position is the
        #centroid rather than an arbitrary corner; this makes the start/goal
        #distance and the (s,g) frame mean what they appear to mean
        self.centers = self.centers - self.centers.mean(axis=0)

    @property
    def n_spheres(self):
        return len(self.centers)

    @property
    def extent(self):
        #farthest sphere centre from the origin, plus the sphere radius
        return float(np.linalg.norm(self.centers, axis=-1).max() + self.radius)

    def world_centers(self, p, R):
        #(...,3), (...,3,3) -> (...,M,3). Sphere centres are POINTS attached to
        #the body: they take the rotation and the translation.
        p = np.asarray(p, dtype=float)
        R = np.asarray(R, dtype=float)
        return np.einsum("...ij,mj->...mi", R, self.centers) + p[..., None, :]


class SE3Env:
    """A rigid body in the point-mass workspace. Wraps an obstacle SDF."""

    def __init__(self, base_env, body=None):
        #base_env must expose sdf(points); its own robot_radius is bypassed so
        #the clearance here is the body's, not a point robot's
        self.base = base_env
        self.body = body if body is not None else RigidBody()
        self.lo, self.hi = base_env.lo, base_env.hi

    def clearance(self, p, R):
        #min over body spheres of (obstacle sdf at centre - sphere radius)
        c = self.body.world_centers(p, R)                # (...,M,3)
        return self.base.sdf(c).min(axis=-1) - self.body.radius

    def pose_free(self, p, R, margin=0.0):
        return bool(np.all(self.clearance(p, R) > margin))

    def segment_free(self, pa, Ra, pb, Rb, margin=0.0, resolution=0.02):
        #Resolution is in swept distance. A rotation of angle a sweeps the
        #farthest point by about a * extent, so the two motions are put on a
        #common scale before choosing the number of checks -- checking a pure
        #rotation at position resolution would check it once.
        from .rotation import geodesic_angle
        d_lin = float(np.linalg.norm(np.asarray(pb) - np.asarray(pa)))
        d_ang = float(geodesic_angle(Ra, Rb)) * self.body.extent
        n = max(2, int(np.ceil((d_lin + d_ang) / resolution)) + 1)
        pos, rot = interpolate(pa, Ra, pb, Rb, n)
        return bool(np.all(self.clearance(pos, rot) > margin))

    def path_free(self, positions, rotations, margin=0.0, resolution=0.02):
        positions = np.asarray(positions, dtype=float)
        rotations = np.asarray(rotations, dtype=float)
        return all(
            self.segment_free(positions[i], rotations[i],
                              positions[i + 1], rotations[i + 1],
                              margin, resolution)
            for i in range(len(positions) - 1)
        )

    def min_clearance(self, positions, rotations, resolution=0.02):
        #Densely sampled minimum clearance along the path, the continuous
        #analogue of path_free and the metric that shows near-misses.
        from .rotation import geodesic_angle
        best = np.inf
        for i in range(len(positions) - 1):
            d_lin = float(np.linalg.norm(positions[i + 1] - positions[i]))
            d_ang = float(geodesic_angle(rotations[i], rotations[i + 1])) * self.body.extent
            n = max(2, int(np.ceil((d_lin + d_ang) / resolution)) + 1)
            pos, rot = interpolate(positions[i], rotations[i],
                                   positions[i + 1], rotations[i + 1], n)
            best = min(best, float(self.clearance(pos, rot).min()))
        return best

    def sample_free_pose(self, rng=None, margin=0.02, max_tries=2000):
        from .rotation import rand_rotation
        rng = np.random.default_rng(rng)
        #keep the whole body inside the workspace, not just its centroid
        m = self.body.extent
        for _ in range(max_tries):
            p = rng.uniform(self.lo + m, self.hi - m, size=3)
            R = rand_rotation(rng)
            if self.clearance(p, R) > margin:
                return p, R
        raise RuntimeError("could not sample a collision-free pose")
