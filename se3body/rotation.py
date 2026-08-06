#SO(3) utilities and the 6D rotation representation.

#The learned model needs rotations in a form that is (a) continuous, so a
#regression target is well-posed, and (b) transforms simply under the (s,g)
#reduction. Quaternions fail (a) -- they double-cover, so q and -q are the same
#rotation and any regression target is ambiguous. Euler angles fail (a) too.

#The 6D representation (Zhou et al., CVPR 2019) is the first two COLUMNS of the
#rotation matrix, with the third recovered by cross product. It is continuous,
#and it has the property this project needs: the columns of a rotation matrix
#are vectors expressed in the world frame, so under a global rotation Q the
#matrix maps A -> QA and each stored column maps a -> Qa. They are FREE VECTORS.
#Positions are points and take the affine map; rotation columns take the
#rotation only. That distinction is the same one that silently broke the box
#half-extents in the point-mass domain, and it is tested in test_se3.py.

import numpy as np


def rand_rotation(rng=None, size=None):
    #Uniform on SO(3) via QR of a Gaussian matrix, sign-fixed for det=+1.
    rng = np.random.default_rng(rng)
    n = 1 if size is None else int(size)
    out = np.empty((n, 3, 3))
    for i in range(n):
        q, r = np.linalg.qr(rng.standard_normal((3, 3)))
        q = q * np.sign(np.diag(r))          # make the decomposition unique
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1.0                  # reflection -> rotation
        out[i] = q
    return out[0] if size is None else out


def matrix_to_6d(R):
    #(...,3,3) -> (...,6): the first two columns, stacked.
    R = np.asarray(R, dtype=float)
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def sixd_to_matrix(d6):
    #(...,6) -> (...,3,3) by Gram-Schmidt. Defined for ANY input, which is what
    #makes it usable as a network output: the model can emit six unconstrained
    #numbers and still produce a valid rotation.
    d6 = np.asarray(d6, dtype=float)
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True).clip(1e-12)
    a2_proj = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = a2_proj / np.linalg.norm(a2_proj, axis=-1, keepdims=True).clip(1e-12)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)   # columns


def geodesic_angle(A, B):
    #Rotation angle of A^T B, in [0, pi].
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    M = np.swapaxes(A, -1, -2) @ B
    tr = np.trace(M, axis1=-2, axis2=-1)
    return np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))


def _log_so3(R):
    #Rotation vector (axis * angle) of R.
    ang = geodesic_angle(np.eye(3), R)
    if ang < 1e-8:
        return np.zeros(3)
    if ang > np.pi - 1e-6:
        #near pi the skew part vanishes; recover the axis from R + I, whose
        #columns are all parallel to the axis
        M = R + np.eye(3)
        k = int(np.argmax(np.linalg.norm(M, axis=0)))
        axis = M[:, k] / np.linalg.norm(M[:, k])
        return axis * ang
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2.0 * np.sin(ang)) * ang


def _exp_so3(w):
    ang = float(np.linalg.norm(w))
    if ang < 1e-12:
        return np.eye(3)
    k = w / ang
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def slerp(A, B, u):
    #Geodesic interpolation on SO(3): A at u=0, B at u=1.
    return A @ _exp_so3(_log_so3(A.T @ B) * float(u))


def interpolate(pa, Ra, pb, Rb, n):
    #n poses from (pa,Ra) to (pb,Rb) inclusive: straight in position, geodesic
    #in rotation.
    us = np.linspace(0.0, 1.0, n)
    pos = (1 - us)[:, None] * pa + us[:, None] * pb
    rot = np.stack([slerp(Ra, Rb, u) for u in us])
    return pos, rot
