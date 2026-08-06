#The (s,g) reduction for SE(3) pose trajectories.

#This is the point of the second domain. In the point-mass case a trajectory is
#a list of POINTS and every entry takes the same affine map, so the typed
#transform distinction only showed up in the obstacle features. Here the state
#itself is mixed: each waypoint carries a position, which is a point, and a
#rotation, whose stored columns are free vectors. Applying the affine map to a
#rotation column adds -R*origin to it and silently destroys the orientation
#while leaving the positions correct -- the same failure mode that hit the box
#half-extents, now inside the trajectory.

#The reduction itself is unchanged: the frame comes from the start and goal
#POSITIONS, so it still removes three translations and two rotations exactly,
#and still cannot remove the roll. What changes is the conditioning. After the
#reduction the start and goal positions are (-+d/2, 0, 0), but their
#orientations survive and must still be given to the model, so

#    world frame:  3 + 6 + 3 + 6 = 18 numbers
#    reduced:      1 + 6 + 6     = 13 numbers

#i.e. the reduction removes the five pose DOF it can and leaves the rest
#intact, rather than collapsing the query to a single scalar as it does when
#the robot is a point.

import numpy as np
import torch

from flowmatch.geometry import apply_points, apply_vectors, sg_frame

STATE_DIM = 9   # 3 position + 6 rotation


def pose_to_state(pos, rot6d):
    #(...,3) + (...,6) -> (...,9)
    return np.concatenate([pos, rot6d], axis=-1)


def state_to_pose(x):
    #(...,9) -> (...,3), (...,6)
    return x[..., :3], x[..., 3:]


def transform_states(x, R, origin):
    """Apply a frame (R, origin) to a batch of pose trajectories.

    x: (B, N, 9). Positions take the affine map; the two rotation columns take
    the rotation only. This function is the whole typed-transform contract for
    this domain and is what test_se3.py checks against a brute-force reference.
    """
    pos, rot6 = x[..., :3], x[..., 3:]
    pos_r = apply_points(R, origin, pos)                       # POINTS
    B, N = rot6.shape[0], rot6.shape[1]
    #the 6D representation is two stacked columns; rotate each as a free vector
    cols = rot6.reshape(B, N * 2, 3)
    cols_r = apply_vectors(R, cols)                            # FREE VECTORS
    return torch.cat([pos_r, cols_r.reshape(B, N, 6)], dim=-1)


def build_conditioning_se3(start, goal, reduced):
    """start/goal: (B,9) pose states. Returns the conditioning vector.

    reduced=True assumes start/goal have ALREADY been mapped into the reduced
    frame, so their positions are (-+d/2,0,0) and only d plus the two
    orientations carry information.
    """
    if not reduced:
        return torch.cat([start, goal], dim=-1)                # (B,18)
    d = (goal[..., 0] - start[..., 0])[..., None]              # = ||g-s||
    return torch.cat([d, start[..., 3:], goal[..., 3:]], dim=-1)   # (B,13)


def reduce_batch_se3(traj, start, goal, spheres, boxes, roll=True, check=False):
    """SE(3) analogue of flowmatch.flow.reduce_batch.

    traj: (B,N,9); start/goal: (B,9); obstacle features as in the point-mass
    domain. Returns the reduced trajectory, conditioning, and rotated scene.
    """
    from flowmatch.geometry import rotate_box_features, rotate_sphere_features

    s_pos, g_pos = start[..., :3], goal[..., :3]
    theta = None
    if roll:
        theta = torch.rand(start.shape[0], device=start.device) * 2 * np.pi
    R, origin, d = sg_frame(s_pos, g_pos, theta)

    traj_r = transform_states(traj, R, origin)
    start_r = transform_states(start[:, None, :], R, origin)[:, 0, :]
    goal_r = transform_states(goal[:, None, :], R, origin)[:, 0, :]
    sg = build_conditioning_se3(start_r, goal_r, reduced=True)

    spheres_r = rotate_sphere_features(spheres, R, origin)
    boxes_r = rotate_box_features(boxes, R, origin)

    if check:
        zeros = torch.zeros_like(d)
        want_s = torch.stack([-0.5 * d, zeros, zeros], dim=-1)
        want_g = torch.stack([0.5 * d, zeros, zeros], dim=-1)
        assert torch.allclose(start_r[..., :3], want_s, atol=1e-4), \
            "reduced start position is not (-d/2,0,0)"
        assert torch.allclose(goal_r[..., :3], want_g, atol=1e-4), \
            "reduced goal position is not (+d/2,0,0)"
        #a rotation column must keep unit norm; if the affine map were applied
        #to it by mistake this is the assertion that fires
        n = start_r[..., 3:6].norm(dim=-1)
        assert torch.allclose(n, torch.ones_like(n), atol=1e-3), \
            "rotation column lost unit norm -- affine map applied to a free vector?"

    return traj_r, sg, spheres_r, boxes_r, R, origin


def unreduce_states(x, R, origin):
    """Map reduced-frame pose states back to the world frame."""
    pos, rot6 = x[..., :3], x[..., 3:]
    pos_w = torch.einsum("bji,bkj->bki", R, pos) + origin[:, None, :]
    B, N = rot6.shape[0], rot6.shape[1]
    cols = rot6.reshape(B, N * 2, 3)
    cols_w = torch.einsum("bji,bkj->bki", R, cols)
    return torch.cat([pos_w, cols_w.reshape(B, N, 6)], dim=-1)
