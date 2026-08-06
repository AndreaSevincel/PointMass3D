#Flow matching over SE(3) pose trajectories.

#Deliberately thin. The generative machinery is identical to the point-mass
#domain -- same straight-line interpolant, same velocity regression, same
#network -- and everything that differs is the geometry, which lives in
#reduction.py. That is the claim the second domain is meant to test: the
#reduction is a statement about the problem, so moving to a state with an
#orientation should require new geometry and no new learning machinery.

#One modelling choice is worth naming. The flow runs in the ambient R^9, not on
#the SO(3) manifold: the model interpolates and regresses the 6D rotation
#coordinates freely, and Gram-Schmidt projects back to a rotation only when a
#sample is decoded. A manifold-native flow would be the more principled object,
#but it would also change the generative model between the two domains and
#confound exactly the comparison being made here.

import numpy as np
import torch

from .reduction import (
    build_conditioning_se3,
    reduce_batch_se3,
    transform_states,
    unreduce_states,
)
from flowmatch.geometry import sg_frame


def flow_matching_loss_se3(model, batch, tables, reduced=False, roll=True,
                           check=False):
    x1 = batch["traj"]                                   # (B,N,9)
    start, goal, env_id = batch["start"], batch["goal"], batch["env_id"]
    spheres = tables["spheres"][env_id]
    boxes = tables["boxes"][env_id]
    sphere_mask = tables["sphere_mask"][env_id]
    box_mask = tables["box_mask"][env_id]

    if reduced:
        x1, sg, spheres, boxes, _, _ = reduce_batch_se3(
            x1, start, goal, spheres, boxes, roll=roll, check=check
        )
    else:
        sg = build_conditioning_se3(start, goal, reduced=False)

    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], device=x1.device)
    xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * x1
    target = x1 - x0
    pred = model(xt, t, spheres, boxes, sg, sphere_mask, box_mask)
    return ((pred - target) ** 2).mean()


@torch.no_grad()
def sample_se3(model, spheres, boxes, start, goal, sphere_mask=None,
               box_mask=None, n_waypoints=64, n_steps=8, reduced=True,
               device="cpu", generator=None):
    """Returns world-frame pose states (B, N, 9)."""
    net = model.module if hasattr(model, "module") else model
    net.eval()
    B = start.shape[0]

    if reduced:
        R, origin, d = sg_frame(start[..., :3], goal[..., :3], None)
        from flowmatch.geometry import rotate_box_features, rotate_sphere_features
        spheres = rotate_sphere_features(spheres, R, origin)
        boxes = rotate_box_features(boxes, R, origin)
        start_r = transform_states(start[:, None, :], R, origin)[:, 0, :]
        goal_r = transform_states(goal[:, None, :], R, origin)[:, 0, :]
        sg = build_conditioning_se3(start_r, goal_r, reduced=True)
    else:
        sg = build_conditioning_se3(start, goal, reduced=False)

    c = net.encode_cond(spheres, boxes, sg, sphere_mask, box_mask)
    x = torch.randn(B, n_waypoints, 9, device=device, generator=generator)
    dt = 1.0 / n_steps
    for k in range(n_steps):
        t = torch.full((B,), k * dt, device=device, dtype=x.dtype)
        x = x + dt * net.decode(x, t, c)

    if reduced:
        x = unreduce_states(x, R, origin)
    return x


def decode_poses(x, normalizer=None):
    """(B,N,9) states -> world positions (B,N,3) and rotations (B,N,3,3).

    Gram-Schmidt is applied here and only here: the model's six rotation
    outputs are unconstrained, and this is the projection back onto SO(3).
    """
    from .rotation import sixd_to_matrix
    x = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    pos, rot6 = x[..., :3], x[..., 3:]
    if normalizer is not None:
        pos = normalizer.denorm_pts(pos)
    return pos, sixd_to_matrix(rot6)
