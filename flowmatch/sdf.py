#Batched, differentiable signed distance field in torch, over the SAME obstacle
#tensors the encoder consumes.

#Why this exists: pointmass3d/env.py is numpy and axis-aligned, so it cannot be
#evaluated inside the ODE loop, and it cannot express the ORIENTED boxes the
#(s,g) reduction produces. This module is the torch counterpart, written to
#agree with env.sdf exactly in the world frame (test_sdf.py asserts it) and to
#be rigid-motion equivariant, which is what makes it usable on the reduced arm.

#It takes the obstacle tensors in WHATEVER frame they are already in --
#the same `spheres` and `boxes` that go to ObstacleEncoder -- so a caller cannot
#accidentally evaluate the field in one frame and the state in another. That is
#the same class of bug the equivariance diagnostic hit (Sec. "Implementation"):
#the group acts on scene and state together or the measurement is meaningless.

#Conventions, matching pointmass3d/env.py:
#  * obstacle SDF is POSITIVE OUTSIDE the obstacle
#  * the workspace is a CONTAINER: positive INSIDE it
#  * the scene SDF is the min over obstacles and the container
#  * clearance = sdf - robot_radius
#All lengths are in whatever units the tensors use. Under the isotropic
#Normalizer (single scalar scale) a normalised-frame SDF is the world SDF
#divided by that scale, so a caller working in normalised coordinates should
#normalise the robot radius too.

import torch

from .geometry import split_box_features


def sphere_sdf(points, spheres, mask=None):
    #points (B,N,3); spheres (B,S,4) = centre + radius -> (B,N,S)
    c, r = spheres[..., :3], spheres[..., 3]
    d = torch.linalg.norm(points[:, :, None, :] - c[:, None, :, :], dim=-1) - r[:, None, :]
    if mask is not None:
        d = d.masked_fill(~mask[:, None, :], float("inf"))
    return d


def obb_sdf(points, boxes, mask=None):
    #points (B,N,3); boxes (B,K,12) = centre + three half-edge VECTORS -> (B,N,K)

    #The box is a rigid transform of an axis-aligned box and the SDF is
    #invariant under rigid motion, so the axis-aligned formula applies verbatim
    #in the box's own frame: project the displacement onto the unit edge
    #directions and use the half-edge LENGTHS as extents. Rotating the box and
    #rotating the query point therefore give the same number, which is exactly
    #the property the reduced arm needs.
    centre, edges = split_box_features(boxes)          # (B,K,3), (B,K,3,3)
    h = torch.linalg.norm(edges, dim=-1)               # (B,K,3) half-extents
    #A degenerate edge would divide by zero; a zero-extent box is a
    #degenerate slab and clamping keeps the direction finite and harmless.
    u = edges / h.clamp_min(1e-12)[..., None]          # (B,K,3,3) unit axes

    rel = points[:, :, None, :] - centre[:, None, :, :]         # (B,N,K,3)
    #project onto each box axis: (B,N,K,3) . (B,1,K,3,3) -> (B,N,K,3)
    local = torch.einsum("bnkj,bkij->bnki", rel, u)
    q = local.abs() - h[:, None, :, :]                          # (B,N,K,3)

    outside = torch.linalg.norm(q.clamp_min(0.0), dim=-1)
    inside = q.amax(dim=-1).clamp_max(0.0)
    d = outside + inside
    if mask is not None:
        d = d.masked_fill(~mask[:, None, :], float("inf"))
    return d


def container_sdf(points, container):
    #Distance to the inside of an oriented box, POSITIVE INSIDE.
    #container (B,1,12) in the same frame as points. This is the workspace: the
    #robot must stay IN it, so the sign convention is opposite to an obstacle,
    #and env.sdf's min(p - lo, hi - p) is the axis-aligned special case.
    centre, edges = split_box_features(container)
    h = torch.linalg.norm(edges, dim=-1)
    u = edges / h.clamp_min(1e-12)[..., None]
    rel = points[:, :, None, :] - centre[:, None, :, :]
    local = torch.einsum("bnkj,bkij->bnki", rel, u)
    return (h[:, None, :, :] - local.abs()).amin(dim=-1).amin(dim=-1)  # (B,N)


def scene_sdf(points, spheres, boxes, sphere_mask=None, box_mask=None,
              container=None):
    #Minimum over spheres, boxes and (optionally) the workspace walls. (B,N)

    #The container is optional and OFF by default because the obstacle encoder
    #is never shown the walls either: including them here would give a probe
    #information the model's own conditioning does not have, which is a
    #different experiment from the one intended. Pass it when the quantity
    #wanted is true clearance rather than obstacle clearance.
    d = points.new_full(points.shape[:2], float("inf"))
    if spheres is not None and spheres.shape[1]:
        d = torch.minimum(d, sphere_sdf(points, spheres, sphere_mask).amin(-1))
    if boxes is not None and boxes.shape[1]:
        d = torch.minimum(d, obb_sdf(points, boxes, box_mask).amin(-1))
    if container is not None:
        d = torch.minimum(d, container_sdf(points, container))
    return d


def scene_sdf_and_grad(points, spheres, boxes, sphere_mask=None, box_mask=None,
                       container=None):
    #(B,N) distances and (B,N,3) spatial gradients.

    #Analytic via autograd rather than finite differences: env.clearance_grad
    #uses central differences with eps=1e-5, which costs six SDF evaluations
    #and loses precision, and here the field is already differentiable. The
    #gradient is the direction of steepest clearance increase -- the local
    #"which way is out" signal a global scene code cannot express.
    pts = points.detach().requires_grad_(True)
    with torch.enable_grad():
        d = scene_sdf(pts, spheres, boxes, sphere_mask, box_mask, container)
        g, = torch.autograd.grad(d.sum(), pts, create_graph=False)
    return d.detach(), g


def workspace_box(lo, hi, normalizer=None, device=None, dtype=torch.float32):
    #The workspace as a (1,1,12) container in normalised coordinates, ready to
    #be rotated by rotate_box_features alongside the obstacles.
    import numpy as np

    centre = np.full(3, 0.5 * (lo + hi), dtype=np.float32)
    half = np.full(3, 0.5 * (hi - lo), dtype=np.float32)
    if normalizer is not None:
        centre = normalizer.norm_pts(centre[None])[0]
        half = normalizer.norm_len(half)
    feat = np.zeros(12, dtype=np.float32)
    feat[:3] = centre
    feat[3::4] = half          # diagonal of the 3x3 edge matrix
    return torch.from_numpy(feat).to(device=device, dtype=dtype).reshape(1, 1, 12)
