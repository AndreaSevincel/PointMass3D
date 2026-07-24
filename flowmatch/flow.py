"""
Pure flow-matching objective and ODE sampler.

Conditional OT / rectified-flow formulation with a *standard Gaussian* prior:

    x0 ~ N(0, I)              (prior — NOT a Brownian bridge)
    x1  = expert trajectory   (data)
    t  ~ U(0, 1)
    xt = (1 - t) x0 + t x1     (straight-line interpolation)
    target velocity u = x1 - x0

The network regresses u; sampling integrates dx/dt = v_theta(x, t, c) from the
Gaussian prior at t=0 to the data manifold at t=1 with explicit Euler steps.
"""

import copy

import torch


def flow_matching_loss(model, batch, tables):
    """
    batch: dict of tensors already on the target device
      traj (B,N,3), start (B,3), goal (B,3), env_id (B,)
    tables: dict of per-env obstacle tensors on the same device, keyed
      spheres (E,S,4), boxes (E,B,6), sphere_mask (E,S), box_mask (E,B).
    """
    x1 = batch["traj"]
    start, goal, env_id = batch["start"], batch["goal"], batch["env_id"]
    spheres = tables["spheres"][env_id]
    boxes = tables["boxes"][env_id]
    sphere_mask = tables["sphere_mask"][env_id]
    box_mask = tables["box_mask"][env_id]

    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], device=x1.device)
    xt = (1.0 - t)[:, None, None] * x0 + t[:, None, None] * x1
    target = x1 - x0

    pred = model(xt, t, spheres, boxes, start, goal, sphere_mask, box_mask)
    return ((pred - target) ** 2).mean()


@torch.no_grad()
def sample(
    model,
    spheres,
    boxes,
    start,
    goal,
    sphere_mask=None,
    box_mask=None,
    n_waypoints=64,
    n_steps=100,
    anchor_endpoints=False,
    device="cpu",
    generator=None,
):
    
    #Draw trajectories by integrating the learned velocity field
    
    model.eval()
    B = start.shape[0]
    x = torch.randn(B, n_waypoints, 3, device=device, generator=generator)

    known_eps = None
    if anchor_endpoints:
        known_eps = x[:, [0, -1], :].clone()  # frozen noise for the endpoints
        known = torch.stack([start, goal], dim=1)  # (B, 2, 3)

    ts = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    for i in range(n_steps):
        t = ts[i].expand(B)
        if anchor_endpoints:
            tt = ts[i]
            x[:, [0, -1], :] = (1.0 - tt) * known_eps + tt * known
        v = model(x, t, spheres, boxes, start, goal, sphere_mask, box_mask)
        x = x + (ts[i + 1] - ts[i]) * v

    if anchor_endpoints:
        x[:, 0, :] = start
        x[:, -1, :] = goal
    return x


class EMA:
    #Exponential moving average of model parameters
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p, alpha=1 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)

    def state_dict(self):
        return self.shadow.state_dict()
