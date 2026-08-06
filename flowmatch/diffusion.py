#Diffuser / MPD-style DDPM baseline over whole trajectories.

#This is the generative-model control for the paper: same backbone, same
#conditioning, same data, same frame options -- only the generative objective
#differs. Flow matching regresses a velocity along a straight interpolant;
#this regresses the noise eps in a variance-preserving forward process, which
#is what Diffuser and Motion Planning Diffusion do.

#Running the (s,g) reduction under BOTH objectives is the point. If the
#reduction only paid off for flow matching it would be a quirk of the
#interpolant; the reduction is a statement about the problem, so it should pay
#off here too.

#The network is reused unchanged: FlowVelocityField maps (x, t, c) -> R^{N x 3}.
#For flow matching that output is a velocity, here it is eps-hat. t is passed
#as a continuous value in [0,1] (timestep/T) so the sinusoidal embedding is
#identical across objectives.

import math

import torch

from .flow import build_conditioning, reduce_batch
from .geometry import rotate_box_features, rotate_sphere_features, sg_frame


def cosine_schedule(T, s=0.008, device="cpu", dtype=torch.float32):
    #Nichol & Dhariwal cosine schedule -- the one Diffuser/MPD use. Returns
    #alpha_bar of length T+1 with alpha_bar[0] = 1 (clean data).
    t = torch.arange(T + 1, device=device, dtype=torch.float64) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = (f / f[0]).clamp(1e-8, 1.0)
    return ab.to(dtype)


class Schedule:
    #Precomputed VP-diffusion coefficients. Indexing convention: level i means
    #x_i = sqrt(ab[i]) x_0 + sqrt(1 - ab[i]) eps, so i = 0 is clean data and
    #i = T is (almost) pure noise.
    def __init__(self, T=100, device="cpu", dtype=torch.float32):
        self.T = T
        self.ab = cosine_schedule(T, device=device, dtype=dtype)  # (T+1,)
        self.beta = (1.0 - self.ab[1:] / self.ab[:-1]).clamp(max=0.999)
        self.alpha = 1.0 - self.beta

    def to(self, device):
        self.ab = self.ab.to(device)
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        return self

    def q_sample(self, x0, i, noise):
        #Forward process at integer level i (B,), broadcasting over (B,N,3).
        ab = self.ab[i][:, None, None]
        return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise


def diffusion_loss(model, batch, tables, schedule, reduced=False, roll=True,
                   check=False):
    #eps-prediction loss. Mirrors flow_matching_loss() line for line, including
    #the reduction, so the two objectives differ in nothing else.
    x1 = batch["traj"]
    start, goal, env_id = batch["start"], batch["goal"], batch["env_id"]
    spheres = tables["spheres"][env_id]
    boxes = tables["boxes"][env_id]
    sphere_mask = tables["sphere_mask"][env_id]
    box_mask = tables["box_mask"][env_id]

    if reduced:
        x1, sg, spheres, boxes, _, _ = reduce_batch(
            x1, start, goal, spheres, boxes, roll=roll, check=check
        )
    else:
        sg = build_conditioning(start, goal, reduced=False)

    B = x1.shape[0]
    i = torch.randint(1, schedule.T + 1, (B,), device=x1.device)
    noise = torch.randn_like(x1)
    xt = schedule.q_sample(x1, i, noise)

    #t in [0,1] for the shared sinusoidal embedding
    t = i.to(x1.dtype) / schedule.T
    pred = model(xt, t, spheres, boxes, sg, sphere_mask, box_mask)
    return ((pred - noise) ** 2).mean()


def _timesteps(T, n_steps):
    #Strided DDIM subsequence, descending, always ending at level 1.
    #n_steps here is the NFE, so it is directly comparable to the flow
    #sampler's Euler-step budget.
    if n_steps >= T:
        return list(range(T, 0, -1))
    idx = torch.linspace(T, 1, n_steps).round().long().tolist()
    return sorted(set(idx), reverse=True)


def _step(x, eps, i, i_prev, sch, eta, generator):
    #One DDIM/DDPM update from level i to level i_prev (i_prev may be 0).
    ab_t = sch.ab[i]
    ab_p = sch.ab[i_prev]
    x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
    x0 = x0.clamp(-4.0, 4.0)  # data is normalised; keeps few-step runs stable
    if i_prev == 0:
        return x0
    #eta=0 is deterministic DDIM, eta=1 recovers the DDPM ancestral variance
    sigma = eta * ((1 - ab_p) / (1 - ab_t)).sqrt() * (1 - ab_t / ab_p).sqrt()
    dir_xt = (1 - ab_p - sigma**2).clamp_min(0).sqrt() * eps
    out = ab_p.sqrt() * x0 + dir_xt
    if eta > 0:
        z = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        out = out + sigma * z
    return out


@torch.no_grad()
def sample_diffusion(
    model, spheres, boxes, sg, schedule,
    anchor_start=None, anchor_goal=None, sphere_mask=None, box_mask=None,
    n_waypoints=64, n_steps=100, eta=0.0, anchor_endpoints=False,
    device="cpu", generator=None,
):
    #World-frame sampler, the DDPM counterpart of flow.sample().
    net = model.module if hasattr(model, "module") else model
    net.eval()
    B = sg.shape[0]
    sch = schedule
    x = torch.randn(B, n_waypoints, 3, device=device, generator=generator)
    c = net.encode_cond(spheres, boxes, sg, sphere_mask, box_mask)

    if anchor_endpoints:
        known = torch.stack([anchor_start, anchor_goal], dim=1)      # (B,2,3)
        known_eps = torch.randn(known.shape, device=device, dtype=known.dtype,
                                generator=generator)

    steps = _timesteps(sch.T, n_steps)
    for k, i in enumerate(steps):
        i_prev = steps[k + 1] if k + 1 < len(steps) else 0
        if anchor_endpoints:
            #inpainting: endpoints are re-noised to the current level from
            #their known values, exactly as RePaint/Diffuser conditioning does
            ab_i = sch.ab[i]
            x[:, [0, -1], :] = ab_i.sqrt() * known + (1 - ab_i).sqrt() * known_eps
        t = torch.full((B,), i / sch.T, device=device, dtype=x.dtype)
        eps = net.decode(x, t, c)
        x = _step(x, eps, i, i_prev, sch, eta, generator)

    if anchor_endpoints:
        x[:, 0, :] = anchor_start
        x[:, -1, :] = anchor_goal
    return x


@torch.no_grad()
def sample_diffusion_reduced(
    model, spheres, boxes, start, goal, schedule,
    sphere_mask=None, box_mask=None, n_waypoints=64, n_steps=100, eta=0.0,
    anchor_endpoints=False, device="cpu", generator=None,
):
    #Reduced-frame sampler, the DDPM counterpart of flow.sample_reduced().
    #No frame averaging here: the K sweep is a flow-matching experiment, and
    #the diffusion arm exists to test the reduction, not the sixth DOF.
    net = model.module if hasattr(model, "module") else model
    net.eval()
    sch = schedule
    B = start.shape[0]

    R0, origin, d = sg_frame(start, goal, None)
    sph_r = rotate_sphere_features(spheres, R0, origin)
    box_r = rotate_box_features(boxes, R0, origin)
    c = net.encode_cond(sph_r, box_r, d[:, None], sphere_mask, box_mask)

    x = torch.randn(B, n_waypoints, 3, device=device, generator=generator)
    if anchor_endpoints:
        zeros = torch.zeros_like(d)
        known = torch.stack([
            torch.stack([-0.5 * d, zeros, zeros], dim=-1),
            torch.stack([0.5 * d, zeros, zeros], dim=-1),
        ], dim=1)
        known_eps = torch.randn(known.shape, device=device, dtype=known.dtype,
                                generator=generator)

    steps = _timesteps(sch.T, n_steps)
    for k, i in enumerate(steps):
        i_prev = steps[k + 1] if k + 1 < len(steps) else 0
        if anchor_endpoints:
            ab_i = sch.ab[i]
            x[:, [0, -1], :] = ab_i.sqrt() * known + (1 - ab_i).sqrt() * known_eps
        t = torch.full((B,), i / sch.T, device=device, dtype=x.dtype)
        eps = net.decode(x, t, c)
        x = _step(x, eps, i, i_prev, sch, eta, generator)

    if anchor_endpoints:
        x[:, 0, :] = known[:, 0]
        x[:, -1, :] = known[:, 1]

    return torch.einsum("bji,bkj->bki", R0, x) + origin[:, None, :]
