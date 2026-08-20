
#Velocity field v_theta(x_t, t, c) for pure flow matching over trajectories.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(t, dim):
    #t: (B,) in [0,1] -> (B, dim) sinusoidal features
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device) / max(half - 1, 1)
    )
    args = t[:, None] * freqs[None] * 1000.0  # spread [0,1] over the freq band
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ObstacleEncoder(nn.Module):
    #PointNet-lite over spheres (S,4) and boxes (B,box_dim) -> env embedding.
    #box_dim=12 is center + three half-edge vectors (an OBB); 6 is the legacy
    #axis-aligned center + half-extents form.

    def __init__(self, hidden=128, out_dim=128, box_dim=12, sg_dim=0):
        super().__init__()
        self.box_dim = box_dim
        #REVIEWER CONTROL. With sg_dim>0 the raw query is concatenated to every
        #obstacle BEFORE the pool, so the scene code can vary with the query
        #without any change of frame.
        #
        #Why it exists: the world-frame arm's scene code has within-environment
        #standard deviation 0.000000 (Sec. "Where the benefit comes from") --
        #the encoder sees only obstacles, so its output is constant across
        #queries in an environment BY CONSTRUCTION. That makes the headline gap
        #open to the reading "canonicalisation is just letting the encoder see
        #the query". This arm separates the two: it is the world frame, with a
        #query-dependent scene code, and no frame change anywhere.
        self.sg_dim = sg_dim
        self.sphere_mlp = nn.Sequential(
            nn.Linear(4 + sg_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.box_mlp = nn.Sequential(
            nn.Linear(box_dim + sg_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.out = nn.Sequential(
            nn.Linear(2 * hidden, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )

    @staticmethod
    def _masked_max(feat, mask):
        # feat (B, K, H); mask (B, K) bool or None.
        if mask is not None:
            feat = feat.masked_fill(~mask[..., None], float("-inf"))
        m = feat.amax(dim=1)
        return torch.nan_to_num(m, neginf=0.0)  # guard fully-empty sets

    def forward(self, spheres, boxes, sphere_mask=None, box_mask=None, sg=None):
        # spheres (B,S,4), boxes (B,B_,6); masks (B,S)/(B,B_) or None
        if self.sg_dim:
            if sg is None:
                raise ValueError("this encoder was built with sg_dim>0 and needs "
                                 "the query at forward()")
            spheres = torch.cat(
                [spheres, sg[:, None, :].expand(-1, spheres.shape[1], -1)], dim=-1)
            boxes = torch.cat(
                [boxes, sg[:, None, :].expand(-1, boxes.shape[1], -1)], dim=-1)
        s = self._masked_max(self.sphere_mlp(spheres), sphere_mask)  # (B, hidden)
        b = self._masked_max(self.box_mlp(boxes), box_mask)          # (B, hidden)
        return self.out(torch.cat([s, b], dim=-1))


class ConditionEncoder(nn.Module):
    """(env_emb, sg) -> conditioning vector.

    sg is the raw (start, goal) pair concatenated (sg_dim=6) for world-frame
    arms, or just ||g-s|| (sg_dim=1) after the (s,g) reduction -- which is all
    that survives it, since start and goal then land on (-+d/2, 0, 0) and the
    other five numbers are structurally constant.
    """

    def __init__(self, env_dim=128, out_dim=256, sg_dim=6):
        super().__init__()
        self.sg_dim = sg_dim
        self.net = nn.Sequential(
            nn.Linear(env_dim + sg_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, env_emb, sg):
        return self.net(torch.cat([env_emb, sg], dim=-1))


class FiLMResBlock(nn.Module):

    def __init__(self, channels, cond_dim, dilation, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv1d(
            channels, channels, 3, padding=dilation, dilation=dilation
        )
        self.film = nn.Linear(cond_dim, 2 * channels)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv1d(
            channels, channels, 3, padding=dilation, dilation=dilation
        )

    def forward(self, x, cond):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(cond)[..., None].chunk(2, dim=1)
        h = h * (1 + scale) + shift
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class FlowVelocityField(nn.Module):

    def __init__(
        self,
        channels=128,
        n_blocks=8,
        dilations=(1, 2, 4, 8),
        time_dim=256,
        env_hidden=128,
        env_dim=128,
        cond_dim=256,
        groups=8,
        box_dim=12,
        sg_dim=6,
        #3 for a point-mass trajectory; 9 for an SE(3) pose trajectory
        #(3 position + a 6D rotation representation). See se3body/.
        state_dim=3,
        #ORACLE DIAGNOSTIC, off by default. Appends the true SDF value and
        #gradient at each waypoint to the trunk input, bypassing the global
        #scene code for the local query.
        #
        #Why it exists: the obstacle encoder max-pools 40 obstacles into one
        #128-d vector and FiLM applies it identically to all N waypoints, so a
        #waypoint has no channel through which to ask "what is near me" --
        #while collision avoidance is exactly that question. This measures the
        #headroom a better encoder could reach WITHOUT designing one.
        #
        #It is not a method. It hands the model exact geometry that no
        #perception-based system would have, so any number it produces is an
        #upper bound, not a result to report as performance.
        local_geom=False,
        #see ObstacleEncoder: concatenate the query to every obstacle
        #before pooling, so the scene code is query-dependent without a
        #change of frame
        query_cond=False,
    ):
        super().__init__()
        self.state_dim = state_dim
        #exposed on the model itself: sweep_steps.py identifies the arm from
        #sg_dim, and reaching through to cond_enc breaks for any backbone that
        #names its conditioning encoder differently
        self.sg_dim = sg_dim
        self.local_geom = local_geom
        self.query_cond = query_cond
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.obstacle_enc = ObstacleEncoder(env_hidden, env_dim, box_dim,
                                            sg_dim if query_cond else 0)
        self.cond_enc = ConditionEncoder(env_dim, cond_dim, sg_dim)

        global_dim = time_dim + cond_dim
        #+4 = scalar SDF and its 3-d gradient at each waypoint
        self.init_conv = nn.Conv1d(state_dim + 4 * local_geom, channels, 5, padding=2)
        self.blocks = nn.ModuleList(
            [
                FiLMResBlock(
                    channels, global_dim, dilations[i % len(dilations)], groups
                )
                for i in range(n_blocks)
            ]
        )
        self.out_norm = nn.GroupNorm(groups, channels)
        self.out_conv = nn.Conv1d(channels, state_dim, 1)
        # Start near the identity velocity field.
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def encode_cond(self, spheres, boxes, sg, sphere_mask=None, box_mask=None):
        #Time-independent conditioning: compute once per query, reuse across
        #every ODE step (and every frame-averaging rotation of the state).
        env_emb = self.obstacle_enc(spheres, boxes, sphere_mask, box_mask,
                                    sg if self.query_cond else None)
        return self.cond_enc(env_emb, sg)  # (B, cond_dim)

    def decode(self, x, t, c, spheres=None, boxes=None, sphere_mask=None,
               box_mask=None):
        # x: (B, N, state_dim), t: (B,), c: (B, cond_dim) from encode_cond
        # spheres/boxes are needed ONLY with local_geom, and must be in the
        # SAME frame as x -- pass the tensors that produced c, never the
        # world-frame originals. Mixing frames here is the failure mode the
        # equivariance diagnostic hit: it does not raise, it just measures
        # something else.
        if self.local_geom:
            if spheres is None or boxes is None:
                raise ValueError(
                    "local_geom=True needs the obstacle tensors at decode(); "
                    "pass the same frame-transformed tensors used for encode_cond"
                )
            from .sdf import scene_sdf_and_grad

            pos = x[..., :3]  # SE(3) states carry rotation columns too
            d, g = scene_sdf_and_grad(pos, spheres, boxes, sphere_mask, box_mask)
            x = torch.cat([x, d[..., None], g], dim=-1)
        h = self.init_conv(x.transpose(1, 2))
        t_emb = self.time_mlp(sinusoidal_embedding(t, self.time_dim))
        cond = torch.cat([t_emb, c], dim=-1)
        for blk in self.blocks:
            h = blk(h, cond)
        h = self.out_conv(F.silu(self.out_norm(h)))
        return h.transpose(1, 2)  # (B, N, state_dim)

    def forward(self, x, t, spheres, boxes, sg, sphere_mask=None, box_mask=None):
        return self.decode(
            x, t, self.encode_cond(spheres, boxes, sg, sphere_mask, box_mask),
            spheres, boxes, sphere_mask, box_mask,
        )


def build_model(cfg):
    #Instantiate from a checkpoint's model_config, filling in keys that
    #predate the OBB box features and the reduced conditioning vector.
    cfg = dict(cfg)
    cfg.setdefault("box_dim", 6)   # legacy: center + half-extents
    cfg.setdefault("sg_dim", 6)    # legacy: raw (start, goal)
    cfg.setdefault("state_dim", 3)  # legacy: point-mass trajectories
    cfg.setdefault("local_geom", False)  # legacy: no oracle channels
    cfg.setdefault("query_cond", False)  # legacy: query-blind obstacle encoder
    #Dispatch on the architecture recorded in the checkpoint. Without this a
    #constrained model would be rebuilt as an unconstrained one; load_state_dict
    #would raise, but only after the caller had already decided which arm it was
    #scoring, so the failure would be confusing rather than informative.
    if cfg.pop("equivariant", False):
        from .equivariant import EquivVelocityField
        if cfg.pop("query_cond", False):
            raise ValueError("query_cond is a world-frame control; the "
                             "equivariant backbone is a reduced-arm model")
        return EquivVelocityField(**cfg)
    return FlowVelocityField(**cfg)
