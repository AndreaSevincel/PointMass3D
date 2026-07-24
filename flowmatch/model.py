
#Velocity field v_theta(x_t, t, c) for pure flow matching over trajectories.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(t, dim):
    """t: (B,) in [0,1] -> (B, dim) sinusoidal features."""
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
    """PointNet-lite over spheres (S,4) and boxes (B,6) -> env embedding."""

    def __init__(self, hidden=128, out_dim=128):
        super().__init__()
        self.sphere_mlp = nn.Sequential(
            nn.Linear(4, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.box_mlp = nn.Sequential(
            nn.Linear(6, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
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

    def forward(self, spheres, boxes, sphere_mask=None, box_mask=None):
        # spheres (B,S,4), boxes (B,B_,6); masks (B,S)/(B,B_) or None
        s = self._masked_max(self.sphere_mlp(spheres), sphere_mask)  # (B, hidden)
        b = self._masked_max(self.box_mlp(boxes), box_mask)          # (B, hidden)
        return self.out(torch.cat([s, b], dim=-1))


class ConditionEncoder(nn.Module):
    """(env_emb, start, goal) -> conditioning vector."""

    def __init__(self, env_dim=128, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(env_dim + 6, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, env_emb, start, goal):
        return self.net(torch.cat([env_emb, start, goal], dim=-1))


class FiLMResBlock(nn.Module):
    """Dilated 1D conv residual block with FiLM (scale/shift) conditioning."""

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
    #Predicts the flow-matching velocity v(x_t, t, c), x_t of shape (B, N, 3).

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
    ):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.obstacle_enc = ObstacleEncoder(env_hidden, env_dim)
        self.cond_enc = ConditionEncoder(env_dim, cond_dim)

        global_dim = time_dim + cond_dim
        self.init_conv = nn.Conv1d(3, channels, 5, padding=2)
        self.blocks = nn.ModuleList(
            [
                FiLMResBlock(
                    channels, global_dim, dilations[i % len(dilations)], groups
                )
                for i in range(n_blocks)
            ]
        )
        self.out_norm = nn.GroupNorm(groups, channels)
        self.out_conv = nn.Conv1d(channels, 3, 1)
        # Start near the identity velocity field.
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, t, spheres, boxes, start, goal, sphere_mask=None, box_mask=None):
        # x: (B, N, 3) -> (B, 3, N)
        h = x.transpose(1, 2)
        h = self.init_conv(h)

        t_emb = self.time_mlp(sinusoidal_embedding(t, self.time_dim))
        env_emb = self.obstacle_enc(spheres, boxes, sphere_mask, box_mask)
        c = self.cond_enc(env_emb, start, goal)
        cond = torch.cat([t_emb, c], dim=-1)

        for blk in self.blocks:
            h = blk(h, cond)
        h = self.out_conv(F.silu(self.out_norm(h)))
        return h.transpose(1, 2)  # (B, N, 3)
