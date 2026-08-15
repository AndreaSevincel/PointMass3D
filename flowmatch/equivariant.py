
  #SO(2)-equivariant velocity field: the comparison the rest of the paper cannot make.

  #The (s,g) reduction leaves exactly one degree of freedom, a roll about the
  #first axis, and the paper handles it by symmetry -- roll augmentation during
  #training, frame averaging at inference -- both of which leave the backbone
  #unconstrained. The third option, constraining the weights, is the one we have
  #argued we cannot evaluate: Corollary "budget" bounds POST-HOC symmetrisation
  #of a given field, and says nothing about a model that searches a different
  #hypothesis class. This module is that model.

  #Representation. Under a rotation R_theta about the first axis a 3-vector
  #splits into two irreducible pieces:
  #    the x component      -- invariant (m=0), an ordinary scalar
  #    the (y,z) components -- m=1, rotating as a complex number z -> e^{i0} z
  #So a feature is a pair (s, v): s of shape (B, Cs, N) and v of shape
  #(B, Cv, 2, N), with the rotation acting only on v's size-2 axis.

  #What is allowed, and why each choice is forced:
  #  * mixing m=1 channels must commute with e^{i0}, so the weights are COMPLEX
  #    linear maps. Real weights would also commute but cannot rotate a feature
  #    by 90 degrees, and "turn perpendicular to this obstacle" is exactly the
  #    operation obstacle avoidance needs.
  #  * no pointwise nonlinearity may touch v. Gating -- multiplying v by a
  #    scalar computed from s -- is the standard equivariant substitute.
  #  * v enters the scalar stream only through invariants (|v|^2).
  #  * pooling over obstacles must be permutation-invariant AND equivariant:
  #    max is fine for s (invariant already), but for v it is not equivariant,
  #    so v is mean-pooled.
  #  * FiLM may scale and shift s freely; on v only a scalar SCALE is allowed,
  #    plus an additive shift that is itself an m=1 feature of the conditioning.

  #The guarantee is architectural: test_equivariant.py asserts it on an
  #UNTRAINED network, which is the hard case and the only one that shows the
  #property comes from the weights' structure rather than from training.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import sinusoidal_embedding


def vec_norm2(v, eps=1e-8):
    #(B, Cv, 2, ...) -> (B, Cv, ...), the rotation invariant of each m=1 channel.
    #Squared norm rather than norm: |v| has an infinite derivative at zero and
    #the field genuinely passes through zero, so the sqrt is a gradient hazard
    #for no expressive gain.
    return (v * v).sum(dim=2) + eps


class ComplexLinear(nn.Module):
    #Channel mixing for m=1 features: z' = W z with W complex.
    #Equivariant because complex multiplication commutes: W (e^{i0} z) = e^{i0} (W z).

    def __init__(self, c_in, c_out, bias=False):
        super().__init__()
        #NO bias on m=1 features unless it is itself m=1: a constant additive
        #vector does not rotate, so it would break equivariance outright.
        assert not bias, "an m=1 feature cannot carry a constant bias"
        self.re = nn.Parameter(torch.randn(c_out, c_in) / c_in**0.5)
        self.im = nn.Parameter(torch.randn(c_out, c_in) / c_in**0.5)

    def forward(self, v):
        # v: (B, Cin, 2, ...) -> (B, Cout, 2, ...)
        y, z = v[:, :, 0], v[:, :, 1]
        out_y = torch.einsum("oi,bi...->bo...", self.re, y) - torch.einsum("oi,bi...->bo...", self.im, z)
        out_z = torch.einsum("oi,bi...->bo...", self.im, y) + torch.einsum("oi,bi...->bo...", self.re, z)
        return torch.stack([out_y, out_z], dim=2)


class ComplexConv1d(nn.Module):
    #The same idea along the WAYPOINT axis, which the group does not act on, so
    #an ordinary complex-weighted convolution is equivariant.

    def __init__(self, c_in, c_out, kernel, dilation=1):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.re = nn.Conv1d(c_in, c_out, kernel, padding=pad, dilation=dilation, bias=False)
        self.im = nn.Conv1d(c_in, c_out, kernel, padding=pad, dilation=dilation, bias=False)

    def forward(self, v):
        # v: (B, Cin, 2, N) -> (B, Cout, 2, N)
        y, z = v[:, :, 0], v[:, :, 1]
        return torch.stack([self.re(y) - self.im(z), self.im(y) + self.re(z)], dim=2)


class EquivBlock(nn.Module):
    #One residual block: scalars get a normal conv stack, m=1 features get a
    #complex conv, the two streams exchange information only through invariants
    #(v -> |v|^2 -> s) and gates (s -> sigma -> scales v).

    def __init__(self, cs, cv, cond_s, cond_v, dilation, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, cs), cs)
        self.conv_s = nn.Conv1d(cs + cv, cs, 3, padding=dilation, dilation=dilation)
        self.conv_v = ComplexConv1d(cv, cv, 3, dilation=dilation)
        #FiLM: free on scalars; on vectors a scalar gate plus an m=1 shift, both
        #derived from the conditioning so they transform correctly.
        self.film_s = nn.Linear(cond_s, 2 * cs)
        self.film_gate = nn.Linear(cond_s, cv)
        self.film_shift = ComplexLinear(cond_v, cv)
        self.norm2 = nn.GroupNorm(min(groups, cs), cs)
        self.conv_s2 = nn.Conv1d(cs, cs, 3, padding=dilation, dilation=dilation)
        self.conv_v2 = ComplexConv1d(cv, cv, 3, dilation=dilation)
        self.gate2 = nn.Conv1d(cs, cv, 1)

    def forward(self, s, v, cs_cond, cv_cond):
        h_s = F.silu(self.norm1(s))
        #the only route from the m=1 stream into the scalar stream
        h_s = self.conv_s(torch.cat([h_s, vec_norm2(v)], dim=1))
        h_v = self.conv_v(v)

        scale, shift = self.film_s(cs_cond)[..., None].chunk(2, dim=1)
        h_s = h_s * (1 + scale) + shift
        #scalar gate, then an m=1 additive shift broadcast along the waypoints
        h_v = h_v * torch.sigmoid(self.film_gate(cs_cond))[:, :, None, None]
        h_v = h_v + self.film_shift(cv_cond)[..., None]

        h_s = self.conv_s2(F.silu(self.norm2(h_s)))
        h_v = self.conv_v2(h_v) * torch.sigmoid(self.gate2(h_s))[:, :, None, :]
        return s + h_s, v + h_v


class EquivObstacleEncoder(nn.Module):
    #Per-obstacle equivariant MLP, then a permutation-invariant pool.

    def __init__(self, hidden_s=128, hidden_v=32, out_s=128, out_v=32, box_dim=12):
        super().__init__()
        assert box_dim == 12, "the equivariant encoder needs the OBB form"
        #spheres: x and radius are invariant, (y,z) is one m=1 feature
        self.sph_s = nn.Sequential(nn.Linear(2 + 1, hidden_s), nn.SiLU(),
                                   nn.Linear(hidden_s, hidden_s))
        self.sph_v = ComplexLinear(1, hidden_v)
        #boxes: 4 x components are invariant, 4 (y,z) pairs are m=1
        self.box_s = nn.Sequential(nn.Linear(4 + 4, hidden_s), nn.SiLU(),
                                   nn.Linear(hidden_s, hidden_s))
        self.box_v = ComplexLinear(4, hidden_v)
        self.out_s = nn.Sequential(nn.Linear(2 * hidden_s, out_s), nn.SiLU(),
                                   nn.Linear(out_s, out_s))
        self.out_v = ComplexLinear(2 * hidden_v, out_v)
        self.box_dim = box_dim

    @staticmethod
    def _split(t):
        #(..., 3) -> x component (...,) and (y,z) (..., 2)
        return t[..., 0], t[..., 1:3]

    def forward(self, spheres, boxes, sphere_mask=None, box_mask=None):
        sx, syz = self._split(spheres[..., :3])
        rad = spheres[..., 3]
        #invariants fed to the scalar MLP: x, radius, and |yz|^2
        s_in = torch.stack([sx, rad, (syz * syz).sum(-1)], dim=-1)
        s_feat = self.sph_s(s_in)                            # (B,S,H)
        v_feat = self.sph_v(syz.permute(0, 2, 1)[:, None])   # (B,1,2,S)->(B,Hv,2,S)

        bx = boxes[..., :3]
        e1, e2, e3 = boxes[..., 3:6], boxes[..., 6:9], boxes[..., 9:12]
        parts = [bx, e1, e2, e3]
        bs = torch.stack([p[..., 0] for p in parts], dim=-1)          # (B,K,4)
        bv = torch.stack([p[..., 1:3] for p in parts], dim=2)         # (B,K,4,2)
        b_in = torch.cat([bs, (bv * bv).sum(-1)], dim=-1)             # (B,K,8)
        b_feat = self.box_s(b_in)
        b_vfeat = self.box_v(bv.permute(0, 2, 3, 1))                  # (B,4,2,K)->(B,Hv,2,K)

        def masked_max(x, m):
            if m is not None:
                x = x.masked_fill(~m[..., None], float("-inf"))
            return torch.nan_to_num(x.amax(dim=1), neginf=0.0)

        def masked_mean(x, m):
            #MEAN, not max: max over an m=1 feature picks a component-wise
            #extremum, which is not equivariant. Mean is linear, hence both
            #permutation-invariant and equivariant.
            if m is None:
                return x.mean(dim=-1)
            w = m[:, None, None, :].to(x.dtype)
            return (x * w).sum(-1) / w.sum(-1).clamp_min(1.0)

        s = torch.cat([masked_max(s_feat, sphere_mask),
                       masked_max(b_feat, box_mask)], dim=-1)
        v = torch.cat([masked_mean(v_feat, sphere_mask),
                       masked_mean(b_vfeat, box_mask)], dim=1)
        return self.out_s(s), self.out_v(v)


class EquivVelocityField(nn.Module):
    #Drop-in replacement for FlowVelocityField on the REDUCED arm, exactly
    #SO(2)-equivariant about the first axis by construction.

    #Defaults chosen so the parameter count matches the unconstrained backbone
    #to within 1% (2.144M against 2.161M). A constrained model that is also
    #smaller would confound the architecture with capacity, and the reading a
    #reviewer would reach for is the uncharitable one.
    def __init__(self, channels=136, vec_channels=34, n_blocks=8,
                 dilations=(1, 2, 4, 8), time_dim=128, env_hidden=128,
                 env_dim=128, env_vec=32, cond_dim=192, cond_vec=32, groups=8,
                 box_dim=12, sg_dim=1, state_dim=3):
        super().__init__()
        assert state_dim == 3, "SE(3) poses need a different irrep decomposition"
        assert sg_dim == 1, ("the equivariant backbone is for the reduced arm; "
                             "a world-frame sg vector is not SO(2)-covariant")
        self.state_dim = state_dim
        self.sg_dim = sg_dim
        self.time_dim = time_dim
        self.local_geom = False

        self.obstacle_enc = EquivObstacleEncoder(env_hidden, env_vec, env_dim,
                                                 env_vec, box_dim)
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(),
                                      nn.Linear(time_dim, time_dim))
        #sg is the invariant scalar d, so it joins the scalar stream directly
        self.cond_s = nn.Sequential(nn.Linear(env_dim + sg_dim, cond_dim), nn.SiLU(),
                                    nn.Linear(cond_dim, cond_dim))
        self.cond_v = ComplexLinear(env_vec, cond_vec)

        self.in_s = nn.Conv1d(1, channels, 5, padding=2)
        self.in_v = ComplexConv1d(1, vec_channels, 5)
        self.blocks = nn.ModuleList([
            EquivBlock(channels, vec_channels, cond_dim + time_dim, cond_vec,
                       dilations[i % len(dilations)], groups)
            for i in range(n_blocks)
        ])
        self.out_norm = nn.GroupNorm(min(groups, channels), channels)
        self.out_s = nn.Conv1d(channels, 1, 1)
        self.out_v = ComplexConv1d(vec_channels, 1, 1)
        nn.init.zeros_(self.out_s.weight)
        nn.init.zeros_(self.out_s.bias)
        nn.init.zeros_(self.out_v.re.weight)
        nn.init.zeros_(self.out_v.im.weight)

    def encode_cond(self, spheres, boxes, sg, sphere_mask=None, box_mask=None):
        es, ev = self.obstacle_enc(spheres, boxes, sphere_mask, box_mask)
        return self.cond_s(torch.cat([es, sg], dim=-1)), self.cond_v(ev)

    def decode(self, x, t, c, spheres=None, boxes=None, sphere_mask=None,
               box_mask=None):
        cs_cond, cv_cond = c
        s = x[..., 0:1].transpose(1, 2)                       # (B,1,N)
        v = x[..., 1:3].transpose(1, 2)[:, None]              # (B,1,2,N)
        s, v = self.in_s(s), self.in_v(v)
        t_emb = self.time_mlp(sinusoidal_embedding(t, self.time_dim))
        cs = torch.cat([t_emb, cs_cond], dim=-1)
        for blk in self.blocks:
            s, v = blk(s, v, cs, cv_cond)
        s = self.out_s(F.silu(self.out_norm(s)))              # (B,1,N)
        v = self.out_v(v)                                     # (B,1,2,N)
        return torch.cat([s.transpose(1, 2), v[:, 0].permute(0, 2, 1)], dim=-1)

    def forward(self, x, t, spheres, boxes, sg, sphere_mask=None, box_mask=None):
        return self.decode(x, t,
                           self.encode_cond(spheres, boxes, sg, sphere_mask, box_mask),
                           spheres, boxes, sphere_mask, box_mask)
