
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
  #  * pooling over obstacles must be permutation-invariant AND equivariant.
  #    Max is fine for s, which is invariant already. For v neither max nor mean
  #    will do: max is not equivariant, and mean IS equivariant but computes the
  #    centroid of a roughly symmetric obstacle cloud -- near zero, retaining
  #    about 14% of a typical obstacle's magnitude on this benchmark, so the one
  #    channel carrying direction in the plane arrives almost empty. We pool with
  #    weights derived from the per-obstacle INVARIANTS instead: the weights are
  #    scalars, so the weighted sum stays equivariant, and a softmax over
  #    obstacles lets the encoder attend to the relevant one rather than
  #    averaging them all away.
  #  * FiLM may scale and shift s freely; on v only a scalar SCALE is allowed,
  #    plus an additive shift that is itself an m=1 feature of the conditioning.

  #The guarantee is architectural: test_equivariant.py asserts it on an
  #UNTRAINED network, which is the hard case and the only one that shows the
  #property comes from the weights' structure rather than from training.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import sinusoidal_embedding


def vec_rms_norm(v, weight, eps=1e-6):
    #Normalise m=1 features by an INVARIANT: the RMS magnitude across channels at
    #each position. Dividing by a rotation-invariant scalar keeps the result
    #equivariant, which is why an ordinary GroupNorm cannot be used here -- it
    #would mix the two components and destroy the property.
    #
    #Without this the vector stream runs through eight residual blocks with
    #nothing controlling its scale while the scalar stream is normalised twice
    #per block. The first version of this module omitted it.
    #
    #Computed in fp32 even under autocast. torch forces GroupNorm to fp32 by its
    #own cast policy, so the scalar stream is protected and a hand-written norm
    #is not: squaring an fp16 activation flushes anything below 2.4e-4 to zero,
    #the clamp then returns eps and the division amplifies that channel by 1e3.
    #Measured at initialisation the margin is wide (smallest mag2 ~1e-5 against
    #fp16's 6e-8 floor), so this is insurance, not a diagnosed fault.
    dtype = v.dtype
    v = v.float()
    mag2 = (v * v).sum(dim=2)                                   # (B,Cv,...)
    rms = mag2.mean(dim=1, keepdim=True).clamp_min(eps).sqrt()  # (B,1,...)
    shape = (1, -1, 1) + (1,) * (v.dim() - 3)
    return (v / rms.unsqueeze(2) * weight.float().view(*shape)).to(dtype)


def vec_norm2(v, eps=1e-8):
    #(B, Cv, 2, ...) -> (B, Cv, ...), the rotation invariant of each m=1 channel.
    #Squared norm rather than norm: |v| has an infinite derivative at zero and
    #the field genuinely passes through zero, so the sqrt is a gradient hazard
    #for no expressive gain.
    #fp32 for the same reason as vec_rms_norm -- and here eps=1e-8 is below
    #fp16's smallest subnormal, so in half precision the guard silently is not
    #one.
    return ((v.float() * v.float()).sum(dim=2) + eps).to(v.dtype)


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
        #per-channel invariant gain: the m=1 analogue of a normalisation's scale
        self.vnorm1 = nn.Parameter(torch.ones(cv))
        self.vnorm2 = nn.Parameter(torch.ones(cv))
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
        h_v = self.conv_v(vec_rms_norm(v, self.vnorm1))

        scale, shift = self.film_s(cs_cond)[..., None].chunk(2, dim=1)
        h_s = h_s * (1 + scale) + shift
        #scalar gate, then an m=1 additive shift broadcast along the waypoints
        h_v = h_v * torch.sigmoid(self.film_gate(cs_cond))[:, :, None, None]
        h_v = h_v + self.film_shift(cv_cond)[..., None]

        h_s = self.conv_s2(F.silu(self.norm2(h_s)))
        h_v = self.conv_v2(vec_rms_norm(h_v, self.vnorm2))
        h_v = h_v * torch.sigmoid(self.gate2(h_s))[:, :, None, :]
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
        #one attention logit per m=1 channel, from the invariant features only
        self.sph_w = nn.Linear(hidden_s, hidden_v)
        #boxes: 4 x components are invariant, 4 (y,z) pairs are m=1
        self.box_s = nn.Sequential(nn.Linear(4 + 4, hidden_s), nn.SiLU(),
                                   nn.Linear(hidden_s, hidden_s))
        self.box_v = ComplexLinear(4, hidden_v)
        self.box_w = nn.Linear(hidden_s, hidden_v)
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

        def attn_pool(x, logits, m):
            #x (B,Cv,2,K); logits (B,K,Cv) from INVARIANTS only, so the weights
            #are scalars and the pooled vector is still equivariant. Softmax over
            #the obstacle axis keeps it permutation-invariant.
            w = logits.permute(0, 2, 1)                      # (B,Cv,K)
            if m is not None:
                w = w.masked_fill(~m[:, None, :], float("-inf"))
            w = torch.softmax(w, dim=-1)
            return (x * w[:, :, None, :]).sum(-1)            # (B,Cv,2)

        s = torch.cat([masked_max(s_feat, sphere_mask),
                       masked_max(b_feat, box_mask)], dim=-1)
        v = torch.cat([attn_pool(v_feat, self.sph_w(s_feat), sphere_mask),
                       attn_pool(b_vfeat, self.box_w(b_feat), box_mask)], dim=1)
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
                 box_dim=12, sg_dim=1, state_dim=3,
                 #ORACLE DIAGNOSTIC, off by default -- see FlowVelocityField for
                 #why it is not a method. It matters more here than there. The
                 #only obstacle-derived m=1 signal reaching the trunk is
                 #cv_cond, ONE vector per query broadcast to all N waypoints, so
                 #"which way do I dodge, here" has no channel to arrive on; the
                 #scalar stream carries per-waypoint geometry but reaches the
                 #vector stream only through non-negative gates. The SDF
                 #gradient splits cleanly along the irreps -- d and g_x are
                 #invariant, g_yz is m=1 -- so it can enter both streams at the
                 #right type and test whether that missing channel is the
                 #bottleneck.
                 local_geom=False):
        super().__init__()
        assert state_dim == 3, "SE(3) poses need a different irrep decomposition"
        assert sg_dim == 1, ("the equivariant backbone is for the reduced arm; "
                             "a world-frame sg vector is not SO(2)-covariant")
        self.state_dim = state_dim
        self.sg_dim = sg_dim
        self.time_dim = time_dim
        self.local_geom = local_geom

        self.obstacle_enc = EquivObstacleEncoder(env_hidden, env_vec, env_dim,
                                                 env_vec, box_dim)
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(),
                                      nn.Linear(time_dim, time_dim))
        #sg is the invariant scalar d, so it joins the scalar stream directly
        self.cond_s = nn.Sequential(nn.Linear(env_dim + sg_dim, cond_dim), nn.SiLU(),
                                    nn.Linear(cond_dim, cond_dim))
        self.cond_v = ComplexLinear(env_vec, cond_vec)

        #+2 invariants (the SDF value and the x component of its gradient) and
        #+1 m=1 feature (the gradient's (y,z) part), each to its own stream
        self.in_s = nn.Conv1d(1 + 2 * local_geom, channels, 5, padding=2)
        self.in_v = ComplexConv1d(1 + local_geom, vec_channels, 5)
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
        if self.local_geom:
            if spheres is None or boxes is None:
                raise ValueError(
                    "local_geom=True needs the obstacle tensors at decode(); "
                    "pass the same frame-transformed tensors used for encode_cond"
                )
            from .sdf import scene_sdf_and_grad

            d, g = scene_sdf_and_grad(x[..., :3], spheres, boxes,
                                      sphere_mask, box_mask)
            #d and g_x are invariant under the roll, g_yz rotates with it --
            #which is the whole reason this can be added without breaking the
            #constraint. test_equivariant.py checks it rather than trusting it.
            s = torch.cat([s, d[:, None], g[..., 0][:, None]], dim=1)  # (B,3,N)
            v = torch.cat([v, g[..., 1:3].transpose(1, 2)[:, None]], dim=1)
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
