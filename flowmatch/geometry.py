#uniform roll augmentation at training
#frame averaging at inference


import torch


def sg_frame(start, goal, theta=None):
    d_vec = goal - start                                 
    d = torch.linalg.norm(d_vec, dim=-1)                
    x_hat = d_vec / d.clamp_min(1e-12)[..., None]

    axis = x_hat.abs().argmin(dim=-1)                    
    e = torch.zeros_like(x_hat).scatter_(-1, axis[..., None], 1.0)
    y_hat = torch.linalg.cross(x_hat, e, dim=-1)
    y_hat = y_hat / torch.linalg.norm(y_hat, dim=-1, keepdim=True)
    z_hat = torch.linalg.cross(x_hat, y_hat, dim=-1)      #right-handed

    if theta is not None:
        c, s = torch.cos(theta)[..., None], torch.sin(theta)[..., None]
        y_hat, z_hat = c * y_hat + s * z_hat, -s * y_hat + c * z_hat

    R = torch.stack([x_hat, y_hat, z_hat], dim=-2)        #rows = basis
    origin = 0.5 * (start + goal)
    return R, origin, d


def apply_points(R, origin, p):
    #Full affine
    return torch.einsum("bij,bkj->bki", R, p - origin[:, None, :])


def apply_vectors(R, v):
    #rotation only, for free vectors
    return torch.einsum("bij,bkj->bki", R, v)


def aabb_edges(half_extents):
    return torch.diag_embed(half_extents)


def box_features(center, edges):
    #center (B,K,3) + edges (B,K,3,3) -> (B,K,12)
    return torch.cat([center, edges.reshape(*edges.shape[:-2], 9)], dim=-1)


def split_box_features(feat):
    #(B,K,12) -> center (B,K,3), edges (B,K,3,3)
    center, flat = feat[..., :3], feat[..., 3:]
    return center, flat.reshape(*flat.shape[:-1], 3, 3)


def rotate_box_features(feat, R, origin):
    center, edges = split_box_features(feat)
    center_r = apply_points(R, origin, center)
    K = edges.shape[1]
    edges_r = apply_vectors(R, edges.reshape(edges.shape[0], K * 3, 3))
    return box_features(center_r, edges_r.reshape(edges.shape[0], K, 3, 3))


def rotate_sphere_features(feat, R, origin):
    #(B,K,4) = center (POINT) + radius (invariant scalar, m=0).
    center_r = apply_points(R, origin, feat[..., :3])
    return torch.cat([center_r, feat[..., 3:]], dim=-1)


def check_frame(R, origin, start, goal, d, atol=None):
    if atol is None:
        atol = 1e-4
    with torch.autocast(device_type=R.device.type, enabled=False):
        R, origin = R.float(), origin.float()
        start, goal, d = start.float(), goal.float(), d.float()
        s_r = apply_points(R, origin, start[:, None, :])[:, 0, :]
        g_r = apply_points(R, origin, goal[:, None, :])[:, 0, :]
        zeros = torch.zeros_like(d)
        want_s = torch.stack([-0.5 * d, zeros, zeros], dim=-1)
        want_g = torch.stack([0.5 * d, zeros, zeros], dim=-1)
        assert torch.allclose(s_r, want_s, atol=atol), (
            f"start does not land on (-d/2,0,0): max err "
            f"{(s_r - want_s).abs().max().item():.2e}"
        )
        assert torch.allclose(g_r, want_g, atol=atol), (
            f"goal does not land on (+d/2,0,0): max err "
            f"{(g_r - want_g).abs().max().item():.2e}"
        )
        det = torch.linalg.det(R)
        assert (det > 0).all(), f"left-handed frame: min det {det.min().item():.4f}"
        eye = torch.eye(3, device=R.device, dtype=R.dtype).expand_as(R)
        assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=atol), (
            "R not orthonormal"
        )
