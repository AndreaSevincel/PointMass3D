#Dataset loading for flow matching.

#Each env_*.npz holds expert trajectories (T, N, 3) with matching
#starts/goals (T, 3) and a fixed obstacle set (spheres (S,4), boxes (B,6)).

#Memory layout: all trajectories live in ONE flat array shared by the train
#and val datasets (which differ only in their index lists). Shards are loaded
#directly into a preallocated buffer — counts come from the tiny `starts`
#arrays in a first pass — so peak memory is one copy of the data plus one
#shard, not the 4x of naive concatenate-then-split.

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class Normalizer:
    #isotropic normalizing

    def __init__(self, center, scale):
        self.center = np.asarray(center, dtype=np.float32)  # (3,)
        self.scale = float(scale)                           # scalar

    @classmethod
    def from_trajs(cls, trajs):
        pts = trajs.reshape(-1, 3)
        center = pts.mean(axis=0)
        scale = float(pts.std())
        return cls(center, scale)

    @classmethod
    def fit_indexed(cls, trajs, idx, chunk=200_000):
        #Fit on trajs[idx] without materializing the subset.
        #Matches from_trajs exactly: per-coord mean, global scalar std.
        #Sorted access: idx is a random permutation slice, and gathering it in
        #random order across a multi-GB array is cache-hostile. Order does not
        #affect a mean or a variance, so sort once and stream sequentially.
        idx = np.sort(np.asarray(idx))
        s = np.zeros(3, dtype=np.float64)
        sq = 0.0
        n = 0
        for lo in range(0, len(idx), chunk):
            block = trajs[idx[lo:lo + chunk]].reshape(-1, 3).astype(np.float64)
            s += block.sum(axis=0)
            sq += float((block * block).sum())
            n += block.shape[0]
        center = s / max(n, 1)
        mu_all = s.sum() / max(3 * n, 1)
        scale = float(np.sqrt(max(sq / max(3 * n, 1) - mu_all * mu_all, 1e-12)))
        return cls(center.astype(np.float32), scale)

    #points (..., 3)
    def norm_pts(self, p):
        return (p - self.center) / self.scale

    def denorm_pts(self, p):
        return p * self.scale + self.center

    #lengths scale only, no shift
    def norm_len(self, x):
        return x / self.scale

    def to_dict(self):
        return {"center": self.center.tolist(), "scale": self.scale}

    @classmethod
    def from_dict(cls, d):
        return cls(d["center"], d["scale"])


def load_envs(data_dir, n_envs=None, env_start=0):
    #Load every env_*.npz in a directory into plain numpy arrays.
    #n_envs takes N shards starting at env_start (nested subsets for the scale
    #sweep; shards are per-env seeded, so a given range is fixed and
    #reproducible). Reserve a tail range as a never-trained test set.
    #NOTE: trajectories are NOT loaded here (they can be tens of GB); use
    #load_trajs for the flat trajectory arrays.
    files = sorted(glob.glob(os.path.join(data_dir, "env_*.npz")))
    if not files:
        raise FileNotFoundError(f"no env_*.npz found in {data_dir}")
    files = files[env_start:]
    if n_envs is not None:
        if n_envs > len(files):
            raise ValueError(
                f"asked for {n_envs} envs from offset {env_start}, "
                f"{data_dir} has only {len(files)} beyond it"
            )
        files = files[:n_envs]
    envs = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        envs.append(
            {
                "spheres": d["spheres"].astype(np.float32),   # (S, 4)
                "boxes": d["boxes"].astype(np.float32),       # (B, 6)
                "starts": d["starts"].astype(np.float32),     # (T, 3)
                "goals": d["goals"].astype(np.float32),       # (T, 3)
                "n_trajs": d["starts"].shape[0],
                "robot_radius": float(d["robot_radius"]),
                "env_seed": int(d["env_seed"]),
                "path": f,
            }
        )
    return envs


def load_trajs(envs):
    #Second pass: load every shard's trajectories into one preallocated
    #float32 array. Returns (trajs (M,N,3), env_ids (M,)).
    total = sum(e["n_trajs"] for e in envs)
    probe = np.load(envs[0]["path"], allow_pickle=True)
    N = probe["trajs"].shape[1]
    del probe

    trajs = np.empty((total, N, 3), dtype=np.float32)
    env_ids = np.empty(total, dtype=np.int64)
    ofs = 0
    for ei, e in enumerate(envs):
        d = np.load(e["path"], allow_pickle=True)
        t = d["trajs"]
        n = t.shape[0]
        trajs[ofs:ofs + n] = t  # casts float64 shards on the fly
        env_ids[ofs:ofs + n] = ei
        ofs += n
        del d, t
    assert ofs == total
    return trajs, env_ids


class TrajectoryDataset(Dataset):
    #One item = a single expert trajectory plus the id of its environment.
    #Built once via from_envs; train/val views share the same tensors and
    #differ only in `indices` (see subset()).

    def __init__(self, norm, spheres, boxes, sphere_mask, box_mask,
                 trajs, starts, goals, env_ids, indices):
        self.norm = norm
        self.spheres = spheres          # (E, S, 4)
        self.boxes = boxes              # (E, B, 6)
        self.sphere_mask = sphere_mask  # (E, S)
        self.box_mask = box_mask        # (E, B)
        self.trajs = trajs              # (M, N, 3) shared
        self.starts = starts            # (M, 3) shared
        self.goals = goals              # (M, 3) shared
        self.env_ids = env_ids          # (M,) shared
        self.indices = np.asarray(indices)

    @classmethod
    def from_envs(cls, envs, normalizer, trajs, env_ids, indices=None):
        #trajs/env_ids come from load_trajs; trajs is normalized IN PLACE.
        #Boxes are stored as 12-dim center + three half-edge VECTORS. In world
        #frame those are axis-aligned -- (hx,0,0), (0,hy,0), (0,0,hz) -- so this
        #is a lossless re-encoding of the stored 6 numbers with structural
        #zeros. It exists because the (s,g) reduction is a general SO(3)
        #rotation, under which an AABB becomes an OBB.
        S = max(e["spheres"].shape[0] for e in envs)
        B = max(e["boxes"].shape[0] for e in envs)
        E = len(envs)
        spheres = np.zeros((E, S, 4), dtype=np.float32)
        boxes = np.zeros((E, B, 12), dtype=np.float32)
        sphere_mask = np.zeros((E, S), dtype=bool)
        box_mask = np.zeros((E, B), dtype=bool)
        for ei, e in enumerate(envs):
            ns, nb = e["spheres"].shape[0], e["boxes"].shape[0]
            sp = e["spheres"].copy()
            sp[:, :3] = normalizer.norm_pts(sp[:, :3])
            sp[:, 3] = normalizer.norm_len(sp[:, 3])
            #center is a POINT (shift+scale); half-extents are LENGTHS (scale)
            bx_c = normalizer.norm_pts(e["boxes"][:, :3])
            bx_h = normalizer.norm_len(e["boxes"][:, 3:])
            boxes[ei, :nb, :3] = bx_c
            boxes[ei, :nb, 3::4] = bx_h  # diagonal of the 3x3 edge block
            spheres[ei, :ns] = sp
            sphere_mask[ei, :ns] = True
            box_mask[ei, :nb] = True

        starts = np.concatenate([e["starts"] for e in envs])  # (M, 3) small
        goals = np.concatenate([e["goals"] for e in envs])

        #normalize the big array in place — no full-size temporaries.
        #A state is 3 numbers (a point-mass waypoint) or 9 (an SE(3) pose:
        #position + 6D rotation). Only the POSITION is normalized; the rotation
        #columns are unit vectors that are already O(1) and would be corrupted
        #by a shift -- the same points-vs-vectors distinction as the box edges.
        if trajs.shape[-1] == 3:
            trajs -= normalizer.center
            trajs /= normalizer.scale
            starts = normalizer.norm_pts(starts).astype(np.float32)
            goals = normalizer.norm_pts(goals).astype(np.float32)
        else:
            trajs[..., :3] -= normalizer.center
            trajs[..., :3] /= normalizer.scale
            starts = starts.copy().astype(np.float32)
            goals = goals.copy().astype(np.float32)
            starts[..., :3] = normalizer.norm_pts(starts[..., :3])
            goals[..., :3] = normalizer.norm_pts(goals[..., :3])

        if indices is None:
            indices = np.arange(len(trajs))
        return cls(
            Normalizer.from_dict(normalizer.to_dict()),
            torch.from_numpy(spheres), torch.from_numpy(boxes),
            torch.from_numpy(sphere_mask), torch.from_numpy(box_mask),
            torch.from_numpy(trajs), torch.from_numpy(starts),
            torch.from_numpy(goals), torch.from_numpy(env_ids),
            indices,
        )

    def subset(self, indices):
        #A view over the same tensors with a different index list.
        return TrajectoryDataset(
            self.norm, self.spheres, self.boxes, self.sphere_mask,
            self.box_mask, self.trajs, self.starts, self.goals,
            self.env_ids, indices,
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        return {
            "traj": self.trajs[j],        # (N, 3)
            "start": self.starts[j],      # (3,)
            "goal": self.goals[j],        # (3,)
            "env_id": self.env_ids[j],    # ()
        }


def env_features(npz, normalizer, box_dim=12):
    #Normalized obstacle tensors for one env shard, as the model expects them.
    #Mirrors TrajectoryDataset.from_envs so training and inference cannot drift.
    #box_dim=12 is center + three half-edge VECTORS; 6 is the legacy
    #center + half-extents form kept for pre-OBB checkpoints.
    sp = npz["spheres"].astype(np.float32).copy()
    sp[:, :3] = normalizer.norm_pts(sp[:, :3])
    sp[:, 3] = normalizer.norm_len(sp[:, 3])

    raw = npz["boxes"].astype(np.float32)
    nb = raw.shape[0]
    if box_dim == 12:
        bx = np.zeros((nb, 12), dtype=np.float32)
        bx[:, :3] = normalizer.norm_pts(raw[:, :3])   # center is a POINT
        bx[:, 3::4] = normalizer.norm_len(raw[:, 3:])  # half-edges: diagonal
    elif box_dim == 6:
        bx = raw.copy()
        bx[:, :3] = normalizer.norm_pts(bx[:, :3])
        bx[:, 3:] = normalizer.norm_len(bx[:, 3:])
    else:
        raise ValueError(f"unsupported box_dim {box_dim}")
    return torch.from_numpy(sp), torch.from_numpy(bx)


def train_val_split(n, val_frac, seed=0):
    #Per-trajectory random split. LEAKY on this dataset: with 30 near-duplicate
    #paths per (start, goal), a held-out trajectory has a train neighbour at RMS
    #distance ~0.03 versus a within-pair std of ~0.054, so val measures nothing
    #about generalization. Kept for comparison; prefer split_by="pair"/"env".
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(round(val_frac * n))
    return perm[n_val:], perm[:n_val]


def _lex_greater(a, b):
    #Row-wise lexicographic a > b, vectorized.
    gt, lt = a > b, a < b
    diff = gt | lt
    first = np.argmax(diff, axis=1)
    rows = np.arange(len(a))
    return diff.any(axis=1) & gt[rows, first]


def pair_groups(envs):
    #Stable group id per trajectory, one group per (env, {start, goal}).
    #The endpoint pair is canonicalized so a path and its reverse-mode=flip
    #copy share a group -- otherwise a reversed duplicate of a training path
    #lands in val and leaks the same geometry back in.
    out, base = [], 0
    for e in envs:
        s, g = e["starts"], e["goals"]
        swap = _lex_greater(s, g)[:, None]
        lo = np.where(swap, g, s)
        hi = np.where(swap, s, g)
        key = np.round(np.concatenate([lo, hi], axis=1), 6)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        out.append(inv + base)
        base += int(inv.max()) + 1
    return np.concatenate(out)


def grouped_split(groups, val_frac, seed=0):
    #Hold out WHOLE groups, so nothing correlated with a val item is in train.
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    perm = rng.permutation(uniq)
    n_val = int(round(val_frac * len(uniq)))
    if n_val == 0 and len(uniq) > 1:
        n_val = 1
    val_groups = perm[:n_val]
    is_val = np.isin(groups, val_groups)
    return np.where(~is_val)[0], np.where(is_val)[0]


def build_datasets(
    data_dir, val_frac=0.05, seed=0, n_envs=None, env_start=0, split_by="pair"
):
    #load envs, fit the normalizer on the training split, return datasets.
    #split_by controls what the val set actually measures:
    #  traj -- per-trajectory random. LEAKY here (see train_val_split).
    #  pair -- whole (start, goal) pairs held out, same environments. Measures
    #          generalization to new queries in a known layout. Default.
    #  env  -- whole environments held out. Measures generalization to new
    #          layouts, but is coarse at small n_envs (20 envs, 5% -> 1 env).
    envs = load_envs(data_dir, n_envs=n_envs, env_start=env_start)
    trajs, env_ids = load_trajs(envs)

    if split_by == "traj":
        train_idx, val_idx = train_val_split(len(trajs), val_frac, seed)
    elif split_by == "pair":
        train_idx, val_idx = grouped_split(pair_groups(envs), val_frac, seed)
    elif split_by == "env":
        train_idx, val_idx = grouped_split(env_ids, val_frac, seed)
    else:
        raise ValueError(f"split_by must be traj|pair|env, got {split_by!r}")

    normalizer = Normalizer.fit_indexed(trajs, train_idx)

    full = TrajectoryDataset.from_envs(envs, normalizer, trajs, env_ids)
    train_ds = full.subset(train_idx)
    val_ds = full.subset(val_idx) if len(val_idx) else None
    return train_ds, val_ds, normalizer, envs
