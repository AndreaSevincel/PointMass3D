"""
Dataset + normalization for flow-matching over PointMass3D trajectories.

Each env_*.npz holds many expert trajectories (T, N, 3) with matching
starts/goals (T, 3) and a fixed obstacle set (spheres (S,4), boxes (B,6)).
We flatten all trajectories into one training set and keep, per environment,
the obstacle tensors so the conditioning encoder can look them up by env id.

Normalization: coordinates are recentred by a per-axis mean and rescaled by a
single (isotropic) scalar std, so that data roughly matches the N(0, I) prior
used by the flow-matching model. Radii / half-extents are scaled by the same
scalar so the geometry stays consistent.
"""

import glob
import json
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


def load_envs(data_dir):
    #Load every env_*.npz in a directory into plain numpy arrays
    files = sorted(glob.glob(os.path.join(data_dir, "env_*.npz")))
    if not files:
        raise FileNotFoundError(f"no env_*.npz found in {data_dir}")
    envs = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        envs.append(
            {
                "spheres": d["spheres"].astype(np.float32),   # (S, 4)
                "boxes": d["boxes"].astype(np.float32),       # (B, 6)
                "trajs": d["trajs"].astype(np.float32),       # (T, N, 3)
                "starts": d["starts"].astype(np.float32),     # (T, 3)
                "goals": d["goals"].astype(np.float32),       # (T, 3)
                "robot_radius": float(d["robot_radius"]),
                "env_seed": int(d["env_seed"]),
                "path": f,
            }
        )
    return envs


class TrajectoryDataset(Dataset):
    #One item = a single expert trajectory plus the id of its environment

    def __init__(self, envs, normalizer, indices=None):
        self.norm = normalizer
        S = max(e["spheres"].shape[0] for e in envs)
        B = max(e["boxes"].shape[0] for e in envs)
        E = len(envs)
        spheres = np.zeros((E, S, 4), dtype=np.float32)
        boxes = np.zeros((E, B, 6), dtype=np.float32)
        sphere_mask = np.zeros((E, S), dtype=bool)
        box_mask = np.zeros((E, B), dtype=bool)
        for ei, e in enumerate(envs):
            ns, nb = e["spheres"].shape[0], e["boxes"].shape[0]
            sp = e["spheres"].astype(np.float32).copy()
            sp[:, :3] = normalizer.norm_pts(sp[:, :3])
            sp[:, 3] = normalizer.norm_len(sp[:, 3])
            bx = e["boxes"].astype(np.float32).copy()
            bx[:, :3] = normalizer.norm_pts(bx[:, :3])
            bx[:, 3:] = normalizer.norm_len(bx[:, 3:])
            spheres[ei, :ns] = sp
            boxes[ei, :nb] = bx
            sphere_mask[ei, :ns] = True
            box_mask[ei, :nb] = True
        self.spheres = torch.from_numpy(spheres)          # (E, S, 4)
        self.boxes = torch.from_numpy(boxes)              # (E, B, 6)
        self.sphere_mask = torch.from_numpy(sphere_mask)  # (E, S)
        self.box_mask = torch.from_numpy(box_mask)        # (E, B)

        #flatten trajectories, remembering which env each came from
        trajs, starts, goals, env_ids = [], [], [], []
        for ei, e in enumerate(envs):
            trajs.append(e["trajs"])
            starts.append(e["starts"])
            goals.append(e["goals"])
            env_ids.append(np.full(len(e["trajs"]), ei, dtype=np.int64))
        trajs = np.concatenate(trajs)     # (M, N, 3)
        starts = np.concatenate(starts)   # (M, 3)
        goals = np.concatenate(goals)     # (M, 3)
        env_ids = np.concatenate(env_ids)

        self.trajs = torch.from_numpy(normalizer.norm_pts(trajs))
        self.starts = torch.from_numpy(normalizer.norm_pts(starts))
        self.goals = torch.from_numpy(normalizer.norm_pts(goals))
        self.env_ids = torch.from_numpy(env_ids)

        self.indices = (
            np.arange(len(self.trajs)) if indices is None else np.asarray(indices)
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


def train_val_split(n, val_frac, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(round(val_frac * n))
    return perm[n_val:], perm[:n_val]


def build_datasets(data_dir, val_frac=0.05, seed=0):
    #load envs, fit the normalizer on the training split, return datasets.
    envs = load_envs(data_dir)
    total = sum(len(e["trajs"]) for e in envs)
    train_idx, val_idx = train_val_split(total, val_frac, seed)

    all_trajs = np.concatenate([e["trajs"] for e in envs])
    normalizer = Normalizer.from_trajs(all_trajs[train_idx])

    train_ds = TrajectoryDataset(envs, normalizer, train_idx)
    val_ds = (
        TrajectoryDataset(envs, normalizer, val_idx) if len(val_idx) else None
    )
    return train_ds, val_ds, normalizer, envs
