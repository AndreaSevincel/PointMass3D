
  #.venv/bin/python sample_flow.py --ckpt checkpoints/flow.pt --data data1 \
  #    --env-idx 0 --n-pairs 5 --n-samples 20 --steps 100 --plot out.png
import argparse

import numpy as np
import torch

from flowmatch.data import Normalizer
from flowmatch.flow import sample
from flowmatch.model import FlowVelocityField
from pointmass3d import (
    BoxObstacle,
    PointMass3DEnv,
    SphereObstacle,
    mean_sq_accel,
    min_clearance,
    path_length,
    plot_scene,
)


def build_env(npz, robot_radius):
    obstacles = [SphereObstacle(s[:3], s[3]) for s in npz["spheres"]]
    obstacles += [BoxObstacle(b[:3], b[3:]) for b in npz["boxes"]]
    return PointMass3DEnv(obstacles, robot_radius=robot_radius)


def distinct_pairs(starts, goals, n):
    """Indices of the first n distinct (start, goal) pairs."""
    seen, idx = set(), []
    for i in range(len(starts)):
        key = np.round(np.concatenate([starts[i], goals[i]]), 4).tobytes()
        if key not in seen:
            seen.add(key)
            idx.append(i)
        if len(idx) >= n:
            break
    return idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/flow.pt")
    ap.add_argument("--data", type=str, default="data1")
    ap.add_argument("--env-idx", type=int, default=0)
    ap.add_argument("--n-pairs", type=int, default=5)
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--anchor-endpoints", action="store_true")
    ap.add_argument("--plot", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    norm = Normalizer.from_dict(ckpt["normalizer"])
    N = ckpt["n_waypoints"]

    model = FlowVelocityField(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["ema"])
    model.eval()

    npz = np.load(f"{args.data}/env_{args.env_idx:04d}.npz", allow_pickle=True)
    robot_radius = float(npz["robot_radius"])
    env = build_env(npz, robot_radius)

    # Normalized, batched obstacle tensors (shared across every sample).
    sp = npz["spheres"].astype(np.float32).copy()
    sp[:, :3] = norm.norm_pts(sp[:, :3]); sp[:, 3] = norm.norm_len(sp[:, 3])
    bx = npz["boxes"].astype(np.float32).copy()
    bx[:, :3] = norm.norm_pts(bx[:, :3]); bx[:, 3:] = norm.norm_len(bx[:, 3:])
    sp_t = torch.from_numpy(sp).to(device)
    bx_t = torch.from_numpy(bx).to(device)

    starts, goals, trajs = npz["starts"], npz["goals"], npz["trajs"]
    pair_idx = distinct_pairs(starts, goals, args.n_pairs)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    all_free, all_clear, all_ep, all_len, all_acc = [], [], [], [], []
    first_samples = None

    for pi in pair_idx:
        start, goal = starts[pi], goals[pi]
        s_n = torch.from_numpy(norm.norm_pts(start).astype(np.float32))
        g_n = torch.from_numpy(norm.norm_pts(goal).astype(np.float32))
        B = args.n_samples
        s_b = s_n.repeat(B, 1).to(device)
        g_b = g_n.repeat(B, 1).to(device)
        sp_b = sp_t.unsqueeze(0).expand(B, -1, -1)
        bx_b = bx_t.unsqueeze(0).expand(B, -1, -1)

        x = sample(
            model, sp_b, bx_b, s_b, g_b,
            n_waypoints=N, n_steps=args.steps,
            anchor_endpoints=args.anchor_endpoints, device=device, generator=gen,
        )
        paths = norm.denorm_pts(x.cpu().numpy())  # (B, N, 3)

        free = [env.path_free(p) for p in paths]
        all_free.extend(free)
        all_clear.extend(min_clearance(env, p) for p in paths)
        all_ep.extend(
            0.5 * (np.linalg.norm(p[0] - start) + np.linalg.norm(p[-1] - goal))
            for p in paths
        )
        all_len.extend(path_length(p) for p in paths)
        all_acc.extend(mean_sq_accel(p) for p in paths)
        if first_samples is None:
            first_samples = (start, goal, paths, trajs[pi])

    def stat(name, xs, fmt="{:.4f}"):
        xs = np.asarray(xs)
        print(f"  {name:<18} mean {fmt.format(xs.mean())}   "
              f"min {fmt.format(xs.min())}   max {fmt.format(xs.max())}")

    n = len(all_free)
    print(f"\nenv {args.env_idx}: {len(pair_idx)} pairs x {args.n_samples} samples "
          f"= {n} trajectories  (anchor={args.anchor_endpoints})")
    print(f"  collision-free      {100*np.mean(all_free):.1f}%")
    stat("min clearance", all_clear)
    stat("endpoint error", all_ep)
    stat("path length", all_len)
    stat("mean sq accel", all_acc, "{:.5f}")

    if args.plot and first_samples is not None:
        start, goal, paths, expert = first_samples
        trajs_to_plot = {f"sample {k}": paths[k] for k in range(min(6, len(paths)))}
        trajs_to_plot["expert"] = expert
        plot_scene(
            env, trajectories=trajs_to_plot, start=start, goal=goal,
            title=f"flow-matching samples (env {args.env_idx})", save=args.plot,
        )
        print(f"  plot -> {args.plot}")


if __name__ == "__main__":
    main()
