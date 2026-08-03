
  #.venv/bin/python sweep_steps.py --ckpt checkpoints/flow_cpu_demo.pt --data data1 \
  #    --n-envs 5 --n-pairs 5 --n-samples 20 --steps 100 50 20 10 8 4

  #Step 0 of the equivariance plan: how few ODE steps can the trained flow
  #tolerate? Seed-matched across step counts (identical prior draws), so any
  #metric drift is attributable to integration error alone. The 100-step row is
  #the reference; the budget is the smallest count that holds its quality.

import argparse
import time

import numpy as np
import torch

from flowmatch import tracking
from flowmatch.data import Normalizer, env_features
from flowmatch.flow import sample, sample_reduced
from flowmatch.model import build_model
from pointmass3d import (
    BoxObstacle,
    PointMass3DEnv,
    SphereObstacle,
    mean_sq_accel,
    min_clearance,
    path_length,
)
from sample_flow import build_env, distinct_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/flow_cpu_demo.pt")
    ap.add_argument("--data", type=str, default="data1")
    ap.add_argument("--n-envs", type=int, default=5)
    ap.add_argument("--env-start", type=int, default=0,
                    help="first env shard to evaluate; point this at the "
                         "held-out test range, not at envs the model trained on")
    ap.add_argument("--n-pairs", type=int, default=5, help="pairs per env")
    ap.add_argument("--n-samples", type=int, default=20, help="samples per pair")
    ap.add_argument("--steps", type=int, nargs="+", default=[100, 50, 20, 10, 8, 4])
    ap.add_argument("--k-fa", type=int, default=1,
                    help="frame-averaging width (reduced checkpoints only)")
    ap.add_argument("--residual", action="store_true",
                    help="also report std(v_k)/||mean(v_k)||, the non-equivariance "
                         "frame averaging is removing. 0 => already equivariant, "
                         "so a larger K_FA cannot help. Needs --k-fa > 1")
    ap.add_argument("--anchor-endpoints", action="store_true")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    tracking.add_args(ap)
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    norm = Normalizer.from_dict(ckpt["normalizer"])
    N = ckpt["n_waypoints"]
    model = build_model(ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    reduced = model.cond_enc.sg_dim == 1
    if args.k_fa > 1 and not reduced:
        raise SystemExit("--k-fa > 1 requires a reduced-frame checkpoint (sg_dim=1)")
    box_dim = model.obstacle_enc.box_dim

    run = tracking.from_args(
        args,
        name=f"sweep-steps-{'treat' if reduced else 'ctrl'}-k{args.k_fa}",
        config={**vars(args), "reduced": reduced, "box_dim": box_dim},
    )

    # Pre-load the eval envs once; the same problems are reused for every step count.
    problems = []  # (env, sp_t, bx_t, list_of_(start,goal))
    for ei in range(args.env_start, args.env_start + args.n_envs):
        npz = np.load(f"{args.data}/env_{ei:04d}.npz", allow_pickle=True)
        env = build_env(npz, float(npz["robot_radius"]))
        sp_t, bx_t = env_features(npz, norm, box_dim=box_dim)
        starts, goals = npz["starts"], npz["goals"]
        idx = distinct_pairs(starts, goals, args.n_pairs)
        problems.append((
            env, sp_t.to(device), bx_t.to(device),
            [(starts[i], goals[i]) for i in idx],
        ))

    header = (f"{'steps':>6} {'free%':>7} {'clearance':>10} {'ep_err':>8} "
              f"{'length':>8} {'sq_accel':>9} {'disp_ref':>9} {'s/query':>8}"
              + (f" {'residual':>9}" if (args.residual and reduced and args.k_fa > 1)
                 else ""))
    print(f"\nenvs [{args.env_start}, {args.env_start + args.n_envs}) x "
          f"{args.n_pairs} pairs x {args.n_samples} samples, "
          f"seed-matched (anchor={args.anchor_endpoints})")
    print(header)

    ref_paths = None  # 100-step (first entry) trajectories, for displacement
    rows = []
    for n_steps in args.steps:
        free, clear, ep, length, acc = [], [], [], [], []
        paths_all, residuals = [], []
        #The residual is identically 0 at K=1 (nothing to disagree with), so
        #only ask for it when there are multiple frames to compare.
        want_residual = args.residual and reduced and args.k_fa > 1
        t0 = time.time()
        n_queries = 0
        for env, sp_t, bx_t, pairs in problems:
            for start, goal in pairs:
                B = args.n_samples
                s_b = torch.from_numpy(norm.norm_pts(start).astype(np.float32)).repeat(B, 1).to(device)
                g_b = torch.from_numpy(norm.norm_pts(goal).astype(np.float32)).repeat(B, 1).to(device)
                sp_b = sp_t.unsqueeze(0).expand(B, -1, -1)
                bx_b = bx_t.unsqueeze(0).expand(B, -1, -1)
                # Fresh generator per query: identical prior draw for every step count.
                gen = torch.Generator(device=device).manual_seed(
                    args.seed * 100003 + n_queries
                )
                if reduced:
                    out = sample_reduced(
                        model, sp_b, bx_b, s_b, g_b, k_fa=args.k_fa,
                        n_waypoints=N, n_steps=n_steps,
                        anchor_endpoints=args.anchor_endpoints,
                        device=device, generator=gen,
                        return_residual=want_residual,
                    )
                    if want_residual:
                        x, res = out
                        residuals.extend(res)
                    else:
                        x = out
                else:
                    x = sample(
                        model, sp_b, bx_b, torch.cat([s_b, g_b], dim=-1),
                        anchor_start=s_b, anchor_goal=g_b,
                        n_waypoints=N, n_steps=n_steps,
                        anchor_endpoints=args.anchor_endpoints,
                        device=device, generator=gen,
                    )
                paths = norm.denorm_pts(x.cpu().numpy())
                paths_all.append(paths)
                free.extend(env.path_free(p) for p in paths)
                clear.extend(min_clearance(env, p) for p in paths)
                ep.extend(
                    0.5 * (np.linalg.norm(p[0] - start) + np.linalg.norm(p[-1] - goal))
                    for p in paths
                )
                length.extend(path_length(p) for p in paths)
                acc.extend(mean_sq_accel(p) for p in paths)
                n_queries += 1
        secs = (time.time() - t0) / n_queries

        paths_all = np.stack(paths_all)  # (Q, B, N, 3)
        if ref_paths is None:
            ref_paths = paths_all
            disp = 0.0
        else:
            disp = float(np.linalg.norm(paths_all - ref_paths, axis=-1).mean())

        row = dict(
            steps=n_steps, free=100 * float(np.mean(free)),
            clearance=float(np.mean(clear)), ep_err=float(np.mean(ep)),
            length=float(np.mean(length)), sq_accel=float(np.mean(acc)),
            disp_ref=disp, s_per_query=secs,
        )
        if want_residual:
            row["residual"] = float(np.mean(residuals))
        rows.append(row)
        print(f"{n_steps:>6} {row['free']:>6.1f}% {row['clearance']:>10.4f} "
              f"{row['ep_err']:>8.4f} {row['length']:>8.4f} {row['sq_accel']:>9.5f} "
              f"{row['disp_ref']:>9.4f} {row['s_per_query']:>8.3f}"
              + (f" {row['residual']:>9.4f}" if want_residual else ""))
        run.log({f"sweep/{k}": v for k, v in row.items()}, step=n_steps)

    ref = rows[0]
    ok = [r for r in rows if r["free"] >= ref["free"] - 2.0]
    budget = min(ok, key=lambda r: r["steps"])
    print(f"\nreference: {ref['steps']} steps at {ref['free']:.1f}% free")
    print(f"budget: {budget['steps']} steps holds within 2% "
          f"({budget['free']:.1f}% free, {budget['s_per_query']:.3f}s/query, "
          f"{ref['s_per_query']/max(budget['s_per_query'],1e-9):.1f}x faster)")
    run.summary(budget_steps=budget["steps"], budget_free=budget["free"])
    run.finish()


if __name__ == "__main__":
    main()
