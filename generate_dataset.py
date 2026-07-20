"""
Usage:
    python generate_dataset.py --n-envs 20 --n-trajs 25 --refine chomp
    If 2D needs 200 distinct environments, 500 valid start-goal pairs with 20 feasible multimodal trajectories per pair, how much would 3D need?
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pointmass3d import (
    BoxObstacle,
    SphereObstacle,
    chomp,
    make_random_env,
    resample_path,
    rrt_connect,
    sample_start_goal,
    shortcut,
    trajopt,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-envs", type=int, default=10)
    ap.add_argument("--n-trajs", type=int, default=20, help="trajectories per environment")
    ap.add_argument("--n-waypoints", type=int, default=64)
    ap.add_argument("--refine", choices=["none", "chomp", "trajopt"], default="chomp")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="data")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    N = args.n_waypoints
    refiner = {"chomp": chomp, "trajopt": trajopt, "none": None}[args.refine]

    t0 = time.time()
    n_ok = n_fallback = n_failed = 0

    for e in range(args.n_envs):
        env_seed = args.seed + e
        env = make_random_env(n_spheres=20, n_boxes=20, seed=env_seed)
        rng = np.random.default_rng(10_000 + env_seed)

        trajs, starts, goals = [], [], []
        attempts = 0
        while len(trajs) < args.n_trajs and attempts < 5 * args.n_trajs:
            attempts += 1
            try:
                start, goal = sample_start_goal(env, rng)
            except RuntimeError:
                break
            raw, _ = rrt_connect(env, start, goal, rng=rng)
            if raw is None:
                n_failed += 1
                continue
            path = resample_path(shortcut(env, raw, rng=rng), N)
            if not env.path_free(path):
                n_failed += 1
                continue

            if refiner is not None:
                refined, info = refiner(env, start, goal, n_waypoints=N, init_path=path)
                if info["success"]:
                    path = refined
                    n_ok += 1
                else:
                    n_fallback += 1  # keep the validated RRT path
            else:
                n_ok += 1

            trajs.append(path)
            starts.append(start)
            goals.append(goal)

        spheres = np.array(
            [[*o.center, o.radius] for o in env.obstacles if isinstance(o, SphereObstacle)]
        )
        boxes = np.array(
            [[*o.center, *o.half_extents] for o in env.obstacles if isinstance(o, BoxObstacle)]
        )
        np.savez(
            out / f"env_{e:04d}.npz",
            spheres=spheres,
            boxes=boxes,
            trajs=np.array(trajs),
            starts=np.array(starts),
            goals=np.array(goals),
            env_seed=env_seed,
            robot_radius=env.robot_radius,
        )
        print(f"env {e:04d}: {len(trajs)}/{args.n_trajs} trajectories")

    meta = {
        "n_envs": args.n_envs,
        "n_trajs_per_env": args.n_trajs,
        "n_waypoints": N,
        "refine": args.refine,
        "seed": args.seed,
        "robot_radius": 0.03,
        "workspace": [-1.0, 1.0],
        "refined_ok": n_ok,
        "rrt_fallback": n_fallback,
        "failed": n_failed,
        "wall_time_s": round(time.time() - t0, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\ndone in {meta['wall_time_s']}s — refined: {n_ok}, "
          f"rrt fallback: {n_fallback}, failed attempts: {n_failed}")
    print(f"dataset written to {out}/")


if __name__ == "__main__":
    main()
