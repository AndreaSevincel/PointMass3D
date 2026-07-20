#python generate_dataset.py --n-envs 20 --n-pairs 25 --n-trajs 20 --refine chomp
#python generate_dataset.py --n-envs 20 --n-pairs 25 --n-trajs 20 --reversible

#If 2D needs 200 distinct environments, 500 valid start-goal pairs with 20 feasible multimodal trajectories per pair, how much would 3D need?

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


def plan_trajectories(env, start, goal, n_trajs, rng, n_waypoints, refiner, counts):
    #plan up to n_trajs trajectories
    trajs = []
    attempts = 0
    while len(trajs) < n_trajs and attempts < 5 * n_trajs:
        attempts += 1
        raw, _ = rrt_connect(env, start, goal, rng=rng)
        if raw is None:
            counts["failed"] += 1
            continue
        path = resample_path(shortcut(env, raw, rng=rng), n_waypoints)
        if not env.path_free(path):
            counts["failed"] += 1
            continue

        if refiner is not None:
            refined, info = refiner(env, start, goal, n_waypoints=n_waypoints, init_path=path)
            if info["success"]:
                path = refined
                counts["ok"] += 1
            else:
                counts["fallback"] += 1#keep the validated RRT path
        else:
            counts["ok"] += 1

        trajs.append(path)
    return trajs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-envs", type=int, default=10, help="n of distinct environments")
    ap.add_argument("--n-pairs", type=int, default=5, help="s-g pairs per environment")
    ap.add_argument("--n-trajs", type=int, default=4, help="trajectories per s-g pair")
    ap.add_argument("--n-waypoints", type=int, default=64)
    ap.add_argument("--refine", choices=["none", "chomp", "trajopt"], default="chomp")
    ap.add_argument(
        "--reversible",
        action="store_true",
        help="for every sampled pair, also plan trajectories for the swapped (g, s) pair",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="data")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    N = args.n_waypoints
    refiner = {"chomp": chomp, "trajopt": trajopt, "none": None}[args.refine]

    t0 = time.time()
    counts = {"ok": 0, "fallback": 0, "failed": 0}

    for e in range(args.n_envs):
        env_seed = args.seed + e
        env = make_random_env(n_spheres=20, n_boxes=20, seed=env_seed)
        rng = np.random.default_rng(10_000 + env_seed)

        trajs, starts, goals, pair_ids, reversed_flags = [], [], [], [], []
        n_pairs_done = 0
        pair_attempts = 0
        while n_pairs_done < args.n_pairs and pair_attempts < 5 * args.n_pairs:
            pair_attempts += 1
            try:
                start, goal = sample_start_goal(env, rng)
            except RuntimeError:
                break

            directions = [(start, goal, False)]
            if args.reversible:
                directions.append((goal, start, True))

            for s, g, is_reversed in directions:
                for path in plan_trajectories(env, s, g, args.n_trajs, rng, N, refiner, counts):
                    trajs.append(path)
                    starts.append(s)
                    goals.append(g)
                    pair_ids.append(n_pairs_done)
                    reversed_flags.append(is_reversed)

            n_pairs_done += 1

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
            pair_ids=np.array(pair_ids),
            reversed=np.array(reversed_flags),
            env_seed=env_seed,
            robot_radius=env.robot_radius,
        )
        n_expected = args.n_pairs * args.n_trajs * (2 if args.reversible else 1)
        print(f"env {e:04d}: {len(trajs)}/{n_expected} trajectories "
              f"({n_pairs_done}/{args.n_pairs} pairs)")

    meta = {
        "n_envs": args.n_envs,
        "n_pairs_per_env": args.n_pairs,
        "n_trajs_per_pair": args.n_trajs,
        "reversible": args.reversible,
        "n_waypoints": N,
        "refine": args.refine,
        "seed": args.seed,
        "robot_radius": 0.03,
        "workspace": [-1.0, 1.0],
        "refined_ok": counts["ok"],
        "rrt_fallback": counts["fallback"],
        "failed": counts["failed"],
        "wall_time_s": round(time.time() - t0, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\ndone in {meta['wall_time_s']}s — refined: {counts['ok']}, "
          f"rrt fallback: {counts['fallback']}, failed attempts: {counts['failed']}")
    print(f"dataset written to {out}/")


if __name__ == "__main__":
    main()
