
  #.venv/bin/python -m se3body.generate --n-envs 300 --n-pairs 600 --n-trajs 5 \
  #    --workers 12 --out data_se3

  #Dataset generation for the SE(3) rigid-body domain.

  #Shard layout matches the point-mass domain so the same loader, obstacle
  #encoder and eval harness work unchanged: spheres (S,4) and boxes (B,12) per
  #environment, plus starts/goals/trajs. The only difference is that a state is
  #9 numbers (position + 6D rotation) rather than 3.

  #Defaults follow the finding from the point-mass dataset that trajectories
  #per pair are the wrong place to spend generation time: 30 near-duplicate
  #paths per pair carried about as much information as 5, while environments
  #were the binding constraint on held-out performance. So n_trajs defaults to
  #5 here, not 30.

import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from pointmass3d import make_random_env
from se3body.body import RigidBody, SE3Env
from se3body.planner import plan_se3, rrt_connect_se3
from se3body.rotation import matrix_to_6d


def _scene_arrays(env):
    #(S,4) spheres and (B,12) oriented boxes, the format env_features expects
    from pointmass3d import BoxObstacle, SphereObstacle
    sph = [o for o in env.obstacles if isinstance(o, SphereObstacle)]
    box = [o for o in env.obstacles if isinstance(o, BoxObstacle)]
    spheres = np.array([[*o.center, o.radius] for o in sph], dtype=np.float64)
    #centre + half-extents, exactly as the point-mass generator writes them,
    #so flowmatch.data loads these shards with no special case: it re-encodes
    #to the 12-dim oriented form itself
    boxes = np.array([[*o.center, *o.half_extents] for o in box], dtype=np.float64)
    return spheres, boxes


def generate_env(task):
    (idx, n_pairs, n_trajs, n_waypoints, n_spheres, n_boxes, min_dist,
     seed, out_dir, timeout, shortcut_iters, fresh_tree) = task
    path = Path(out_dir) / f"env_{idx:04d}.npz"
    if path.exists():
        return idx, "skip", 0, 0.0

    t0 = time.time()
    rng = np.random.default_rng(seed)
    base = make_random_env(n_spheres=n_spheres, n_boxes=n_boxes, seed=seed)
    env = SE3Env(base, RigidBody())
    spheres, boxes = _scene_arrays(base)

    starts, goals, trajs, pair_ids = [], [], [], []
    attempts, solved = 0, 0
    while solved < n_pairs and attempts < n_pairs * 12:
        attempts += 1
        try:
            ps, Rs = env.sample_free_pose(rng)
            pg, Rg = env.sample_free_pose(rng)
        except RuntimeError:
            continue
        if np.linalg.norm(pg - ps) < min_dist:
            continue

        #One tree per pair, then n_trajs randomised shortcut variants of it.
        #A fresh RRT per trajectory costs n_trajs times as much and buys little:
        #the point-mass dataset showed within-pair paths are near-duplicates
        #regardless (std 0.054 within pair vs 0.426 overall), so the expensive
        #source of diversity was already not paying for itself. --fresh-tree
        #restores the old behaviour.
        raw = None
        if not fresh_tree:
            rpos, rrot, rinfo = rrt_connect_se3(env, (ps, Rs), (pg, Rg),
                                                rng=rng, timeout=timeout)
            if rpos is None:
                continue
            raw = (rpos, rrot)

        got = []
        for _ in range(n_trajs):
            pos, rot, info = plan_se3(env, (ps, Rs), (pg, Rg),
                                      n_waypoints=n_waypoints, rng=rng,
                                      timeout=timeout, shortcut_iters=shortcut_iters,
                                      raw=raw)
            if pos is None or not info["success"]:
                continue
            got.append(np.concatenate([pos, matrix_to_6d(rot)], axis=-1))
        if not got:
            continue

        #starts/goals carry ONE ROW PER TRAJECTORY, not per pair: that is the
        #layout flowmatch.data assumes (n_trajs = starts.shape[0], and
        #pair_groups() derives the validation grouping from them). Writing one
        #row per pair makes the shard unloadable.
        pid = solved
        s_row = np.concatenate([ps, matrix_to_6d(Rs)])
        g_row = np.concatenate([pg, matrix_to_6d(Rg)])
        for g in got:
            trajs.append(g)
            pair_ids.append(pid)
            starts.append(s_row)
            goals.append(g_row)
        solved += 1

    if solved == 0:
        return idx, "empty", 0, time.time() - t0

    #starts/goals are per PAIR; trajs carry pair_id so the loader can build the
    #per-pair validation split that the point-mass domain needs
    np.savez_compressed(
        path,
        spheres=spheres.astype(np.float32),
        boxes=boxes.astype(np.float32),
        starts=np.array(starts, dtype=np.float32),
        goals=np.array(goals, dtype=np.float32),
        trajs=np.array(trajs, dtype=np.float32),
        pair_id=np.array(pair_ids, dtype=np.int32),
        robot_radius=np.float32(env.body.radius),
        env_seed=np.int64(seed),
        body_centers=env.body.centers.astype(np.float32),
        state_dim=np.int32(9),
    )
    return idx, "ok", solved, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-envs", type=int, default=300)
    ap.add_argument("--n-pairs", type=int, default=600)
    ap.add_argument("--n-trajs", type=int, default=5,
                    help="paths per pair; 5 rather than 30 because "
                         "environments, not duplicate paths, bound performance")
    ap.add_argument("--n-waypoints", type=int, default=64)
    ap.add_argument("--n-spheres", type=int, default=20)
    ap.add_argument("--n-boxes", type=int, default=20)
    ap.add_argument("--min-dist", type=float, default=1.5)
    ap.add_argument("--timeout", type=float, default=20.0,
                    help="per-plan RRT timeout in seconds")
    ap.add_argument("--shortcut-iters", type=int, default=200,
                    help="max randomised shortcut proposals per path; it stops "
                         "early once they stop succeeding")
    ap.add_argument("--fresh-tree", action="store_true",
                    help="run a full RRT per trajectory instead of reusing one "
                         "tree per pair. n_trajs times slower, marginally more "
                         "diverse")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="data_se3")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    tasks = [
        (i, args.n_pairs, args.n_trajs, args.n_waypoints, args.n_spheres,
         args.n_boxes, args.min_dist, args.seed * 100003 + i, args.out,
         args.timeout, args.shortcut_iters, args.fresh_tree)
        for i in range(args.n_envs)
    ]
    print(f"{len(tasks)} environments x {args.n_pairs} pairs x {args.n_trajs} "
          f"paths -> {args.out}  ({args.workers} workers)")
    print("NOTE: run with OMP_NUM_THREADS=1, or the workers oversubscribe the "
          "cores and this gets several times slower.\n")

    t0 = time.time()
    done = ok = 0
    with Pool(args.workers) as pool:
        for idx, status, solved, secs in pool.imap_unordered(generate_env, tasks):
            done += 1
            ok += status == "ok"
            if done % 5 == 0 or status != "ok":
                el = time.time() - t0
                print(f"  {done}/{len(tasks)}  env {idx} {status} "
                      f"({solved} pairs, {secs:.0f}s)  "
                      f"~{el/done*(len(tasks)-done)/60:.0f} min left", flush=True)
    print(f"\n{ok}/{len(tasks)} shards written in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
