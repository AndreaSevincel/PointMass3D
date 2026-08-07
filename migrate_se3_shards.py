
  #.venv/bin/python migrate_se3_shards.py data_se3            # report only
  #.venv/bin/python migrate_se3_shards.py data_se3 --apply    # rewrite in place

  #Repair SE(3) shards written before 2026-08-07.

  #Those shards store starts/goals with one row per PAIR and omit env_seed.
  #flowmatch.data assumes one row per TRAJECTORY (it reads n_trajs from
  #starts.shape[0], and pair_groups() builds the validation split from them),
  #so an unmigrated shard is either unloadable or silently mis-grouped.

  #The repair is exact and needs no replanning: pair_id already records which
  #pair every trajectory came from, so the per-trajectory rows are just the
  #per-pair rows gathered by pair_id. env_seed is recoverable from the
  #generator's formula, seed * 100003 + index.

import argparse
import shutil
from pathlib import Path

import numpy as np


def needs_migration(d):
    #A migrated shard has one starts row per trajectory.
    return d["starts"].shape[0] != d["trajs"].shape[0]


def migrate(path, base_seed, apply, backup):
    d = np.load(path, allow_pickle=True)
    keys = set(d.files)
    idx = int(Path(path).stem.split("_")[1])

    fixed = {k: d[k] for k in d.files}
    changed = []

    if "pair_id" not in keys:
        return "no pair_id -- cannot migrate", False

    if needs_migration(d):
        pid = d["pair_id"]
        n_pairs = d["starts"].shape[0]
        if pid.max() >= n_pairs:
            return f"pair_id max {pid.max()} >= {n_pairs} starts -- inconsistent", False
        fixed["starts"] = d["starts"][pid]
        fixed["goals"] = d["goals"][pid]
        changed.append(f"starts/goals {n_pairs} -> {len(pid)} rows")

    if "env_seed" not in keys:
        fixed["env_seed"] = np.int64(base_seed * 100003 + idx)
        changed.append("env_seed added")

    if not changed:
        return "already current", False
    if not apply:
        return "; ".join(changed) + "  (dry run)", True

    if backup:
        shutil.copy2(path, str(path) + ".bak")
    tmp = str(path) + ".tmp.npz"
    np.savez_compressed(tmp, **fixed)
    Path(tmp).replace(path)
    return "; ".join(changed), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--seed", type=int, default=0,
                    help="the --seed the generator ran with (for env_seed)")
    ap.add_argument("--apply", action="store_true",
                    help="rewrite the shards; without it, only report")
    ap.add_argument("--backup", action="store_true",
                    help="keep a .bak beside each rewritten shard")
    args = ap.parse_args()

    files = sorted(Path(args.data_dir).glob("env_*.npz"))
    if not files:
        raise SystemExit(f"no env_*.npz in {args.data_dir}")

    n_changed = 0
    for f in files:
        msg, did = migrate(f, args.seed, args.apply, args.backup)
        n_changed += did
        if did or "cannot" in msg or "inconsistent" in msg:
            print(f"  {f.name}: {msg}")
    verb = "migrated" if args.apply else "would migrate"
    print(f"\n{verb} {n_changed}/{len(files)} shards"
          + ("" if args.apply else "   (re-run with --apply)"))

    #index gaps break the train/test split, which is addressed by index
    idxs = sorted(int(f.stem.split("_")[1]) for f in files)
    gaps = [i for i in range(idxs[0], idxs[-1] + 1) if i not in set(idxs)]
    if gaps:
        print(f"NOTE: {len(gaps)} index gap(s) present: {gaps[:10]}"
              f"{' ...' if len(gaps) > 10 else ''}")
        print("      Re-run the generator (it resumes) before training: the "
              "held-out range is addressed by index.")


if __name__ == "__main__":
    main()
