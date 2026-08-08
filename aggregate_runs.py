
  #.venv/bin/python aggregate_runs.py results/*.json
  #.venv/bin/python aggregate_runs.py results/*.json --metric free --steps 8

  #Pool sweep_steps.py JSON outputs into a table with error bars ACROSS SEEDS.

  #Why this exists: the paper's original error bars came from the spread over
  #held-out problems, which answers "would another sample of problems give this
  #number?". It does not answer "would another training run give this number?".
  #Those are different questions and the second is the one a reviewer asks of a
  #claim about a method. With >= 3 seeds per cell this reports both, and they
  #should be quoted separately rather than blended.

  #Filenames are the metadata: <objective->arm-e<envs>[-sN].json, matching the
  #checkpoint names that run_grid.py writes.

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

#Filenames carry the variant and the scale; the ARM and OBJECTIVE are read from
#the JSON payload instead, because sweep_steps.py records them (`reduced`,
#`objective`) and a filename cannot be trusted to. `long-e60` for instance does
#not say which arm it is, and an earlier regex that insisted on a `ctrl`/`treat`
#prefix silently skipped every run whose name carried a variant tag.
#Everything after the env count is parsed as free-form tags rather than a fixed
#sequence, so a new suffix (-se3res, -kfa3, whatever comes next) never silently
#drops a file. An earlier version hard-coded the order and skipped every file
#carrying a tag it had not been told about.
NAME_RE = re.compile(r"^(?P<variant>.*?)e(?P<envs>\d+)(?P<rest>.*)$")
SEED_RE = re.compile(r"^s(\d+)$")
KFA_RE = re.compile(r"^kfa(\d+)$")


def parse_name(path, payload=None):
    stem = Path(path).stem
    m = NAME_RE.match(stem)
    if not m:
        return None
    g = m.groupdict()

    #normalise the variant: strip the arm words and separators, keep the rest
    v = g["variant"] or ""
    for token in ("ddpm-", "flow-", "ctrl-", "treat-", "long-"):
        v = v.replace(token, "")
    v = v.strip("-")

    #trailing tags: -sN is the seed, -kfaN the frame-averaging width, and
    #anything else is part of the variant
    seed, kfa, extra = None, None, []
    for tok in filter(None, (g["rest"] or "").split("-")):
        if SEED_RE.match(tok):
            seed = int(SEED_RE.match(tok).group(1))
        elif KFA_RE.match(tok):
            kfa = int(KFA_RE.match(tok).group(1))
        else:
            extra.append(tok)
    v = "-".join(filter(None, [v] + extra)) or "base"

    arm, objective = "?", "flow"
    if payload is not None:
        arm = "treat" if payload.get("reduced") else "ctrl"
        objective = payload.get("objective", "flow")
    if "long-" in stem:
        v = "long" if v == "base" else f"long-{v}"

    if kfa is None and payload is not None:
        kfa = payload.get("config", {}).get("k_fa", 1)
    return dict(objective=objective, arm=arm, variant=v,
                envs=int(g["envs"]), seed=seed or 0, kfa=kfa or 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--metric", type=str, default="free",
                    help="row key from sweep_steps (free, solved_any, "
                         "clearance, ep_err, length, residual, ...)")
    ap.add_argument("--steps", type=int, default=None,
                    help="integrator budget to read (default: the smallest "
                         "present, which is the reported operating point)")
    args = ap.parse_args()

    cells = defaultdict(dict)  # (objective, arm, envs) -> {seed: value}
    skipped = []
    for f in args.files:
        try:
            with open(f) as fh:
                payload = json.load(fh)
        except OSError:
            skipped.append(f)
            continue
        meta = parse_name(f, payload)
        if meta is None:
            skipped.append(f)
            continue
        rows = {r["steps"]: r for r in payload["rows"]}
        n_steps = args.steps if args.steps is not None else min(rows)
        if n_steps not in rows:
            skipped.append(f"{f} (no steps={n_steps})")
            continue
        row = rows[n_steps]
        if args.metric not in row:
            avail = ", ".join(k for k in row if k != "steps")
            skipped.append(f"{f} (no '{args.metric}'; has: {avail})")
            continue
        key = (meta["objective"], f"{meta['arm']}/{meta['variant']}"
               + (f"/K{meta['kfa']}" if meta["kfa"] != 1 else ""), meta["envs"])
        cells[key][meta["seed"]] = row[args.metric]

    if skipped:
        print(f"skipped {len(skipped)}: {', '.join(map(str, skipped[:4]))}"
              + (" ..." if len(skipped) > 4 else ""))
    if not cells:
        raise SystemExit("nothing to aggregate")

    print(f"\nmetric={args.metric}  "
          f"steps={args.steps if args.steps is not None else 'min per file'}\n")
    header = f"{'objective':<11}{'arm / variant':<26}{'envs':>6}{'seeds':>7}{'mean':>10}{'SE':>8}{'spread':>18}"
    print(header)
    print("-" * len(header))
    for key in sorted(cells, key=lambda k: (k[0], k[1], k[2])):
        obj, arm, envs = key
        by_seed = cells[key]
        v = np.array([by_seed[s] for s in sorted(by_seed)], dtype=float)
        #SE over seeds; with n=1 there is no spread to report and saying 0
        #would be a lie, so print a dash
        se = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
        spread = ", ".join(f"{x:.3g}" for x in v)
        se_s = f"{se:>8.3f}" if len(v) > 1 else f"{'--':>8}"
        print(f"{obj:<11}{arm:<26}{envs:>6}{len(v):>7}{v.mean():>10.3f}{se_s}"
              f"{spread:>18}")

    #the comparison the paper makes, with a seed-aware error bar
    print()
    for obj in sorted({k[0] for k in cells}):
        for envs in sorted({k[2] for k in cells if k[0] == obj}):
            c = cells.get((obj, "ctrl/base", envs))
            t = cells.get((obj, "treat/base", envs))
            if not c or not t:
                continue
            cv = np.array(list(c.values()), dtype=float)
            tv = np.array(list(t.values()), dtype=float)
            gap = tv.mean() - cv.mean()
            if len(cv) > 1 and len(tv) > 1:
                se = float(np.sqrt(cv.var(ddof=1) / len(cv) + tv.var(ddof=1) / len(tv)))
                print(f"{obj}  e{envs}: treat - ctrl = {gap:+.2f} "
                      f"+- {se:.2f} (across seeds, n={len(cv)}/{len(tv)})")
            else:
                print(f"{obj}  e{envs}: treat - ctrl = {gap:+.2f} "
                      f"(single seed -- no across-seed error bar)")


if __name__ == "__main__":
    main()
