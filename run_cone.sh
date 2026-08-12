#!/bin/bash
# The cone sweep: does r PREDICT what frame averaging is worth?
#
#   mkdir -p logs && ./run_cone.sh
#
# Sec. "twosigns" shows r does not forecast the benefit of a symmetry mechanism,
# but it shows it at THREE points, two of which have nearly identical r (0.0181
# vs 0.0174). The fair objection is that nothing varied: of course r fails to
# discriminate when every r is small. This sweep generates the variation.
#
# Mechanism: r is small here because the gauge is a deterministic function of
# x-hat and ~600 start-goal directions per environment already span many rolls.
# Restricting training to a narrow cone of directions starves the model of that
# diversity, which should drive r UP. If r is predictive, frame averaging should
# start to pay in proportion; if it does not pay even at large r, that is a much
# harder negative result. Either outcome turns one measurement into a curve.
#
# --subsample is not optional. A 15-degree cone keeps a few percent of the
# trajectories, so an unmatched sweep would confound roll diversity with
# training-set SIZE -- and size is the variable the whole paper is about.

set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"
mkdir -p logs results

PY=.venv/bin/python
ENVS=60
CONES="15 30 45 90"
#Gradient steps to give every arm. The main 60-environment cells see
#2.16M/1024 x 20 ~= 42k steps; matching that is what makes a cone arm
#comparable to them rather than merely to each other.
TARGET_STEPS=40000
BATCH=1024

say() { echo "[$(date -Is)] $*"; }

# --- how many trajectories survive the narrowest cone? -----------------------
# Measured rather than assumed: the start-goal direction distribution is NOT
# uniform on the sphere (a box workspace with a minimum separation biases it),
# so the solid-angle estimate 1-cos(theta) is the wrong number to size against.
say "counting trajectories per cone"
CAP=$($PY - "$ENVS" "$CONES" <<'EOF'
import sys, numpy as np
n_envs, cones = int(sys.argv[1]), [float(c) for c in sys.argv[2].split()]
counts = {c: 0 for c in cones}
for ei in range(n_envs):
    z = np.load(f"data/env_{ei:04d}.npz", allow_pickle=True)
    d = z["goals"] - z["starts"]
    xh = np.abs(d[:, 0]) / np.linalg.norm(d, axis=-1)   # |.|: a path and its
    for c in cones:                                     # reverse are one cone
        #starts/goals are stored PER TRAJECTORY, not per pair -- which is
        #why distinct_pairs() exists -- so this count is already the
        #trajectory count and must not be scaled by 30x2.
        counts[c] += int((xh >= np.cos(np.deg2rad(c))).sum())
for c in cones:
    print(f"  cone {c:>5.1f} deg: {counts[c]:>10,d} trajectories", file=sys.stderr)
print(min(counts.values()))
EOF
)
#--subsample equalises the DATASET SIZE, so a fixed epoch count equalises
#gradient steps automatically. But the narrowest cone keeps ~2% of the data, so
#20 epochs would be ~900 steps against the ~42k the main runs get: every arm
#would be untrained, and the frame-averaging effect being measured (~0.5 points
#on a converged model) would sit far inside noise. Scale epochs to the cap so
#the sweep is matched in size AND in optimisation.
EPOCHS=$(( TARGET_STEPS * BATCH / CAP ))
say "capping every arm at $CAP trajectories ($EPOCHS epochs ~= $TARGET_STEPS steps)"
say "only the cone width varies: size, steps, seed and architecture are fixed"

for c in $CONES; do
  tag="treat-e${ENVS}-cone${c%.*}"
  say "cone ${c} deg"
  $PY train_flow.py --data data --n-envs "$ENVS" --env-start 0 \
      --out "checkpoints/grid/${tag}.pt" --epochs "$EPOCHS" --batch 1024 \
      --reduced --xhat-cone "$c" --subsample "$CAP" \
      --num-workers 8 --split-by pair --amp \
      --wandb --wandb-group cone-sweep --wandb-name "$tag" \
      >> "logs/${tag}.log" 2>&1 || { say "FAILED $tag"; continue; }

  # K=1 and K=3 on the same checkpoint: the benefit of frame averaging is the
  # PAIRED difference, and --residual reports r on the same pass, so each cone
  # width yields one (r, benefit) point with no seed noise in the difference.
  for k in 1 3; do
    out="results/${tag}$([ $k -gt 1 ] && echo "-kfa$k").json"
    [ -e "$out" ] && continue
    extra=""; [ $k -gt 1 ] && extra="--k-fa $k --residual"
    $PY sweep_steps.py --ckpt "checkpoints/grid/${tag}.pt" --data data \
        --env-start 250 --n-envs 50 --n-pairs 10 --n-samples 20 --steps 8 \
        $extra --out-json "$out" >> "logs/${tag}-score.log" 2>&1 \
        || say "FAILED scoring $tag K=$k"
  done
done

say "done. The curve is (r, K=3 minus K=1) across cone widths:"
for c in $CONES; do
  tag="treat-e${ENVS}-cone${c%.*}"
  $PY - "$tag" "$c" <<'EOF'
import json, sys, pathlib
tag, c = sys.argv[1], sys.argv[2]
try:
    a = json.loads(pathlib.Path(f"results/{tag}.json").read_text())["rows"][0]
    b = json.loads(pathlib.Path(f"results/{tag}-kfa3.json").read_text())["rows"][0]
except OSError:
    sys.exit(0)
print(f"  cone {c:>5} deg   r={b.get('residual', float('nan')):.4f}   "
      f"K=1 {a['free']:.2f}   K=3 {b['free']:.2f}   "
      f"delta {b['free'] - a['free']:+.2f}")
EOF
done
