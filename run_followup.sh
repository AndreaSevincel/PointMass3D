#!/bin/bash
# Runs AFTER the convergence study, unattended.
#
#   mkdir -p logs
#   setsid nohup ./run_followup.sh > logs/followup.log 2>&1 < /dev/null &
#
# Safe to start while run_convergence.sh is still going: this waits for those
# cells to clear before touching a GPU. It is a SEPARATE file on purpose --
# editing run_convergence.sh while bash is executing it corrupts the run,
# because bash reads a script by byte offset as it goes.
#
# Sequence:
#   1. wait for the convergence cells to exit
#   2. score their snapshots and print the curve
#   3. train the DDPM arms, one arm per GPU
#   4. score DDPM across the NFE sweep
#
# Step 3 is the generative-model control: same backbone, same conditioning, same
# data, only the objective differs. If the (s,g) reduction pays off here too,
# the "this is a flow-matching artifact" objection is answered.

set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"
mkdir -p logs results

PY=.venv/bin/python
CONV_PAT="train_flow.py.*conv-s0"

say() { echo "[$(date -Is)] $*"; }

# --- 1. wait -----------------------------------------------------------------
if pgrep -f "$CONV_PAT" > /dev/null; then
  say "convergence run detected; waiting for it to finish"
  while pgrep -f "$CONV_PAT" > /dev/null; do sleep 300; done
  say "convergence cells have exited"
else
  say "WARNING: no convergence cells running; assuming the GPUs are free"
fi

# A cell that died leaves the GPU free, so we proceed either way -- but say so,
# because the curve in step 2 will then be shorter than 300 epochs.
say "snapshots present: $(ls checkpoints/conv/*.ep*.pt 2>/dev/null | wc -l)"

# --- 2. score the convergence curve ------------------------------------------
say "scoring convergence snapshots"
$PY run_convergence.py score  >> logs/followup-score.log 2>&1
$PY run_convergence.py table  >> logs/followup-score.log 2>&1
say "curve written to results/conv/curve.json"

# --- 3. DDPM baselines, one arm per GPU --------------------------------------
# 60 environments is the only cell with seed-matched flow numbers to compare
# against (ctrl 13.57+-0.05, treat 34.43+-1.56), so a DDPM gap here is directly
# commensurable. ~2.1 h per run, three seeds per arm, two arms in parallel.
say "training DDPM arms"
CUDA_VISIBLE_DEVICES=0 $PY run_grid.py --data data --arms control \
    --objectives ddpm --n-envs 60 --seeds 0 1 2 --epochs 20 --batch 1024 \
    --amp --group ddpm-e60 --continue-on-error >> logs/ddpm-ctrl.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 $PY run_grid.py --data data --arms treatment \
    --objectives ddpm --n-envs 60 --seeds 0 1 2 --epochs 20 --batch 1024 \
    --amp --group ddpm-e60 --continue-on-error >> logs/ddpm-treat.log 2>&1 &
P1=$!
wait $P0; R0=$?
wait $P1; R1=$?
say "DDPM training done (control=$R0 treatment=$R1)"

# --- 4. score DDPM across the NFE sweep --------------------------------------
# NOT 8 steps alone. The 8-step operating point was chosen because straight-line
# interpolants tolerate it; DDPM generally does not, and scoring it only there
# would floor both arms and answer nothing. sweep_steps matches NFE across
# objectives, so every column is a fair comparison.
for ck in checkpoints/grid/ddpm-*.pt; do
  [ -e "$ck" ] || continue
  tag=$(basename "$ck" .pt)
  [ -e "results/$tag.json" ] && { say "skip $tag (already scored)"; continue; }
  say "scoring $tag"
  $PY sweep_steps.py --ckpt "$ck" --data data \
      --env-start 250 --n-envs 50 --n-pairs 10 --n-samples 20 \
      --steps 100 50 20 10 8 --out-json "results/$tag.json" \
      >> logs/ddpm-score.log 2>&1 || say "FAILED to score $tag"
done

# --- 5. seeds 1 and 2 at the scales that currently have only one -------------
# The headline gap (+30.2 at 250 envs) rests on one seed per arm, and the
# scaling figure has error bands at 60 environments only. This is the single
# highest-return experiment left: it converts the headline from a point estimate
# into a tested difference, and it is the first thing a reviewer looks for.
#
# Cheapest scale first, so a failure surfaces in an hour rather than after the
# 250-environment cells have burned most of a day. Each cell is independent, so
# a crash costs that cell and nothing else.
for n in 20 150 250; do
  say "seeds 1,2 at $n environments"
  CUDA_VISIBLE_DEVICES=0 $PY run_grid.py --data data --arms control \
      --n-envs "$n" --seeds 1 2 --epochs 20 --batch 1024 --amp \
      --group seeds-e"$n" --continue-on-error >> logs/seeds-ctrl.log 2>&1 &
  Q0=$!
  CUDA_VISIBLE_DEVICES=1 $PY run_grid.py --data data --arms treatment \
      --n-envs "$n" --seeds 1 2 --epochs 20 --batch 1024 --amp \
      --group seeds-e"$n" --continue-on-error >> logs/seeds-treat.log 2>&1 &
  Q1=$!
  wait $Q0; wait $Q1
  say "scoring the new $n-environment seeds"
  for ck in checkpoints/grid/ctrl-e"$n"-s[12].pt checkpoints/grid/treat-e"$n"-s[12].pt; do
    [ -e "$ck" ] || continue
    tag=$(basename "$ck" .pt)
    [ -e "results/$tag.json" ] && continue
    $PY sweep_steps.py --ckpt "$ck" --data data \
        --env-start 250 --n-envs 50 --n-pairs 10 --n-samples 20 \
        --steps 8 --out-json "results/$tag.json" \
        >> logs/seeds-score.log 2>&1 || say "FAILED to score $tag"
  done
done

say "all done. Read the comparison with:"
say "  $PY aggregate_runs.py results/*.json --metric free --steps 8"
say "  $PY aggregate_runs.py results/*.json --metric free --steps 100"
say "  $PY aggregate_runs.py results/*.json --metric solved_any --steps 8"
