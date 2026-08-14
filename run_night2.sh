#!/bin/bash
# Error bars on the convergence claims, and DDPM at a budget that is not
# self-evidently inadequate.
#
#   mkdir -p logs && setsid nohup ./run_night2.sh > logs/night2.log 2>&1 < /dev/null &
#
# Why these two:
#
#   1. Seeds for the 60-environment convergence study. The paper's strongest
#      claims are now the converged ones -- the control plateauing at the
#      straight-line floor, the gap widening to 35.8, and frame averaging's
#      benefit growing ninefold with model quality -- and every one of them
#      rests on a single seed. The 20-epoch grid has three seeds at this scale,
#      so the convergence cells should too, or the strongest claims in the paper
#      are the least supported ones.
#
#      250 environments would be better still and is not affordable: ~15 h per
#      cell, so four cells is 60 GPU-hours against 14 here.
#
#   2. DDPM trained to 300 epochs. Sec. "objective" reports the flow/DDPM
#      comparison at 20 epochs and says plainly that this is a common but
#      inadequate budget. Running the same two arms to convergence answers
#      whether the objective-generality result survives at a budget where the
#      undertraining objection does not apply.
#
# Order matters: the seeds go first because they close a hole in claims the
# paper already makes, and DDPM extends one it makes with a caveat attached.

set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"
mkdir -p logs results

PY=.venv/bin/python
say() { echo "[$(date -Is)] $*"; }

# --- 1. convergence seeds 1 and 2, both arms, 60 environments ----------------
# One arm per GPU, seeds sequential within an arm: 2 x 3.5 h per lane.
say "convergence seeds at 60 environments (4 cells, ~7 h)"
for s in 1 2; do
  CUDA_VISIBLE_DEVICES=0 $PY train_flow.py --data data --n-envs 60 --env-start 0 \
      --out "checkpoints/conv/ctrl-e60-conv-s${s}.pt" --epochs 300 --batch 1024 \
      --seed "$s" --num-workers 8 --split-by pair --snapshot-every 10 --resume \
      --amp --wandb --wandb-group convergence --wandb-name "ctrl-e60-conv-s${s}" \
      >> "logs/ctrl-e60-conv-s${s}.log" 2>&1 &
  P0=$!
  CUDA_VISIBLE_DEVICES=1 $PY train_flow.py --data data --n-envs 60 --env-start 0 \
      --out "checkpoints/conv/treat-e60-conv-s${s}.pt" --epochs 300 --batch 1024 \
      --reduced --seed "$s" --num-workers 8 --split-by pair --snapshot-every 10 \
      --resume --amp --wandb --wandb-group convergence \
      --wandb-name "treat-e60-conv-s${s}" \
      >> "logs/treat-e60-conv-s${s}.log" 2>&1 &
  P1=$!
  wait $P0; wait $P1
  say "seed $s done"
done

say "scoring the convergence snapshots (K=1 both arms, K=3 on the reduced arm)"
$PY run_convergence.py score --n-envs 60 --seeds 0 1 2 >> logs/night2-score.log 2>&1
$PY run_convergence.py table --n-envs 60 --seeds 0 1 2 >> logs/night2-score.log 2>&1

# --- 2. DDPM at 300 epochs --------------------------------------------------
say "DDPM to convergence at 60 environments (2 cells, ~3.5 h)"
# NOT run_grid.py here: it derives the checkpoint name from arm/envs/objective,
# so a 300-epoch run would overwrite checkpoints/grid/ddpm-ctrl-e60.pt -- the
# 20-epoch checkpoint the paper's +16.71 result was measured on. The JSON would
# survive but the model behind a published number would not. Call train_flow.py
# directly with an explicit -conv suffix instead.
CUDA_VISIBLE_DEVICES=0 $PY train_flow.py --data data --n-envs 60 --env-start 0 \
    --out checkpoints/grid/ddpm-ctrl-e60-conv.pt --objective ddpm \
    --epochs 300 --batch 1024 --seed 0 --num-workers 8 --split-by pair --amp \
    --wandb --wandb-group ddpm-conv --wandb-name ddpm-ctrl-e60-conv \
    >> logs/ddpm-conv-ctrl.log 2>&1 &
Q0=$!
CUDA_VISIBLE_DEVICES=1 $PY train_flow.py --data data --n-envs 60 --env-start 0 \
    --out checkpoints/grid/ddpm-treat-e60-conv.pt --objective ddpm --reduced \
    --epochs 300 --batch 1024 --seed 0 --num-workers 8 --split-by pair --amp \
    --wandb --wandb-group ddpm-conv --wandb-name ddpm-treat-e60-conv \
    >> logs/ddpm-conv-treat.log 2>&1 &
Q1=$!
wait $Q0; wait $Q1

# The NFE sweep, not 8 steps alone: DDPM gains ~4 points on both arms between 8
# and 100 evaluations, so scoring it only at the flow arm's operating point
# understates it and invites the fair objection that the baseline was
# handicapped by a budget tuned for the method it is meant to test.
for ck in checkpoints/grid/ddpm-ctrl-e60-conv.pt checkpoints/grid/ddpm-treat-e60-conv.pt; do
  [ -e "$ck" ] || continue
  tag="$(basename "$ck" .pt)"
  [ -e "results/$tag.json" ] && continue
  say "scoring $tag"
  $PY sweep_steps.py --ckpt "$ck" --data data --env-start 250 --n-envs 50 \
      --n-pairs 10 --n-samples 20 --steps 100 50 20 10 8 \
      --out-json "results/$tag.json" >> logs/ddpm-conv-score.log 2>&1 \
      || say "FAILED scoring $tag"
done

say "all done. Read with:"
say "  $PY run_convergence.py table --n-envs 60 --seeds 0 1 2"
say "  $PY aggregate_runs.py results/*.json --metric free --steps 100"
