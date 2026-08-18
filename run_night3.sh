#!/bin/bash
# Overnight watcher for the equivariant diagnostics.
#
#   mkdir -p logs && setsid nohup ./run_night3.sh > logs/night3.log 2>&1 < /dev/null &
#
# Two runs are training: equiv2 (the v2 backbone -- attention pooling and
# vec_rms_norm) and equivgeo (v2 plus oracle SDF geometry, split by irrep).
# Both write a snapshot every 10 epochs. This scores each one as it appears, so
# the morning has a curve rather than a single endpoint, and then spends the
# freed GPU on the data-scaling question.
#
# The three numbers to compare against, all at 60 environments:
#   v1 equivariant backbone   22.2  (epoch 60; it overfits from epoch 40)
#   unconstrained reduced     42.6  (epoch 60) -> 50.6 (epoch 300)
#   world-frame control       15.0  (converged)

set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"
mkdir -p logs results/conv checkpoints/conv

PY=.venv/bin/python
#Arms to watch, as arguments: ./run_night3.sh treatgeo-e60-conv-s0 equiv2-e60-conv-s1
#NOT ("${@:-a b}") -- that expands the default as ONE word, so the loop would
#watch a single arm named "a b" and silently score nothing.
ARMS=("$@")
[ ${#ARMS[@]} -eq 0 ] && ARMS=(equiv2-e60-conv-s0 equivgeo-e60-conv-s0)
say() { echo "[$(date -Is)] $*"; }

# Match the TRAINER specifically. Matching the bare run name would also match
# this script's own sweep_steps.py invocations -- their argv carries the
# checkpoint path -- and the wait loop would then never terminate.
training() { pgrep -f "train_flow.py.*$1" > /dev/null; }

score_new() {
  local arm="$1" ck tag n=0
  for ck in checkpoints/conv/"$arm".ep*.pt; do
    [ -e "$ck" ] || return 0                      # nothing yet; not an error
    tag="$(basename "$ck" .pt)"
    [ -e "results/conv/$tag.json" ] && continue
    say "scoring $tag"
    $PY sweep_steps.py --ckpt "$ck" --data data --env-start 250 --n-envs 50 \
        --n-pairs 10 --n-samples 20 --steps 8 \
        --out-json "results/conv/$tag.json" >> logs/night3-score.log 2>&1 \
      || { say "FAILED scoring $tag"; rm -f "results/conv/$tag.json"; }
    n=$((n + 1))
  done
  [ "$n" -gt 0 ] && say "$arm: scored $n new"
  return 0
}

any_training() {
  local a
  for a in "${ARMS[@]}"; do training "$a" && return 0; done
  return 1
}

#Wait for the trainers to appear before deciding they are finished. Launching
#the watcher and the trainer in the same shell block is a race the watcher can
#win: pgrep finds nothing, any_training is false on the first evaluation, and
#the loop falls straight through to the final pass and exits -- leaving a real
#training run with nothing scoring its snapshots. Cost of being wrong here is
#one wasted overnight, so wait generously.
say "waiting for ${ARMS[*]} to start"
for _ in $(seq 60); do
  any_training && break
  sleep 10
done
any_training || { say "ERROR: none of ${ARMS[*]} started within 10 min"; exit 1; }

say "watching ${ARMS[*]}"
while any_training; do
  for a in "${ARMS[@]}"; do score_new "$a"; done
  sleep 600
done
say "training finished; final scoring pass"
for a in "${ARMS[@]}"; do score_new "$a"; done
exit 0

# Both GPUs are now free. The val-loss curve says the v1 backbone stopped
# improving at epoch 40 and got WORSE by 70, while the unconstrained arm
# descended monotonically for 280 -- overfitting, at 60 environments, from the
# model carrying the symmetry prior. The direct test of that is more data.
# 250 envs is 4.2x the trajectories, so 80 epochs here is ~700k steps, slightly
# more than the 60-env run's full 300.
say "launching the 250-environment arm"
CUDA_VISIBLE_DEVICES=0 nohup $PY train_flow.py --data data --n-envs 250 --env-start 0 \
    --out checkpoints/conv/equiv2-e250-conv-s0.pt --epochs 80 --batch 1024 \
    --reduced --equivariant --seed 0 --num-workers 8 --split-by pair \
    --snapshot-every 10 --amp --wandb --wandb-group equivariant \
    --wandb-name equiv2-e250-conv-s0 >> logs/equiv2-e250.log 2>&1 &

sleep 60
while training "equiv2-e250"; do
  score_new equiv2-e250-conv-s0
  sleep 600
done
score_new equiv2-e250-conv-s0
say "all done"
