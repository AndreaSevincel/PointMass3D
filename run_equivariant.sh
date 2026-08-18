#!/bin/bash
# The comparison the paper explicitly could not make.
#
#   mkdir -p logs && setsid nohup ./run_equivariant.sh > logs/equiv.log 2>&1 < /dev/null &
#
# Three ways exist to impose a symmetry: in the data (roll augmentation), in the
# operator (frame averaging), or in the weights. The paper measures the first
# two at +0.8 and +0.68 and says plainly that the third is untested -- and that
# Corollary "budget" cannot stand in for it, because it bounds POST-HOC
# projection of a given field, not a model searching a different hypothesis
# class. This runs the third.
#
# The comparison, at 60 environments and 300 epochs, against a converged
# reduced arm at 51.10 +- 0.25:
#
#   * beats it by more than frame averaging's +0.68  -> a constrained
#     architecture does something projection cannot, and the paper's caution
#     about not extrapolating from r was warranted.
#   * matches it                                     -> the residual roll really
#     is exhausted, and all three mechanisms agree.
#   * loses to it                                    -> the constraint costs
#     expressiveness, which is the claim the introduction deliberately does NOT
#     make and would then be entitled to.
#
# Every outcome is reportable; only the third would be a surprise.
#
# Capacity is NOT matched, and the earlier claim here that it was matched to
# 0.8% was wrong: the run builds 2.582M parameters against the unconstrained
# 2.161M, i.e. 19.5% MORE. The module defaults were chosen to match, but
# make_model_config overrides channels/time_dim/cond_dim from the shared CLI
# flags, so those defaults never reach a run. Left as-is deliberately, because
# the direction is favourable: a constrained model that is also SMALLER
# confounds architecture with capacity and invites the uncharitable reading,
# whereas one that is larger and still no better at the ceiling does not.
# train_flow.py now prints the comparison at startup so it cannot drift again.

set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"
mkdir -p logs results checkpoints/conv

PY=.venv/bin/python
say() { echo "[$(date -Is)] $*"; }

# Two seeds in parallel, one per GPU. The complex convolutions roughly double
# the work on the m=1 stream, so expect ~6 h rather than the 3.5 h the
# unconstrained arm takes -- which is itself worth reporting, since "cheap" is
# one of the claims made for canonicalisation over architecture.
say "SO(2)-equivariant backbone, 60 envs, 300 epochs, seeds 0 and 1"
for gpu in 0 1; do
  CUDA_VISIBLE_DEVICES=$gpu $PY train_flow.py --data data --n-envs 60 --env-start 0 \
      --out "checkpoints/conv/equiv-e60-conv-s${gpu}.pt" --epochs 300 --batch 1024 \
      --reduced --equivariant --seed "$gpu" --num-workers 8 --split-by pair \
      --snapshot-every 10 --resume --amp \
      --wandb --wandb-group equivariant --wandb-name "equiv-e60-conv-s${gpu}" \
      >> "logs/equiv-e60-conv-s${gpu}.log" 2>&1 &
done
wait
say "training done"

# Scored exactly like every other arm: same held-out shards, same pairs, same
# sample count, same integrator budget. K=1 only -- frame averaging on a model
# that is ALREADY equivariant is the identity up to floating point, and running
# it would invite the reading that the two mechanisms stack.
# train_flow.py names snapshots <stem>.epNNNN<suffix>, i.e. equiv-e60-conv-s0
# .ep0010.pt -- NOT <name>.pt.epNNNN.pt. An earlier version of this loop globbed
# the latter, matched nothing, and the per-file guard below turned that into
# silence rather than an error: hours of training, no scores, no complaint.
# Count the matches first and fail loudly if the pattern is wrong.
for s in 0 1; do
  snaps=(checkpoints/conv/equiv-e60-conv-s${s}.ep*.pt)
  if [ ! -e "${snaps[0]}" ]; then
    say "ERROR: no snapshots matched for seed $s -- check the naming"
    continue
  fi
  say "seed $s: ${#snaps[@]} snapshots to score"
  for ck in "${snaps[@]}"; do
    tag="$(basename "$ck" .pt)"
    [ -e "results/conv/$tag.json" ] && continue
    $PY sweep_steps.py --ckpt "$ck" --data data --env-start 250 --n-envs 50 \
        --n-pairs 10 --n-samples 20 --steps 8 \
        --out-json "results/conv/$tag.json" >> logs/equiv-score.log 2>&1 \
        || say "FAILED scoring $tag"
  done
  say "scored seed $s"
done

say "done. Compare against the unconstrained reduced arm with:"
say "  $PY run_convergence.py table --n-envs 60 --seeds 0 1 2"
say "  ls results/conv/equiv-*.json | tail -3"
