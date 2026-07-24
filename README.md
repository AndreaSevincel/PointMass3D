# PointMass3D

Stage 1 benchmark: a 3D point-mass motion-planning environment with expert
trajectories from classical planners (**RRT-Connect**, **CHOMP**, **TrajOpt**),
3D collision checking, and visualization.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Quick start

```bash
# one problem, all three planners, comparison table + demo.png
.venv/bin/python demo.py --seed 0

# expert-trajectory dataset (RRT-Connect -> shortcut -> CHOMP refinement)
.venv/bin/python generate_dataset.py --n-envs 10 --n-trajs 20 --refine chomp
```

## Environment

`PointMass3DEnv` (pointmass3d/env.py): a spherical robot (radius 0.03) in the
workspace `[-1, 1]^3` with sphere and axis-aligned box obstacles. Collision
checking goes through an analytic **signed distance field**, each obstacle
implements `sdf(points)`, the environment takes the min over obstacles and the
workspace walls, and `clearance(q) = sdf(q) - robot_radius` is positive iff q
is collision-free. `clearance_grad` (central differences on the SDF) drives
both optimizers. Segments/paths are validated by dense sampling.

## Planners (pointmass3d/planners/)

- **RRT-Connect** — bidirectional tree search with a greedy connect step
  Probabilistically complete; output is jagged, so it is post-processed 
  with random shortcutting and arc-length resampling to a fixed number of waypoints.
- **CHOMP** — covariant gradient descent on `F_obs + λ F_smooth`. Waypoint 
  gradients combine the SDF hinge cost (arc-length weighted, with the curvature term)
  and are preconditioned by the inverse
  finite-difference metric `A⁻¹`, which spreads updates smoothly along the
  trajectory. Local method: may need an RRT initialization in clutter.
- **TrajOpt** — sequential convex optimization: SDF
  constraints `clearance ≥ d_safe` are convexified via their gradient and
  enforced through an escalating penalty loop; each convex subproblem is
  solved in closed form with a proximal trust region. (Simplification vs. the
  paper: squared-hinge penalties instead of L1, no external QP solver.)

## Expert data pipeline

`generate_dataset.py`: sample env + start/goal → RRT-Connect → shortcut →
resample to N waypoints → refine with CHOMP/TrajOpt (initialized from the RRT
path) → dense collision validation (fall back to the raw RRT path if the
refined one collides). One `.npz` per environment:

| key | shape | meaning |
|---|---|---|
| `spheres` | (S, 4) | center xyz, radius |
| `boxes` | (B, 6) | center xyz, half-extents |
| `trajs` | (T, N, 3) | expert trajectories |
| `starts`, `goals` | (T, 3) | endpoints |

Fixed-length, smooth, collision-free trajectories, directly usable as
training data for diffusion / flow-matching planners (MPD, FlowMP style).

## Flow-matching planner (`flowmatch/`)

A **pure, non-SE(3)-equivariant flow-matching** baseline that learns to
generate trajectories conditioned on (start, goal, obstacles):

- **Prior** standard Gaussian `x0 ~ N(0, I)` over `ℝ^{N×3}` — *not* the
  Brownian bridge in `brownian.py`.
- **Path / target** straight-line `xt = (1-t)x0 + t x1`, velocity `u = x1 - x0`
  (conditional OT / rectified flow); loss `‖v_θ(xt,t,c) - u‖²`.
- **Backbone** dilated temporal ConvNet (WaveNet-style, FiLM conditioning) on
  raw world coordinates — deliberately *not* equivariant, no canonicalization.
- **Conditioning** start, goal, and a permutation-invariant PointNet-style
  encoder over the sphere/box sets (generalizes to unseen obstacle layouts).

```bash
.venv/bin/pip install -r requirements-train.txt   # torch (use the CUDA wheel on the cluster)

# train (single GPU); add --multi-gpu for DataParallel across both 4090s
.venv/bin/python train_flow.py --data data1 --epochs 300 --batch 512 --amp

# sample + evaluate against the real env (collision-free %, clearance, endpoints)
.venv/bin/python sample_flow.py --ckpt checkpoints/flow.pt --data data1 \
    --env-idx 0 --n-pairs 5 --n-samples 20 --plot samples.png
```

Sampling integrates `dx/dt = v_θ` with Euler steps from the Gaussian prior;
`--anchor-endpoints` holds the first/last waypoints on the flow path so samples
land exactly on start/goal (flow-matching inpainting). The checkpoint bundles
weights, EMA weights, the normalizer and model config for standalone sampling.
