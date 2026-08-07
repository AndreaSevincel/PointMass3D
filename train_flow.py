

  #.venv/bin/python train_flow.py --data data1 --epochs 300 --batch 1024 --amp --multi-gpu

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from flowmatch import tracking
from flowmatch.data import build_datasets
from flowmatch.diffusion import Schedule, diffusion_loss
from flowmatch.flow import EMA, flow_matching_loss
from flowmatch.model import FlowVelocityField


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data1")
    ap.add_argument("--out", type=str, default="checkpoints/flow.pt")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true", help="mixed precision (CUDA)")
    ap.add_argument("--multi-gpu", action="store_true", help="DataParallel over all visible GPUs")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the model. Often 1.3-2x on this "
                         "conv trunk, at the cost of a slow first epoch")
    # model
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--n-blocks", type=int, default=8)
    # equivariance arm
    ap.add_argument("--reduced", action="store_true",
                    help="treatment arm: (s,g) reduction + roll augmentation")
    ap.add_argument("--no-roll", action="store_true",
                    help="with --reduced, disable roll augmentation (ablation)")
    ap.add_argument("--n-envs", type=int, default=None,
                    help="use N environments starting at --env-start (scale sweep)")
    ap.add_argument("--env-start", type=int, default=0,
                    help="first env shard to use; reserve a tail range as test set")
    ap.add_argument("--split-by", choices=["traj", "pair", "env"], default="pair",
                    help="what val holds out. traj is LEAKY on this dataset "
                         "(30 near-duplicate paths per pair); pair is the default")
    ap.add_argument("--check-frame", action="store_true",
                    help="assert the reduction is well-formed on every batch (slow)")
    #generative-model control: same backbone, same conditioning, same frame
    #options, different objective. See flowmatch/diffusion.py.
    #The two arms a referee asks for: augmentation is the practitioner's
    #alternative to canonicalisation, and translation-only splits the five
    #removable DOF into 3 translations vs 2 rotations.
    ap.add_argument("--augment", type=float, default=0.0,
                    help="world-frame arm: random SE(3) augmentation per batch. "
                         "The value is the translation half-width in normalised "
                         "units (0.25 is a quarter of the workspace). 0 = off")
    ap.add_argument("--reduce-mode", choices=["full", "translation"], default="full",
                    help="with --reduced: full removes 3 translations + 2 "
                         "rotations; translation removes only the translations")
    ap.add_argument("--domain", choices=["pointmass", "se3"], default="pointmass",
                    help="pointmass: 3-dim waypoints. se3: 9-dim poses "
                         "(position + 6D rotation), see se3body/")
    ap.add_argument("--objective", choices=["flow", "ddpm"], default="flow",
                    help="flow: conditional OT velocity regression. "
                         "ddpm: Diffuser/MPD-style eps-prediction")
    ap.add_argument("--diffusion-steps", type=int, default=100,
                    help="T for --objective ddpm (sampling NFE is set at eval)")
    tracking.add_args(ap)
    return ap.parse_args()


def default_run_name(args):
    #Arm and env count first: those are the 2x4 grid axes.
    arm = "treat" if args.reduced else "ctrl"
    if args.reduced and args.no_roll:
        arm += "-noroll"
    if args.objective != "flow":
        arm = f"{args.objective}-{arm}"
    if args.domain != "pointmass":
        arm = f"{args.domain}-{arm}"
    if args.reduced and args.reduce_mode != "full":
        arm = f"{arm}-{args.reduce_mode}"
    if args.augment > 0:
        arm = f"{arm}-aug"
    envs = f"e{args.n_envs}" if args.n_envs else f"e{Path(args.data).name}"
    return f"{arm}-{envs}-c{args.channels}-b{args.n_blocks}-s{args.seed}"


def make_model_config(args):
    return dict(
        channels=args.channels,
        n_blocks=args.n_blocks,
        dilations=(1, 2, 4, 8),
        time_dim=256,
        env_hidden=128,
        env_dim=128,
        cond_dim=256,
        groups=8,
        box_dim=12,
        #After the reduction start/goal are (-+d/2, 0, 0), so the six numbers
        #collapse to the one invariant scalar d. The control genuinely needs
        #the raw pair -- this asymmetry is intrinsic to the treatment.
        #point mass: 6 raw numbers, or the single invariant d after reduction.
        #SE(3): 18 raw (two poses), or d plus both orientations = 13, since the
        #reduction canonicalises the positions but not the orientations.
        #translation-only leaves the query DIRECTION informative, so the
        #conditioning is the full vector g-s rather than the scalar d
        sg_dim=(
            (3 if args.reduce_mode == "translation" else 1) if args.reduced else 6
        ) if args.domain == "pointmass" else (13 if args.reduced else 18),
        state_dim=3 if args.domain == "pointmass" else 9,
    )


def compute_loss(net, batch, tables, args, schedule, check=False):
    if args.domain == "se3":
        if args.objective == "ddpm":
            raise SystemExit("the ddpm arm is implemented for the point-mass "
                             "domain only; use --objective flow with --domain se3")
        from se3body.flow import flow_matching_loss_se3
        return flow_matching_loss_se3(net, batch, tables, reduced=args.reduced,
                                      roll=not args.no_roll, check=check)
    if args.objective == "ddpm":
        return diffusion_loss(net, batch, tables, schedule, reduced=args.reduced,
                              roll=not args.no_roll, check=check)
    return flow_matching_loss(net, batch, tables, reduced=args.reduced,
                              roll=not args.no_roll, check=check,
                              mode=args.reduce_mode, augment=args.augment)


def evaluate(net, loader, tables, device, args, schedule):
    net.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = compute_loss(net, batch, tables, args, schedule)
            bs = batch["traj"].shape[0]
            total += loss.item() * bs
            n += bs
    return total / max(n, 1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        #Shapes are fixed for the whole run, so let cuDNN autotune once instead
        #of picking a generic algorithm every call. TF32 costs nothing here:
        #the targets are noisy regression values, not something needing full
        #fp32 mantissa.
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"device={device}  cuda_devices={torch.cuda.device_count()}")

    train_ds, val_ds, normalizer, _ = build_datasets(
        args.data, val_frac=args.val_frac, seed=args.seed, n_envs=args.n_envs,
        env_start=args.env_start, split_by=args.split_by,
    )
    print(f"split_by={args.split_by}  envs [{args.env_start}, "
          f"{args.env_start + (args.n_envs or 0)})")
    print(f"train={len(train_ds)}  val={0 if val_ds is None else len(val_ds)} trajectories")

    #obstacle lookup tables live on the device the whole time
    tables = {
        "spheres": train_ds.spheres.to(device),
        "boxes": train_ds.boxes.to(device),
        "sphere_mask": train_ds.sphere_mask.to(device),
        "box_mask": train_ds.box_mask.to(device),
    }

    #persistent_workers matters here: without it the worker processes are torn
    #down and respawned every epoch, and each respawn re-imports torch and
    #re-inherits the dataset. Over 20-60 epochs that is pure overhead.
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch, num_workers=args.num_workers)
        if val_ds is not None else None
    )

    #A fixed slice of TRAINING data, scored with the same (EMA) weights and the
    #same code path as validation. Without it the printed train/val pair is not
    #a comparison: train is a running average over the epoch on the raw weights
    #while the optimiser is still moving them, and val is the EMA weights at
    #epoch end. Both differences push val below train, so "val < train" is the
    #expected reading even under mild overfitting, and cannot be used as
    #evidence of underfitting. This loader makes the gap mean what it says.
    train_eval_loader = None
    if val_ds is not None and len(train_ds) > 0:
        n_probe = min(len(val_ds), len(train_ds))
        probe_idx = np.random.default_rng(args.seed).choice(
            np.asarray(train_ds.indices), size=n_probe, replace=False
        )
        train_eval_loader = DataLoader(
            train_ds.subset(probe_idx), batch_size=args.batch,
            num_workers=args.num_workers,
        )

    cfg = make_model_config(args)
    core = FlowVelocityField(**cfg).to(device)
    n_params = sum(p.numel() for p in core.parameters())
    print(f"model params: {n_params/1e6:.2f}M")

    ema = EMA(core, decay=args.ema_decay)
    net = core
    if args.compile:
        net = torch.compile(net)
    if args.multi_gpu and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(core)
        print(f"DataParallel over {torch.cuda.device_count()} GPUs")

    run = tracking.from_args(
        args,
        name=default_run_name(args),
        config={
            **vars(args),
            "model": cfg,
            "n_params": n_params,
            "n_train": len(train_ds),
            "n_val": 0 if val_ds is None else len(val_ds),
            "n_envs": train_ds.spheres.shape[0],
            "n_waypoints": train_ds.trajs.shape[1],
            "cuda_devices": torch.cuda.device_count(),
        },
    )
    if run.active:
        print(f"wandb: {run.url}")

    #Noise schedule for --objective ddpm; unused by the flow arm.
    schedule = Schedule(args.diffusion_steps, device=device)

    opt = torch.optim.AdamW(core.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = args.amp and device.type == "cuda"
    try:  # current API (torch >= 2.3); fall back for older builds
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    t0 = time.time()
    gstep = 0

    for epoch in range(args.epochs):
        net.train()
        running, seen = 0.0, 0
        t_epoch = time.time()
        for it, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = compute_loss(net, batch, tables, args, schedule,
                                    check=args.check_frame)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ema.update(core)

            bs = batch["traj"].shape[0]
            running += loss.item() * bs
            seen += bs
            gstep += 1
            if args.log_every and it % args.log_every == 0:
                print(f"  epoch {epoch:03d} it {it:04d}  loss {loss.item():.4f}")
                run.log({"train/loss_step": loss.item(), "epoch": epoch}, step=gstep)

        train_loss = running / max(seen, 1)
        msg = f"epoch {epoch:03d}  train {train_loss:.4f}"
        if val_loader is not None:
            val_loss = evaluate(ema.shadow, val_loader, tables, device,
                                args, schedule)
            #same weights, same code path, training data
            train_ema = evaluate(ema.shadow, train_eval_loader, tables, device,
                                 args, schedule)
            msg += f"  train(ema) {train_ema:.4f}"
            msg += f"  val(ema) {val_loss:.4f}"
        else:
            val_loss = train_loss
        msg += f"  [{time.time()-t0:.0f}s]"
        print(msg)
        run.log(
            {
                "train/loss": train_loss,
                "train/loss_ema": train_ema,
                "val/loss_ema": val_loss,
                #the only honest generalisation gap: like-for-like weights
                "gap/val_minus_train_ema": val_loss - train_ema,
                "lr": opt.param_groups[0]["lr"],
                "epoch": epoch,
                "time/epoch_s": time.time() - t_epoch,
                "time/elapsed_s": time.time() - t0,
            },
            step=gstep,
        )

        if val_loss < best_val:
            best_val = val_loss
            run.summary(best_val_loss=best_val, best_epoch=epoch)
            torch.save(
                {
                    "model": core.state_dict(),
                    "ema": ema.state_dict(),
                    "model_config": cfg,
                    "normalizer": normalizer.to_dict(),
                    "n_waypoints": train_ds.trajs.shape[1],
                    "epoch": epoch,
                    "val_loss": best_val,
                    #eval needs to know which sampler to use; absent => "flow"
                    "objective": args.objective,
                    "domain": args.domain,
                    "reduce_mode": args.reduce_mode,
                    "augment": args.augment,
                    "diffusion_steps": args.diffusion_steps,
                    "reduced": args.reduced,
                },
                args.out,
            )
    print(f"done. best val {best_val:.4f}. checkpoint -> {args.out}")
    run.summary(total_time_s=time.time() - t0)
    run.finish()


if __name__ == "__main__":
    main()
