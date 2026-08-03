

  #.venv/bin/python train_flow.py --data data1 --epochs 300 --batch 1024 --amp --multi-gpu

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from flowmatch import tracking
from flowmatch.data import build_datasets
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
    tracking.add_args(ap)
    return ap.parse_args()


def default_run_name(args):
    #Arm and env count first: those are the 2x4 grid axes.
    arm = "treat" if args.reduced else "ctrl"
    if args.reduced and args.no_roll:
        arm += "-noroll"
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
        sg_dim=1 if args.reduced else 6,
    )


def evaluate(net, loader, tables, device, reduced=False, roll=True):
    net.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = flow_matching_loss(net, batch, tables, reduced=reduced, roll=roll)
            bs = batch["traj"].shape[0]
            total += loss.item() * bs
            n += bs
    return total / max(n, 1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
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

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=(device.type == "cuda"),
    )
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch, num_workers=args.num_workers)
        if val_ds is not None else None
    )

    cfg = make_model_config(args)
    core = FlowVelocityField(**cfg).to(device)
    n_params = sum(p.numel() for p in core.parameters())
    print(f"model params: {n_params/1e6:.2f}M")

    ema = EMA(core, decay=args.ema_decay)
    net = core
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
                loss = flow_matching_loss(
                    net, batch, tables, reduced=args.reduced,
                    roll=not args.no_roll, check=args.check_frame,
                )
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
            val_loss = evaluate(
                ema.shadow, val_loader, tables, device,
                reduced=args.reduced, roll=not args.no_roll,
            )
            msg += f"  val(ema) {val_loss:.4f}"
        else:
            val_loss = train_loss
        msg += f"  [{time.time()-t0:.0f}s]"
        print(msg)
        run.log(
            {
                "train/loss": train_loss,
                "val/loss_ema": val_loss,
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
                },
                args.out,
            )
    print(f"done. best val {best_val:.4f}. checkpoint -> {args.out}")
    run.summary(total_time_s=time.time() - t0)
    run.finish()


if __name__ == "__main__":
    main()
