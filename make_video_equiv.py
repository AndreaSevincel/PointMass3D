#The equivariance shot: one query, rotated through a full turn, with each
#model's answer rotated back.
#
#  .venv/bin/python make_video_equiv.py --out icra_equivariance.mp4
#
#Both panels show the SAME fixed scene. What varies is the rotation R applied to
#the model's inputs. The output is pulled back by R^T before it is drawn, so an
#equivariant planner draws one curve that never moves and a non-equivariant one
#draws a different curve at every angle. The overlaid number is the RMS
#displacement of the SAMPLE MEAN from the theta=0 answer, in workspace units.
#The mean matters: the reduced sampler draws its noise in the reduced frame,
#which the rotation rolls, so a fixed seed does not fix the realised noise and a
#per-sample displacement would charge sampling scatter to non-equivariance.
#Averaging over samples removes that confound, and what survives for the reduced
#model is the roll of Prop. 1. Measured over a full turn on env 252:
#   world frame  mean 0.0506  max 0.1377
#   reduction    mean 0.0081  max 0.0145      (robot radius 0.03)
#
#Nothing is collision-checked here. Rotating an axis-aligned scene produces
#oriented boxes that PointMass3DEnv does not model, and the claim under test is
#geometric consistency rather than success rate, so a free/blocked colouring
#would be both wrong and beside the point.

import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

from flowmatch.data import Normalizer, env_features
from flowmatch.flow import sample, sample_reduced
from flowmatch.geometry import rotate_box_features, rotate_sphere_features
from flowmatch.model import build_model
from make_video import BLOCKED, FREE, INK, build_env, draw_env, load, style


def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
                        dtype=torch.float32)


def answer_at(model, sp, bx, s_n, g_n, R, origin, N, steps, n_samples, seed):
    #Rotate the inputs about `origin` in NORMALISED coordinates, sample, and
    #pull the result back by R^T. Same generator seed at every angle, so any
    #movement on screen is the model and not the noise.
    Rb = R.unsqueeze(0)
    ob = origin.unsqueeze(0)
    spR = rotate_sphere_features(sp.unsqueeze(0), Rb, ob)
    bxR = rotate_box_features(bx.unsqueeze(0), Rb, ob)
    sR = (R @ (s_n - origin)) + origin
    gR = (R @ (g_n - origin)) + origin

    B = n_samples
    spB = spR.expand(B, -1, -1)
    bxB = bxR.expand(B, -1, -1)
    sB, gB = sR.repeat(B, 1), gR.repeat(B, 1)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        if model.cond_enc.sg_dim == 1:
            x = sample_reduced(model, spB, bxB, sB, gB, n_waypoints=N,
                               n_steps=steps, generator=gen)
        else:
            x = sample(model, spB, bxB, torch.cat([sB, gB], -1),
                       anchor_start=sB, anchor_goal=gB, n_waypoints=N,
                       n_steps=steps, generator=gen)
    #pull back: R^T (x - origin) + origin
    return torch.einsum("ji,bkj->bki", R, x - origin) + origin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctrl", default="checkpoints/grid/ctrl-e60.pt")
    ap.add_argument("--treat", default="checkpoints/grid/treat-e60.pt")
    ap.add_argument("--data", default="data")
    ap.add_argument("--env-idx", type=int, default=252)
    ap.add_argument("--pair", type=int, default=0)
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--angles", type=int, default=72)
    ap.add_argument("--hold", type=int, default=8)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--out", default="icra_equivariance.mp4")
    a = ap.parse_args()

    ctrl, cn, N = load(a.ctrl, torch.device("cpu"))
    treat, tn, _ = load(a.treat, torch.device("cpu"))
    npz = np.load(f"{a.data}/env_{a.env_idx:04d}.npz", allow_pickle=True)
    env = build_env(npz)
    start, goal = npz["starts"][a.pair], npz["goals"][a.pair]

    thetas = np.linspace(0.0, 2 * np.pi, a.angles, endpoint=False)
    series, devs = {}, {}
    for tag, model, norm in (("ctrl", ctrl, cn), ("treat", treat, tn)):
        sp, bx = env_features(npz, norm, box_dim=model.obstacle_enc.box_dim)
        s_n = torch.from_numpy(norm.norm_pts(start).astype(np.float32))
        g_n = torch.from_numpy(norm.norm_pts(goal).astype(np.float32))
        origin = torch.from_numpy(norm.norm_pts(np.zeros(3)).astype(np.float32))
        out = []
        for th in thetas:
            x = answer_at(model, sp, bx, s_n, g_n, rot_z(float(th)), origin,
                          N, a.steps, a.n_samples, seed=0)
            out.append(norm.denorm_pts(x.numpy()))
        series[tag] = out
        means = [o.mean(0) for o in out]
        base = means[0]
        devs[tag] = [float(np.sqrt(((m - base) ** 2).sum(-1).mean())) for m in means]
        series[tag + "_mean"] = means
        print(f"{tag}: sample-mean pull-back displacement over a full turn, "
              f"mean {np.mean(devs[tag]):.4f}, max {np.max(devs[tag]):.4f}",
              flush=True)

    per = a.hold
    total = a.angles * per
    fig = plt.figure(figsize=(10, 5.4), dpi=140)
    axl = fig.add_subplot(121, projection="3d")
    axr = fig.add_subplot(122, projection="3d")
    head = fig.text(0.5, 0.955, "the same problem rotated by $\\theta$, "
                    "with the planner's answer rotated back",
                    ha="center", fontsize=11.5, color=INK)
    cap = fig.text(0.5, 0.05, "", ha="center", fontsize=11, color=INK)
    sub = fig.text(0.5, 0.012, "RMS displacement of the sample mean from the "
                   "$\\theta=0$ answer, in workspace units (robot radius 0.03)",
                   ha="center", fontsize=9, color="#6B7683")
    lines = {}
    for ax, name, tag in ((axl, "world frame", "ctrl"),
                          (axr, "$(s,g)$ reduction", "treat")):
        draw_env(ax, env); style(ax, name)
        ax.scatter(*start, s=40, color=INK, depthshade=False)
        ax.scatter(*goal, s=60, color=INK, marker="*", depthshade=False)
        #the theta=0 mean, drawn once and left in place as the reference
        p0 = series[tag + "_mean"][0]
        ax.plot(p0[:, 0], p0[:, 1], p0[:, 2], lw=1.1, color=BLOCKED,
                alpha=0.9, zorder=1)
        #a thinned spray of individual samples, so the mean is visibly a mean
        lines[tag] = [ax.plot([], [], [], lw=0.6, color=FREE, alpha=0.22,
                              zorder=4)[0] for _ in range(12)]
        lines[tag + "_mean"] = ax.plot([], [], [], lw=2.0, color=FREE,
                                       zorder=7)[0]

    def frame(i):
        k = min(i // per, a.angles - 1)
        for tag in ("ctrl", "treat"):
            for j, ln in enumerate(lines[tag]):
                p = series[tag][k][j]
                ln.set_data(p[:, 0], p[:, 1]); ln.set_3d_properties(p[:, 2])
            m = series[tag + "_mean"][k]
            lines[tag + "_mean"].set_data(m[:, 0], m[:, 1])
            lines[tag + "_mean"].set_3d_properties(m[:, 2])
        cap.set_text(f"$\\theta$ = {np.degrees(thetas[k]):5.0f}$^\\circ$"
                     f"          world {devs['ctrl'][k]:.3f}"
                     f"     reduced {devs['treat'][k]:.3f}")
        return []

    print(f"rendering {total} frames -> {a.out}", flush=True)
    anim = animation.FuncAnimation(fig, frame, frames=total, interval=1000 / a.fps)
    anim.save(a.out, writer=animation.FFMpegWriter(fps=a.fps, bitrate=3200))
    print(f"wrote {a.out} ({total / a.fps:.1f} s)")


if __name__ == "__main__":
    main()
