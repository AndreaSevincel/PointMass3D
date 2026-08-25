#Side-by-side sampling video for the ICRA submission.
#
#  .venv/bin/python make_video.py --queries 6 --out icra_video.mp4
#
#Left panel is the world-frame model, right panel is the same architecture on
#the (s,g)-reduced representation, both at 60 environments and both sampled with
#the eight Euler steps the paper reports. Nothing here re-scores anything: the
#free/blocked verdict per sample comes from env.path_free, the same call
#sweep_steps.py uses, so a viewer counting coloured curves recovers the metric.
#
#Environments 250-299 are the held-out split. Using anything below 250 would put
#training data in the video.

import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from flowmatch.data import Normalizer, env_features
from flowmatch.flow import sample, sample_reduced
from flowmatch.model import build_model
from pointmass3d import BoxObstacle, PointMass3DEnv, SphereObstacle

OBST = "#8894A2"
EDGE = "#3C4653"     # box wireframe: pale edges vanish on a projector
FREE = "#C2560F"
BLOCKED = "#98A0AA"
HIT = "#D01C1C"      # first point at which a path is inside an obstacle
INK = "#141C24"


def first_hit(env, path, resolution=0.01):
    #The first point along the densely resampled path with clearance <= 0, i.e.
    #where env.path_free stops being true. Same resolution as segment_free, so
    #the dot marks a point the collision check actually rejected.
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / resolution)) + 1)
        pts = np.linspace(a, b, n)
        c = env.clearance(pts)
        k = np.argmax(c <= 0.0)
        if c[k] <= 0.0:
            return pts[k]
    return None


def build_env(npz):
    obs = [SphereObstacle(s[:3], s[3]) for s in npz["spheres"]]
    obs += [BoxObstacle(b[:3], b[3:]) for b in npz["boxes"]]
    return PointMass3DEnv(obs, robot_radius=float(npz["robot_radius"]))


def draw_env(ax, env, alpha=0.24):
    #coarse facets on purpose: this is redrawn once per frame and 40 obstacles
    #at figure quality costs more than the video gains
    for o in env.obstacles:
        if isinstance(o, SphereObstacle):
            u = np.linspace(0, 2 * np.pi, 12)
            v = np.linspace(0, np.pi, 8)
            ax.plot_surface(
                o.center[0] + o.radius * np.outer(np.cos(u), np.sin(v)),
                o.center[1] + o.radius * np.outer(np.sin(u), np.sin(v)),
                o.center[2] + o.radius * np.outer(np.ones_like(u), np.cos(v)),
                color=OBST, alpha=alpha, linewidth=0, shade=True)
        else:
            c, h = np.asarray(o.center), np.asarray(o.half_extents)
            k = c + h * np.array([[sx, sy, sz] for sx in (-1, 1)
                                  for sy in (-1, 1) for sz in (-1, 1)])
            faces = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
                     (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
            ax.add_collection3d(Poly3DCollection(
                [[k[i] for i in f] for f in faces], facecolor=OBST,
                edgecolor=EDGE, linewidths=0.7, alpha=alpha))


def style(ax, title, lim=1.0):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1), zoom=1.30)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor((1, 1, 1, 0))
        pane.pane.set_edgecolor("#C9D1D9")
    ax.set_title(title, fontsize=11, color=INK, pad=-4)


def load(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = build_model(ck["model_config"]).to(device)
    m.load_state_dict(ck["ema"]); m.eval()
    return m, Normalizer.from_dict(ck["normalizer"]), ck["n_waypoints"]


def roll_out(model, norm, N, npz, start, goal, n_samples, steps, device, seed):
    reduced = model.cond_enc.sg_dim == 1
    sp, bx = env_features(npz, norm, box_dim=model.obstacle_enc.box_dim)
    sp = sp.to(device).unsqueeze(0).expand(n_samples, -1, -1)
    bx = bx.to(device).unsqueeze(0).expand(n_samples, -1, -1)
    s = torch.from_numpy(norm.norm_pts(start).astype(np.float32)).repeat(n_samples, 1).to(device)
    g = torch.from_numpy(norm.norm_pts(goal).astype(np.float32)).repeat(n_samples, 1).to(device)
    gen = torch.Generator(device=device).manual_seed(seed)
    cap = []
    with torch.no_grad():
        if reduced:
            sample_reduced(model, sp, bx, s, g, n_waypoints=N, n_steps=steps,
                           device=device, generator=gen, capture=cap)
        else:
            sample(model, sp, bx, torch.cat([s, g], -1), anchor_start=s, anchor_goal=g,
                   n_waypoints=N, n_steps=steps, device=device, generator=gen, capture=cap)
    return [norm.denorm_pts(c.numpy()) for c in cap]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctrl", default="checkpoints/grid/ctrl-e60.pt")
    ap.add_argument("--treat", default="checkpoints/grid/treat-e60.pt")
    ap.add_argument("--data", default="data")
    ap.add_argument("--env-start", type=int, default=250)
    ap.add_argument("--queries", type=int, default=6)
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--hold", type=int, default=5, help="frames per Euler step")
    ap.add_argument("--settle", type=int, default=28, help="frames on the finished draw")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--out", default="icra_video.mp4")
    a = ap.parse_args()

    dev = torch.device("cpu")
    ctrl, cn, N = load(a.ctrl, dev)
    treat, tn, _ = load(a.treat, dev)
    assert ctrl.cond_enc.sg_dim == 6 and treat.cond_enc.sg_dim == 1, \
        "expected a world-frame ctrl and a reduced treat checkpoint"

    shots = []
    for q in range(a.queries):
        idx = a.env_start + q
        npz = np.load(f"{a.data}/env_{idx:04d}.npz", allow_pickle=True)
        env = build_env(npz)
        start, goal = npz["starts"][0], npz["goals"][0]
        L = roll_out(ctrl, cn, N, npz, start, goal, a.n_samples, a.steps, dev, q)
        R = roll_out(treat, tn, N, npz, start, goal, a.n_samples, a.steps, dev, q)
        fl = [env.path_free(p) for p in L[-1]]
        fr = [env.path_free(p) for p in R[-1]]
        #hit points at EVERY step, so the dots track the trajectory as it settles
        hl, hr = [], []
        for traj, out in ((L, hl), (R, hr)):
            for st in traj:
                pts = [first_hit(env, p) for p in st]
                pts = [q for q in pts if q is not None]
                out.append(np.array(pts) if pts else np.empty((0, 3)))
        shots.append(dict(env=env, s=start, g=goal, L=L, R=R, fl=fl, fr=fr,
                          hl=hl, hr=hr, idx=idx))
        print(f"env {idx}: world {sum(fl)}/{a.n_samples} free, "
              f"reduced {sum(fr)}/{a.n_samples} free", flush=True)

    per = a.steps * a.hold + a.settle
    total = per * len(shots)
    fig = plt.figure(figsize=(10, 5.2), dpi=140)
    axl = fig.add_subplot(121, projection="3d")
    axr = fig.add_subplot(122, projection="3d")
    cap = fig.text(0.5, 0.045, "", ha="center", fontsize=11, color=INK)
    sub = fig.text(0.5, 0.008, "", ha="center", fontsize=9, color="#6B7683")
    state = {"shot": -1, "lines": None}

    def setup(k):
        sh = shots[k]
        for ax, name in ((axl, "world frame"), (axr, "$(s,g)$ reduction")):
            ax.clear(); draw_env(ax, sh["env"]); style(ax, name)
            ax.scatter(*sh["s"], s=40, color=INK, depthshade=False)
            ax.scatter(*sh["g"], s=60, color=INK, marker="*", depthshade=False)
        ln = ([axl.plot([], [], [], lw=1.0)[0] for _ in range(a.n_samples)],
              [axr.plot([], [], [], lw=1.0)[0] for _ in range(a.n_samples)])
        for j in range(a.n_samples):
            for side, ok in ((0, sh["fl"][j]), (1, sh["fr"][j])):
                ln[side][j].set_color(FREE if ok else BLOCKED)
                ln[side][j].set_linewidth(1.3 if ok else 0.6)
                ln[side][j].set_alpha(0.9 if ok else 0.38)
                ln[side][j].set_zorder(5 if ok else 2)
        #one red dot per failing path, at the first point the checker rejected
        dots = (axl.plot([], [], [], "o", ms=4.0, color=HIT, mew=0,
                         zorder=8, ls="none")[0],
                axr.plot([], [], [], "o", ms=4.0, color=HIT, mew=0,
                         zorder=8, ls="none")[0])
        state["shot"], state["lines"], state["dots"] = k, ln, dots
        sub.set_text(f"held-out environment {sh['idx']}, "
                     f"{a.n_samples} samples, {a.steps} Euler steps")

    def frame(i):
        k, f = divmod(i, per)
        if k != state["shot"]:
            setup(k)
        sh = shots[k]
        step = min(f // a.hold, a.steps - 1)
        for j in range(a.n_samples):
            for lines, traj in ((state["lines"][0], sh["L"]), (state["lines"][1], sh["R"])):
                p = traj[step][j]
                lines[j].set_data(p[:, 0], p[:, 1]); lines[j].set_3d_properties(p[:, 2])
        for side, key in ((0, "hl"), (1, "hr")):
            h = sh[key][step]
            state["dots"][side].set_data(h[:, 0], h[:, 1])
            state["dots"][side].set_3d_properties(h[:, 2])
        cap.set_text(f"step {step + 1}/{a.steps}          "
                     f"collision-free   {sum(sh['fl'])}/{a.n_samples}"
                     f"   vs   {sum(sh['fr'])}/{a.n_samples}")
        return []

    print(f"rendering {total} frames -> {a.out}", flush=True)
    anim = animation.FuncAnimation(fig, frame, frames=total, interval=1000 / a.fps)
    anim.save(a.out, writer=animation.FFMpegWriter(fps=a.fps, bitrate=3200))
    print(f"wrote {a.out} ({total / a.fps:.1f} s)")


if __name__ == "__main__":
    main()
