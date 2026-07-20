#3D visualization of environments and trajectories with matplotlib.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .env import BoxObstacle, SphereObstacle


def _draw_sphere(ax, center, radius, color="0.55", alpha=0.35):
    u = np.linspace(0.0, 2.0 * np.pi, 24)
    v = np.linspace(0.0, np.pi, 16)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, shade=True)


def _draw_box(ax, center, half, color="0.55", alpha=0.35):
    c, h = np.asarray(center), np.asarray(half)
    corners = c + h * np.array(
        [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    )
    faces_idx = [
        (0, 1, 3, 2), (4, 5, 7, 6),  # x-, x+
        (0, 1, 5, 4), (2, 3, 7, 6),  # y-, y+
        (0, 2, 6, 4), (1, 3, 7, 5),  # z-, z+
    ]
    faces = [[corners[i] for i in idx] for idx in faces_idx]
    ax.add_collection3d(
        Poly3DCollection(faces, facecolor=color, edgecolor="0.35",
                         linewidths=0.4, alpha=alpha)
    )


def draw_environment(ax, env, color="0.55", alpha=0.35):
    for obs in env.obstacles:
        if isinstance(obs, SphereObstacle):
            _draw_sphere(ax, obs.center, obs.radius, color, alpha)
        elif isinstance(obs, BoxObstacle):
            _draw_box(ax, obs.center, obs.half_extents, color, alpha)


def plot_scene(
    env,
    trajectories=None,
    start=None,
    goal=None,
    title=None,
    save=None,
    elev=22,
    azim=48,
    show=False,
):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection="3d")
    draw_environment(ax, env)

    colors = plt.cm.tab10.colors
    if trajectories:
        for k, (label, path) in enumerate(trajectories.items()):
            if path is None:
                continue
            path = np.asarray(path)
            ax.plot(path[:, 0], path[:, 1], path[:, 2], "-", lw=2.2,
                    color=colors[k % 10], label=label)
            ax.scatter(path[1:-1, 0], path[1:-1, 1], path[1:-1, 2],
                       s=6, color=colors[k % 10], depthshade=False)

    if start is not None:
        ax.scatter(*start, s=120, marker="^", color="tab:green",
                   edgecolor="k", label="start", depthshade=False)
    if goal is not None:
        ax.scatter(*goal, s=160, marker="*", color="tab:red",
                   edgecolor="k", label="goal", depthshade=False)

    ax.set_xlim(env.lo, env.hi)
    ax.set_ylim(env.lo, env.hi)
    ax.set_zlim(env.lo, env.hi)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x"), ax.set_ylabel("y"), ax.set_zlabel("z")
    ax.view_init(elev=elev, azim=azim)
    if title:
        ax.set_title(title)
    if trajectories or start is not None:
        ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax
