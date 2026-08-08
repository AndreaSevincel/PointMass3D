
  #.venv/bin/python make_figures.py

  #Regenerate the report figures from the measured numbers. Kept as a script so
  #the figures track the results rather than drifting from them.

  #Source: sweep_steps.py on held-out envs 250-299 (50 envs x 10 pairs x
  #20 samples, 8 Euler steps, K_FA=1). See main.tex Section 4 / paper.tex Section VI.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ENVS = [20, 60, 150, 250]
# 60-env cells are 3-seed means (seed 0 was the worst treatment draw of three);
# the others are single seed. See aggregate_runs.py output, 2026-08-08.
CONTROL = [12.77, 13.57, 14.60, 15.38]     # world frame
AUGMENTED = [12.72, 13.84, 15.25, 17.50]   # world frame + random SE(3) augmentation
TREATMENT = [18.97, 34.43, 41.00, 45.63]   # (s,g) reduction
NOROLL_X, NOROLL_Y = 60, 30.06          # ablation: reduction without roll augmentation

# mechanism decomposition, percentage points on the held-out collision-free rate
MECHANISMS = [
    (r"$(s,g)$ reduction" "\n" "(5 DOF, exact)", 30.2),
    ("SE(3) augmentation\n(world frame)", 2.1),
    ("roll augmentation\n(training)", 4.4),
    ("frame averaging\n($K{=}1\\to3$)", 0.1),
]
# across-SEED standard error at 60 envs, treatment arm (n=3). This is the right
# floor for a claim about a method; the old 2.0 came from the spread over
# held-out problems, which answers a different question.
SE = 1.56

# non-equivariance residual, RMS (Hilbert) definition of Eq. 6, K=3.
# Flat across a 12.5x range of training data -- the earlier "falls with scale"
# reading came from two points under the old mean-of-norms definition.
RESID_ENVS = [20, 60, 150, 250]
RESID = [0.0183, 0.0170, 0.0177, 0.0168]
RESID_NOROLL = (60, 0.0279)

# classical reference points, measured by baselines_classical.py on the SAME
# 500 held-out problems (envs 250-299 x 10 pairs). Times are single-core CPU.
CLASSICAL = [
    ("RRT-Connect", 99.6, 0.284),
    ("expert (RRT+CHOMP)", 100.0, 0.342),
    ("CHOMP", 88.8, 0.394),
    ("TrajOpt", 74.0, 0.352),
]
STRAIGHT_LINE = 15.6           # trivial baseline: the segment start -> goal
# learned arms at 250 training envs, 8 Euler steps, 20 samples/query on one GPU
LEARNED = [
    (r"world frame", 15.38, 0.067),
    (r"world frame + aug.", 17.50, 0.067),
    (r"$(s,g)$ reduction", 45.63, 0.067),
]

C_CTRL = "#3E6DA8"
C_TREAT = "#C2560F"
INK = "#333333"
MUTED = "#777777"


def _despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        if side in keep:
            ax.spines[side].set_color("0.6")
        else:
            ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)


def fig_scaling(out="fig_scaling.pdf", size=(6.4, 3.9), fs=9.0):
    """size/fs are set so the figure is placed at 1:1 -- report figures are wide
    (single column, 0.82\\textwidth), paper figures are one column of two."""
    fig, ax = plt.subplots(figsize=size)

    # the gap between the arms is the result -- shade it
    ax.fill_between(ENVS, CONTROL, TREATMENT, color=C_TREAT, alpha=0.08, linewidth=0)

    ax.plot(ENVS, TREATMENT, "-o", color=C_TREAT, linewidth=2, markersize=6,
            label=r"$(s,g)$ reduction", zorder=3)
    ax.plot(ENVS, CONTROL, "-s", color=C_CTRL, linewidth=2, markersize=5.5,
            label="World frame", zorder=3)
    #Augmentation is the same arm using the symmetry a different way, so it
    #keeps the control's hue and is distinguished by a dashed stroke.
    ax.plot(ENVS, AUGMENTED, "--^", color=C_CTRL, linewidth=1.6, markersize=5,
            markerfacecolor="white", label="World frame + SE(3) aug.", zorder=3)

    # the no-roll ablation is the same arm with one part removed: same hue, hollow
    ax.plot([NOROLL_X], [NOROLL_Y], "o", markerfacecolor="white",
            markeredgecolor=C_TREAT, markeredgewidth=1.6, markersize=7, zorder=4)
    ax.annotate("no-roll ablation", xy=(NOROLL_X, NOROLL_Y), xytext=(9, -9),
                textcoords="offset points", fontsize=fs - 1.0, color="#555555")

    for x, y, c, dy in ((250, TREATMENT[-1], C_TREAT, 6),
                        (250, AUGMENTED[-1], C_CTRL, 5),
                        (250, CONTROL[-1], C_CTRL, -12)):
        ax.annotate(f"{y:.1f}%", xy=(x, y), xytext=(-4, dy),
                    textcoords="offset points", fontsize=fs, color=c,
                    fontweight="bold", ha="right")

    ax.set_xscale("log")
    ax.set_xticks(ENVS)
    ax.set_xticklabels([str(e) for e in ENVS])
    ax.minorticks_off()
    ax.set_xlim(17, 320)
    ax.set_ylim(0, 50)
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.set_xlabel("Training environments (log scale)", fontsize=fs + 0.5)
    ax.set_ylabel("Collision-free rate (%)", fontsize=fs + 0.5)
    ax.tick_params(labelsize=fs)
    ax.grid(axis="y", color="0.88", linewidth=0.7)
    _despine(ax)
    ax.legend(frameon=False, fontsize=fs, loc="upper left")

    fig.tight_layout(pad=0.4)
    fig.savefig(out)
    print(f"wrote {out}")


def fig_mechanisms(out="fig_mechanisms.pdf"):
    """Left: what each mechanism is worth, against the noise floor.
    Right: the quantity frame averaging exists to remove, versus scale."""
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(6.9, 2.9),
                                   gridspec_kw={"width_ratios": [1.55, 1.0]})

    # -- left: one series, so no legend; direct labels on every bar --------
    labels = [m[0] for m in MECHANISMS][::-1]
    vals = [m[1] for m in MECHANISMS][::-1]
    ypos = range(len(vals))

    axl.axvspan(-2 * SE, 2 * SE, color="0.86", alpha=0.55, linewidth=0, zorder=0)
    axl.annotate("$\\pm2$ SE\n(across seeds)", xy=(2 * SE, -0.42), xytext=(5, 0),
                 textcoords="offset points", fontsize=7, color=MUTED, va="center")
    axl.barh(list(ypos), vals, height=0.52, color=C_TREAT, zorder=2)
    for y, v in zip(ypos, vals):
        axl.annotate(f"+{v:.1f} pp", xy=(v, y), xytext=(7, 0), textcoords="offset points",
                     va="center", fontsize=8.5, color=INK, fontweight="bold")

    axl.set_yticks(list(ypos))
    axl.set_yticklabels(labels, fontsize=8)
    axl.set_xlim(-5, 38)
    axl.set_xticks([0, 10, 20, 30])
    axl.set_ylim(-0.95, 3.6)
    axl.set_xlabel("Contribution to collision-free rate (pp)", fontsize=8.5)
    axl.tick_params(labelsize=8)
    axl.grid(axis="x", color="0.9", linewidth=0.7)
    _despine(axl, keep=("bottom",))
    axl.tick_params(axis="y", length=0)
    axl.set_title("(a)  What each mechanism is worth", fontsize=9, loc="left",
                  color=INK, pad=6)

    # -- right: the residual falls with scale ------------------------------
    axr.plot(RESID_ENVS, RESID, "-o", color=C_TREAT, linewidth=2, markersize=6,
             zorder=3)
    axr.plot([RESID_NOROLL[0]], [RESID_NOROLL[1]], "o", markerfacecolor="white",
             markeredgecolor=C_TREAT, markeredgewidth=1.6, markersize=7, zorder=4)
    axr.annotate("no roll aug.", xy=RESID_NOROLL, xytext=(8, -2),
                 textcoords="offset points", fontsize=7.5, color=MUTED)
    for x, y in zip(RESID_ENVS, RESID):
        axr.annotate(f"{100*y:.2f}%", xy=(x, y), xytext=(0, 9),
                     textcoords="offset points", fontsize=8, color=C_TREAT,
                     ha="center", fontweight="bold")

    axr.set_xscale("log")
    axr.set_xticks(RESID_ENVS)
    axr.set_xticklabels([str(e) for e in RESID_ENVS])
    axr.minorticks_off()
    axr.set_xlim(15, 400)
    axr.set_ylim(0, 0.034)
    axr.set_yticks([0, 0.01, 0.02, 0.03])
    axr.set_yticklabels(["0", "1%", "2%", "3%"])
    axr.set_xlabel("Training environments", fontsize=8.5)
    axr.set_ylabel("Non-equivariance residual $r$", fontsize=8.5)
    axr.tick_params(labelsize=8)
    axr.grid(axis="y", color="0.9", linewidth=0.7)
    _despine(axr)
    axr.set_title("(b)  The residual is flat", fontsize=9, loc="left",
                  color=INK, pad=6)

    fig.tight_layout(pad=0.4, w_pad=2.2)
    fig.savefig(out)
    print(f"wrote {out}")


def fig_baselines(out="fig_baselines.pdf"):
    """Where the learned planner actually sits. Success against wall-clock,
    with the straight-line floor drawn as a reference: the world-frame arm is
    on it, which is the finding, and the classical planners are far above."""
    fig, ax = plt.subplots(figsize=(3.35, 2.75))

    #the trivial baseline is a threshold, not a competitor -- draw it as a rule
    ax.axhline(STRAIGHT_LINE, color=MUTED, linewidth=1, linestyle=(0, (4, 3)),
               zorder=1)
    ax.annotate(f"straight line ({STRAIGHT_LINE}%)", xy=(1.35, STRAIGHT_LINE),
                xytext=(0, -9), textcoords="offset points", fontsize=6.5,
                color=MUTED, ha="right")

    #hand-placed so no label touches another; the points are too clustered in
    #x for any single rule to work
    PLACE = {
        "RRT-Connect":        (-9, 3, "right", "bottom"),
        "expert (RRT+CHOMP)": (9, 2, "left", "bottom"),
        "CHOMP":              (9, 0, "left", "center"),
        "TrajOpt":            (9, 0, "left", "center"),
        "world frame":        (9, -6, "left", "top"),
        "world frame + aug.": (9, 3, "left", "bottom"),
        r"$(s,g)$ reduction": (9, 0, "left", "center"),
    }

    xs = [t for _, _, t in CLASSICAL]
    ys = [s for _, s, _ in CLASSICAL]
    ax.plot(xs, ys, "s", color=C_CTRL, markersize=6, zorder=3,
            label="classical (1 CPU core)")
    xs = [t for _, _, t in LEARNED]
    ys = [s for _, s, _ in LEARNED]
    ax.plot(xs, ys, "o", color=C_TREAT, markersize=7, zorder=3,
            label="learned (1 GPU, 20 samples)")

    for name, s, t in CLASSICAL + LEARNED:
        dx, dy, ha, va = PLACE[name]
        ax.annotate(name, xy=(t, s), xytext=(dx, dy), textcoords="offset points",
                    fontsize=6.5, color=INK, ha=ha, va=va)

    #the gap the paper closes, and the gap it does not
    ax.annotate("", xy=(0.067, 45.63), xytext=(0.067, 15.38),
                arrowprops=dict(arrowstyle="<->", color=C_TREAT, linewidth=1.1,
                                shrinkA=3, shrinkB=3))
    ax.annotate("+30.2", xy=(0.067, 30), xytext=(-4, 0),
                textcoords="offset points", fontsize=7, color=C_TREAT,
                ha="right", va="center", fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlim(0.035, 2.6)
    ax.set_xticks([0.05, 0.1, 0.3, 1.0])
    ax.set_xticklabels(["0.05", "0.1", "0.3", "1.0"])
    ax.minorticks_off()
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Wall-clock per query (s, log scale)", fontsize=7.5)
    ax.set_ylabel("Collision-free rate (%)", fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", color="0.9", linewidth=0.7)
    _despine(ax)
    #empty mid-left region; anywhere lower collides with the world-frame point
    ax.legend(frameon=False, fontsize=6.5, loc="center left", handletextpad=0.4,
              bbox_to_anchor=(-0.02, 0.66))

    fig.tight_layout(pad=0.4)
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig_scaling()                                        # main.tex, 0.82\textwidth
    fig_scaling("fig_scaling_col.pdf", size=(3.35, 2.45), fs=7.0)   # paper.tex, one column
    fig_mechanisms()                                     # paper.tex, figure* (both columns)
    fig_baselines()                                      # paper.tex, one column
