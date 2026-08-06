
  #.venv/bin/python make_figures.py

  #Regenerate the report figures from the measured numbers. Kept as a script so
  #the figures track the results rather than drifting from them.

  #Source: sweep_steps.py on held-out envs 250-299 (50 envs x 10 pairs x
  #20 samples, 8 Euler steps, K_FA=1). See main.tex Section 4 / paper.tex Section VI.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ENVS = [20, 60, 150, 250]
CONTROL = [12.8, 13.5, 14.6, 15.4]      # world frame
TREATMENT = [19.0, 31.3, 41.0, 45.6]    # (s,g) reduction
NOROLL_X, NOROLL_Y = 60, 30.1           # ablation: reduction without roll augmentation

# mechanism decomposition, percentage points on the held-out collision-free rate
MECHANISMS = [
    (r"$(s,g)$ reduction" "\n" "(5 DOF, exact)", 30.2),
    ("roll augmentation\n(training)", 1.2),
    ("frame averaging\n($K{=}1\\to9$)", 0.2),
]
SE = 2.0                                # standard error, ~500 distinct problems

# non-equivariance residual  E||v_k - vbar|| / ||vbar||, K=3
RESID_ENVS = [60, 250]
RESID = [0.0186, 0.0157]
RESID_NOROLL = (60, 0.0292)

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

    # the no-roll ablation is the same arm with one part removed: same hue, hollow
    ax.plot([NOROLL_X], [NOROLL_Y], "o", markerfacecolor="white",
            markeredgecolor=C_TREAT, markeredgewidth=1.6, markersize=7, zorder=4)
    ax.annotate("no-roll ablation", xy=(NOROLL_X, NOROLL_Y), xytext=(9, -9),
                textcoords="offset points", fontsize=fs - 1.0, color="#555555")

    for x, y, c in ((250, 45.6, C_TREAT), (250, 15.4, C_CTRL)):
        ax.annotate(f"{y}%", xy=(x, y), xytext=(-4, 6), textcoords="offset points",
                    fontsize=fs, color=c, fontweight="bold", ha="right")

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
    axl.annotate("$\\pm2$ SE\n(noise)", xy=(2 * SE, 2.42), xytext=(6, 0),
                 textcoords="offset points", fontsize=7.5, color=MUTED, va="center")
    axl.barh(list(ypos), vals, height=0.52, color=C_TREAT, zorder=2)
    for y, v in zip(ypos, vals):
        axl.annotate(f"+{v} pp", xy=(v, y), xytext=(7, 0), textcoords="offset points",
                     va="center", fontsize=8.5, color=INK, fontweight="bold")

    axl.set_yticks(list(ypos))
    axl.set_yticklabels(labels, fontsize=8)
    axl.set_xlim(-5, 38)
    axl.set_xticks([0, 10, 20, 30])
    axl.set_ylim(-0.6, 2.75)
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
        axr.annotate(f"{100*y:.2f}%", xy=(x, y), xytext=(0, -14),
                     textcoords="offset points", fontsize=8, color=C_TREAT,
                     ha="center", fontweight="bold")

    axr.set_xscale("log")
    axr.set_xticks(RESID_ENVS)
    axr.set_xticklabels([str(e) for e in RESID_ENVS])
    axr.minorticks_off()
    axr.set_xlim(45, 340)
    axr.set_ylim(0, 0.034)
    axr.set_yticks([0, 0.01, 0.02, 0.03])
    axr.set_yticklabels(["0", "1%", "2%", "3%"])
    axr.set_xlabel("Training environments", fontsize=8.5)
    axr.set_ylabel("Non-equivariance residual $r$", fontsize=8.5)
    axr.tick_params(labelsize=8)
    axr.grid(axis="y", color="0.9", linewidth=0.7)
    _despine(axr)
    axr.set_title("(b)  The residual falls with data", fontsize=9, loc="left",
                  color=INK, pad=6)

    fig.tight_layout(pad=0.4, w_pad=2.2)
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig_scaling()                                        # main.tex, 0.82\textwidth
    fig_scaling("fig_scaling_col.pdf", size=(3.35, 2.45), fs=7.0)   # paper.tex, one column
    fig_mechanisms()                                     # paper.tex, figure* (both columns)
