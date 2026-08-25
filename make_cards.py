#Title and closing cards for the supplementary video.
#
#  .venv/bin/python make_cards.py --title "..." --out-title t.mp4 --out-end e.mp4
#
#No author names on either card, so the video is safe to attach to a blind
#submission. Sized to match the sampling shot exactly (1400x756 at 24 fps) so
#ffmpeg concat needs no scaling.

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

INK = "#141C24"
MUTED = "#6B7683"
FREE = "#C2560F"
CTRL = "#3E6DA8"


def card(path, draw, seconds=3.0, fps=24, w=1400, h=756):
    fig = plt.figure(figsize=(w / 140, h / 140), dpi=140)
    fig.patch.set_facecolor("white")
    draw(fig)
    n = int(seconds * fps)
    anim = animation.FuncAnimation(fig, lambda i: [], frames=n, interval=1000 / fps)
    anim.save(path, writer=animation.FFMpegWriter(fps=fps, bitrate=3200))
    plt.close(fig)
    print(f"wrote {path} ({seconds:.0f} s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default="What Symmetry Buys a Learned Motion Planner")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--out-title", default="card_title.mp4")
    ap.add_argument("--out-end", default="card_end.mp4")
    a = ap.parse_args()

    def title(fig):
        fig.text(0.5, 0.56, a.title, ha="center", va="center",
                 fontsize=26, color=INK, wrap=True)
        fig.text(0.5, 0.42, "supplementary video", ha="center", va="center",
                 fontsize=13, color=MUTED)

    def end(fig):
        fig.text(0.5, 0.80, "held-out collision-free rate", ha="center",
                 fontsize=15, color=MUTED)
        rows = [("world frame", "14.60", CTRL),
                ("straight line from start to goal", "15.6", MUTED),
                ("$(s,g)$ reduction", "51.10", FREE)]
        for i, (name, val, col) in enumerate(rows):
            y = 0.60 - 0.14 * i
            fig.text(0.46, y, name, ha="right", va="center", fontsize=19, color=col)
            fig.text(0.54, y, val + r"$\,\%$", ha="left", va="center",
                     fontsize=19, color=col)
        fig.text(0.5, 0.16, "same architecture, same data, same training budget",
                 ha="center", fontsize=13, color=MUTED)

    card(a.out_title, title, a.seconds)
    card(a.out_end, end, a.seconds)


if __name__ == "__main__":
    main()
