#.venv/bin/python audit_numbers.py [--values 45.5,51.1] [--context 1]

#Every occurrence of every headline number, so none can go stale unnoticed.

#check_paper_numbers.py answers "does this value appear somewhere" and "has a
#retired value come back". It cannot answer the question that has now gone wrong
#four rounds running: when a number changes, WHERE ELSE does it live? A value
#moves, the section that owns it is updated, and a caption or a figure label two
#pages away keeps the old one. Compression makes this worse, because it
#multiplies cross-references while deleting the surrounding text that would have
#made a stale number obvious.
#
#So: print every hit, with its line and enough context to classify it, and make
#the human decide intentional-or-stale for each. Run it after every number
#change and again after the cut.

import argparse
import pathlib
import re

#The values that carry the paper's argument and therefore appear in several
#places each. Add to this list whenever a number becomes load-bearing.
DEFAULT_VALUES = [
    "14.6", "14.60",     # world frame, converged, 60 envs
    "15.4",              # world frame, 20 epochs, 250 envs
    "15.6",              # straight-line floor
    "30.1", "30.2",      # gap at 20 epochs, 250 envs
    "36.5",              # gap converged, 60 envs -- the headline
    "45.5",              # reduction, 20 epochs, 250 envs
    "51.1", "51.10",     # reduction, converged, 60 envs -- the headline
    "55.8",              # reduction, converged, 250 envs, single seed
]
SOURCES = ["paper.tex", "main.tex", "make_figures.py"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", help="comma-separated, overrides the default set")
    ap.add_argument("--context", type=int, default=1,
                    help="lines of context either side")
    ap.add_argument("--files", help="comma-separated, overrides the default set")
    args = ap.parse_args()

    values = args.values.split(",") if args.values else DEFAULT_VALUES
    files = args.files.split(",") if args.files else SOURCES

    total = 0
    for value in values:
        #\D or start/end, so 45.5 does not match inside 145.53 or 45.55
        pat = re.compile(r"(?<![\d.])" + re.escape(value) + r"(?![\d])")
        hits = []
        for name in files:
            path = pathlib.Path(name)
            if not path.exists():
                continue
            lines = path.read_text().splitlines()
            for i, line in enumerate(lines):
                if pat.search(line):
                    hits.append((name, i + 1, lines, i))
        if not hits:
            continue
        print(f"\n{'=' * 72}\n{value}  --  {len(hits)} occurrence(s)\n{'=' * 72}")
        for name, lineno, lines, i in hits:
            lo, hi = max(0, i - args.context), min(len(lines), i + args.context + 1)
            print(f"\n  {name}:{lineno}")
            for j in range(lo, hi):
                mark = ">>" if j == i else "  "
                print(f"  {mark} {lines[j].rstrip()[:110]}")
        total += len(hits)
    print(f"\n{total} occurrences across {len(values)} values. "
          f"Classify each as intentional-and-labelled, or stale.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
