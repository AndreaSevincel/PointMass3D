#.venv/bin/python section_map.py [paper.tex]

#What each section costs and what depends on it, for cutting deliberately.

#A naive squeeze to six pages deletes the scaffolding that earns the claims: the
#corrections, the "an earlier version of this paper" passages, the null results.
#What survives should be a decision, and a decision needs the cost and the
#dependencies in front of you. This prints, per section: estimated page cost,
#which labels it defines, how many times the rest of the paper points AT it, and
#which headline numbers it owns.

import pathlib
import re
import sys

VALUES = ["14.60", "15.6", "36.5", "45.5", "51.10", "51.1", "55.8", "40.0",
          "24.8", "79.37", "82.13", "54.54", "15.59", "1.01", "0.15", "0.68"]
#Calibrated against the actual build rather than guessed: at 127 source lines
#per page the estimate totals the 20 pages pdfinfo reports for the current
#paper.tex. Tables and figures cost more than their source length suggests, so
#they carry a separate allowance. Recalibrate if the class or margins change.
LINES_PER_PAGE = 127


def main(path="paper.tex"):
    text = pathlib.Path(path).read_text()
    lines = text.splitlines()
    heads = [(i, m.group(1), m.group(2))
             for i, l in enumerate(lines)
             if (m := re.match(r"\\(sub)?section\*?\{(.+?)\}", l))]
    all_refs = re.findall(r"\\ref\{([^}]+)\}", text)

    print(f"{'sec':<4} {'pp':>4} {'tab':>4} {'fig':>4} {'in':>4}  section")
    print("-" * 78)
    total = 0.0
    for k, (start, level, title) in enumerate(heads):
        end = heads[k + 1][0] if k + 1 < len(heads) else len(lines)
        body = "\n".join(lines[start:end])
        pages = (end - start) / LINES_PER_PAGE
        tabs = body.count(r"\begin{table}") + body.count(r"\begin{table*}")
        figs = body.count(r"\begin{figure}") + body.count(r"\begin{figure*}")
        pages += 0.28 * tabs + 0.32 * figs
        total += pages
        mine = set(re.findall(r"\\label\{([^}]+)\}", body))
        incoming = sum(all_refs.count(l) for l in mine)
        owns = [v for v in VALUES if re.search(r"(?<![\d.])" + re.escape(v) + r"(?![\d])", body)]
        mark = " " if level else ">"
        print(f"{mark}{k:<3} {pages:>4.1f} {tabs:>4} {figs:>4} {incoming:>4}  "
              f"{title[:44]}")
        if owns:
            print(f"{'':>23}owns: {', '.join(owns[:8])}")
    print("-" * 78)
    print(f"     {total:>4.1f} estimated pages\n")
    print("in = times the rest of the document \\ref's a label defined here.")
    print("A section with high 'in' cannot be cut without rewriting its dependants.")
    print("A section with 0 'in' and no owned values is the cheapest thing to lose.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "paper.tex"))
