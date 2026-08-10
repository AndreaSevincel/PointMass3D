
  #python check_paper_numbers.py

  #Verify the paper's numbers against the measured results, mechanically.

  #Every number below was hand-propagated across a dozen revisions, and each
  #revision left some of them stale. Reading for it does not work: the same
  #figure appears in the abstract, an intro contribution, a table, a caption and
  #a discussion paragraph, and a reader checks whichever they are looking at.
  #This checks all of them at once, and fails loudly on values that were
  #superseded.

import pathlib
import re
import sys

#paper.tex is the submission; main.tex is the internship report of the same work.
#They share every measured number, and they have drifted apart before -- main.tex
#carried +1.2/+0.2 and "~11 standard errors" for several revisions after paper.tex
#had superseded both. REQUIRED is checked against the submission, since it is the
#document whose claims are load-bearing; FORBIDDEN is checked against BOTH, because
#a retired number is retired everywhere.
TEX = pathlib.Path("paper.tex").read_text()
REPORT = pathlib.Path("main.tex").read_text()

#Values that must be present somewhere. Not exhaustive -- these are the ones
#that appear in several places and so drift apart.
REQUIRED = {
    "reduction at 250 envs": r"45\.6",
    "world frame at 250 envs": r"15\.4",
    "augmented control at 250 envs": r"17\.5",
    "headline gap": r"30\.2",
    "roll augmentation": r"\+0\.8",
    "frame averaging": r"\+0\.1",
    "SE(3) augmentation": r"\+2\.1|2\.1 pp",
    "epoch gain, reduced arm": r"\+8\.0",
    "epoch gain, world frame": r"\+0\.9|0\.87",
    "interaction": r"\+7\.1",
    "translational DOF": r"\+12\.3",
    "rotational DOF": r"\+18\.0",
    "best-of-20": r"90\.4",
    "straight-line floor": r"15\.6",
    "untrained prior best-of-20": r"15\.8",
    "r under SE(3), world frame": r"0\.0881",
    "r under SE(3), augmented": r"0\.0181",
    "r under SE(3), reduction": r"0\.0174",
    "world frame is 91% equivariant": r"91\\%",
    "symmetrisation budget stated as a bound": r"under\s*\n?\$3\\times10\^\{-4\}",
    "seed-matched gap at 60 envs": r"\+20\.9",
    "best-of-20 interaction": r"\+10\.7",
    "world frame loses best-of-20 with 3x budget": r"-3\.1|\$39\.2\$",
    "SE(3) domain, world frame": r"\$3\.0\$|3\.0\%",
    "SE(3) domain, reduction": r"\$8\.6\$|8\.6\%",
    "SE(3) trivial floor": r"4\.8\\%",
    "cross-domain ratio": r"2\.9\\times",
}

#Values superseded by a later measurement. Their presence is a bug.
FORBIDDEN = {
    r"\+11\.1": "epoch gain against the single-seed baseline; use +8.0",
    r"0\.0157": "residual under the old mean-of-norms definition",
    r"1\.86\\%": "residual trend that the four-tier measurement refuted",
    r"\+30\.3": "gap; 45.6 - 15.4 = 30.2",
    r"both inside noise": "roll augmentation is +4.4, not inside noise",
    r"are inert\}": "heading; only the K sweep is inert",
    r"three independent measurements": "the roll ablation no longer supports it",
    r"know in advance": "framed r as predictive; Sec. V-F refutes it",
    r"before\}?\s*\n?an equivariant model is built": "same, in the abstract",
    r"is worth none": "the roll ablation is unresolved, not zero",
    r"All of the measured benefit lies in the five": "same",
    r"constraint that is already satisfied": "the world-frame arm satisfies it too",
    r"\+30\.3": "gap label; 45.6 - 15.4 = 30.2",
    r"We no longer make that argument": "state-then-retract; cut the first claim",
    r"r=0\.016\$": "r is 0.017 (0.0168-0.0174); pick one",
    r"2\.5\\times10\^\{-4\}": "0.0168^2 = 2.8e-4, not 2.5e-4",
    r"case for not building a constrained architecture rests": "the retracted inference",
    r"Two independent measurements support it": "both bear on frame averaging only",
    r"\+4\.4": "roll augmentation is +0.8 seed-matched; +4.4 was an unpaired artefact",
    r"one seed, see text": "the no-roll ablation now has three seeds",
    r"roughly \$?19\$? standard\s*\n?errors": "applies the 60-env SE to a single-seed "
                                             "250-env difference; quote +20.9+-1.6 at 60 instead",
    r"\\approx19 SE": "same, in the ablation table",
    r"\$\\approx 11\$ standard": "the 11-SE form of the same invalid test",
    r"not isotropic in the plane normal": "false; an i.i.d. bridge has covariance "
                                          "sigma^2 g(1-g) I and is roll-invariant. The real "
                                          "cost is the singular endpoint covariance",
}

def main():
    bad = []
    for name, pat in REQUIRED.items():
        if not re.search(pat, TEX):
            bad.append(f"MISSING  {name}  (/{pat}/)")
    for fname, text in (("paper.tex", TEX), ("main.tex", REPORT)):
        for pat, why in FORBIDDEN.items():
            for m in re.finditer(pat, text):
                line = text[:m.start()].count("\n") + 1
                #a deliberate contrast, flagged in prose, is allowed
                ctx = text[max(0, m.start() - 220):m.start() + 60]
                if any(k in ctx for k in ("would have reported", "would read",
                                          "would put", "reported roll augmentation as")):
                    continue
                bad.append(f"STALE    {fname}:{line}: {m.group()}  -- {why}")
    for b in bad:
        print(b)
    print(f"\n{len(bad)} problem(s)" if bad else "\nall numbers consistent")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
