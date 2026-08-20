
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
#The figures are generated from HARD-CODED constants in make_figures.py, not
#from the results files, so they drift independently of the prose and this
#checker could not see them. That gap was real: the scaling figure plotted
#single-seed values for several revisions after the tables had moved to
#three-seed means, and no amount of checking the .tex would have caught a
#figure that disagreed with the table beside it.
FIGS = pathlib.Path("make_figures.py").read_text()

#Values that must be present somewhere. Not exhaustive -- these are the ones
#that appear in several places and so drift apart.
REQUIRED = {
    "reduction at 250 envs (3-seed mean)": r"45\.5",
    "world frame at 250 envs": r"15\.4",
    "augmented control at 250 envs": r"17\.5",
    "headline gap (3-seed)": r"\+30\.08|30\.1",
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
    "probe: world-frame scene code is constant": r"0\.000000",
    "probe: reduced scene code predicts clearance": r"\+0\.778",
    "probe: world-frame conditioning cannot": r"\+0\.07",
    "ddpm gap at its own best NFE": r"\+16\.71|\+16\.7",
    "ddpm gap at 8 steps": r"\+14\.32",
    "seed-matched gap at 20 envs": r"\+6\.37",
    "seed-matched gap at 150 envs": r"\+25\.85",
    "seed-matched gap at 250 envs": r"\+30\.08",
    "converged at 250: world frame": r"14\.4",
    "converged at 250: reduction": r"55\.8",
    "converged gap at 250": r"41\.4",
    "frame averaging at convergence": r"\+1\.57",
    "cone sweep raises r": r"0\.0238",
    "density: control tracks the line at 8 obstacles": r"61\.4",
    "density: gap peaks at 24 obstacles": r"\+41\.2",
    "converged at 60, 3 seeds: world frame": r"14\.60",
    "converged at 60, 3 seeds: reduction": r"51\.10",
    "converged gap at 60, 3 seeds": r"\+36\.5",
    "frame averaging at convergence, 3 seeds": r"\+0\.68",
    "DDPM converged gap": r"\+37\.0",
    "cone decoupling failed": r"attempted decoupling that\s*\n?\\emph\{failed\}",
    #--- the constrained architecture, measured 2026-08-18 -------------------
    "equivariant backbone at convergence": r"\+0\.15",
    "equivariant backbone, epoch 10": r"\+13\.14",
    "equivariant parameter count": r"2\{,\}582\{,\}233",
    "unconstrained parameter count": r"2\{,\}161\{,\}283",
    "equivariant capacity is LARGER": r"19\.5\\%",
    "equivariance verified untrained": r"1\.2\\times10\^\{-6\}",
    "equivariance test is not vacuous": r"5\.03",
    #--- local geometry ------------------------------------------------------
    "local geometry, world frame": r"54\.54",
    "local geometry, reduction": r"79\.37",
    "local geometry + equivariant": r"82\.13",
    "geometry beats the reduction": r"\+40\.0",
    "reduction survives a good encoder": r"\+24\.8",
    "sub-additive": r"91\.1",
    "equivariance pays only with geometry": r"\+2\.8",
}

#Figure constants that must match the tables. Checked separately because they
#live in Python rather than LaTeX, and a mismatch here is invisible to a reader
#who trusts the figure over the table -- or worse, to a reviewer who spots it.
FIGURE_REQUIRED = {
    "scaling curve: control is 3-seed means": r"12\.86, 13\.57, 14\.75, 15\.37",
    "scaling curve: treatment is 3-seed means": r"19\.23, 34\.43, 40\.60, 45\.45",
    "scaling curve carries seed spreads": r"TREATMENT_SE\s*=",
    "no-roll ablation is the 3-seed mean": r"NOROLL_X, NOROLL_Y = 60, 33\.53",
    "mechanism bar matches the ablation table": r"\(5 DOF, exact\)\", 30\.1\)",
    #the convergence curve is the only place the 4x efficiency claim is shown
    "convergence curve: equivariant arm starts at 35.8": r"CONV_EQUIV = \[35\.8",
    "convergence curve: unconstrained arm starts at 22.2": r"CONV_TREAT = \[22\.2",
    "convergence curve carries the read-off marks": r"CONV_MARKS = \[\(10, 40\), \(40, 90\)\]",
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
    r"until it is measured the possibility": "the DDPM arm has now been run; the gap survives at +16.7+-0.4",
    r"are inert; the roll": "frame averaging is +0.9 at convergence, not inert",
    r"indistinguishable from joining the endpoints": "at convergence the control is BELOW the straight line, not indistinguishable from it",
    r"rests on one seed per arm": "all four scales now have three seeds; the 250-env gap is +30.08+-0.38",
    r"running at the time of writing": "the convergence study finished; both arms plateau (15.0 vs 50.8)",
    r"not isotropic in the plane normal": "false; an i.i.d. bridge has covariance "
                                          "sigma^2 g(1-g) I and is roll-invariant. The real "
                                          "cost is the singular endpoint covariance",
    #--- retired when the equivariant backbone was actually built ------------
    r"untested rather than as unpromising": "the constrained architecture has been "
                                            "built and measured; see Sec. arch",
    r"largest single gap in this work": "same -- the gap is closed",
    r"explicitly cannot make": "the comparison was made",
    r"The two mechanisms built for the sixth": "there are three now, all measured",
    r"honest answer is currently ``untested''": "it is measured",
    #--- retired when the local-geometry arms got their second seed ----------
    r"\+35\.8": "seed-0 value; the multi-seed reduction gap is +36.5",
    r"\+39\.5": "seed-0 value; the multi-seed geometry gap is +39.9",
    r"All cells are\s*\n?seed \$0\$": "the local-geometry arms now carry two seeds each",
    r"right-hand column is currently a single seed": "only the equivariant cell is",
    #--- claims broader than the propositions actually establish -------------
    r"provably not removable": "Prop. 2 rules out CONTINUOUS gauges and Prop. 3 is "
                              "an expectation under a symmetric obstacle law; "
                              "neither forbids a discontinuous rule",
    r"sixth degree of freedom is not removable": "same overreach",
    r"does not benchmark a constrained backbone": "Sec. arch benchmarks one",
    r"fourfold": "the saving is 2.2x-3.2x on multi-seed means, not 4x; the 4x came from seed 0 alone",
    r"quarter of the epochs": "same overstatement in words",
}

def main():
    bad = []
    for name, pat in REQUIRED.items():
        if not re.search(pat, TEX):
            bad.append(f"MISSING  {name}  (/{pat}/)")
    for name, pat in FIGURE_REQUIRED.items():
        if not re.search(pat, FIGS):
            bad.append(f"FIGURE   {name}  (/{pat}/ in make_figures.py)")
    for fname, text in (("paper.tex", TEX), ("main.tex", REPORT),
                        ("make_figures.py", FIGS)):
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
