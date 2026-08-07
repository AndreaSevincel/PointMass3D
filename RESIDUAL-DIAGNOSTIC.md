# Validating the non-equivariance residual as a model-selection diagnostic

**Scope warning up front: this is a follow-up paper, not part of the current
submission.** The current paper can honestly claim that `r` bounds what
*symmetrising a given model* buys (Corollary 5, which is a theorem) and can
support it with the cone dial. It cannot claim `r` predicts what an *equivariant
architecture* buys, because that requires training equivariant architectures,
which the paper explicitly declines to do. Do not let this document's ambition
leak into the submission's claims.

## The claim worth testing

Not "does `r` correlate with performance" -- that is nearly vacuous. The claim is:

> `r`, measured on a cheap unconstrained model, predicts the marginal benefit of
> imposing equivariance -- by architecture, not merely by post-hoc averaging.

If true across task families and architectures, `r` is a model-selection
diagnostic: run one unconstrained model, measure `r`, and decide whether the
constrained architecture is worth building. That is a general ML claim, not a
robotics one.

## Per-setting measurements

For each setting, record all five. The first four are the columns of the
analysis; the fifth is what makes the recommendation actionable.

| # | quantity | how |
|---|---|---|
| 1 | `r` of the unconstrained model | `sweep_steps.py --k-fa 3 --residual` (RMS form, Eq. 6) |
| 2 | performance, unconstrained | held-out collision-free, `--k-fa 1` |
| 3 | performance after group averaging | same checkpoint, `--k-fa 9` |
| 4 | performance, explicitly equivariant architecture | **not built yet** -- see below |
| 5 | cost of the constraint | wall-clock per epoch, params, and implementation time |

The gap that matters is **(4) − (2)**, and the question is whether `r` predicts
it. Note that (3) − (2) is already bounded by `r` as a matter of theory, so it
tests nothing; only (4) is informative, because an equivariant architecture
searches a different hypothesis class and is not bounded by Corollary 5.

## The confound that decides whether this works

`r` conflates two situations that call for opposite decisions:

- **A. the task is equivariant, the model has not learned it.** Enforcing
  equivariance should help. Expect large `r`, large (4) − (2).
- **B. the task is not equivariant in the given representation.** Enforcing
  equivariance should hurt, or at best do nothing. Expect large `r`, and
  (4) − (2) ≤ 0.

If the study contains only situation A, a positive result is unfalsifiable
decoration: `r` was never at risk of being wrong. **At least one setting must be
of type B**, or the diagnostic is not shown to discriminate.

Constructing B here is not trivial, because the reduced-frame task is *exactly*
SO(2)-equivariant for any obstacle distribution -- roll is a gauge, and
Proposition 1 guarantees it. Changing the scene statistics does not break it.
What does break it is making the task depend on something the reduced frame
throws away:

> Add an anisotropic cost tied to world "up" -- a slope limit, or a penalty on
> altitude gain -- and **withhold the up-vector from the conditioning**. The
> optimal field is then genuinely not a function of the reduced representation.
> `r` is large for an irreducible reason and no equivariant architecture can
> recover it.

Then re-run the same setting *with* the up-vector supplied (rotated correctly
into the reduced frame, as a free vector). The task becomes equivariant again
and equivariance should pay. Same task, same data, one conditioning variable --
a clean A/B pair sharing everything else.

## Settings axis

1. **Cone dial** (implemented: `--xhat-cone`, `--subsample`). Varies how much
   roll diversity the training data supplies, hence how equivariant the learned
   field becomes. Type A at every width. Gives the dose-response curve.
2. **Withheld up-vector** (not built). Type B. The discriminating case.
3. **SE(3) rigid body** (`se3body/`, dataset generating). Different state space,
   same symmetry structure. Tests transfer of the relationship, not just of the
   conclusion.
4. **Second objective** (`--objective ddpm`, implemented, untrained). Tests
   whether the relationship is a property of the problem or of flow matching.

Publishable-beyond-robotics needs at least axes 1, 2 and one of 3/4.

## The missing instrument: an SO(2)-equivariant backbone

This is the long pole and the reason the study is a follow-up. After the
reduction the residual group is SO(2) (roll about `x̂`), so the required
architecture is modest -- far cheaper than a full SE(3) network:

- a reduced-frame waypoint splits into an **m=0 scalar** (the `x` component,
  roll-invariant) and an **m=1 vector** (the `(y,z)` components, which rotate);
- obstacle features split the same way (centres, and each box edge vector);
- equivariant linear maps: scalar→scalar arbitrary; vector→vector by scalar
  multiples; vector→scalar via norms; scalar→vector by scaling;
- nonlinearity: gate the vectors by a learned scalar function of their norms and
  the scalars, which keeps equivariance exactly.

Verification must mirror `test_geometry.py`: assert the network commutes with a
roll *for random untrained weights*, so the property is structural rather than
learned.

## Build order

1. SO(2)-equivariant backbone + equivariance unit tests (untrained-network check).
2. Run it across the cone dial; that alone gives (4) at four `r` values.
3. Build the withheld-up-vector setting; run both arms of the A/B pair.
4. Extend to the SE(3) domain and/or the DDPM objective.
5. Only then: the model-selection claim.

## What counts as the result

A scatter of (4) − (2) against `r`, across settings, with type-B settings
marked. The diagnostic is validated if the type-A points show a monotone
relationship **and** the type-B points sit near or below zero despite large `r`
-- i.e. `r` bounds the available gain, and some second signal (whether the task
is equivariant in the given representation) is needed to know whether the bound
is attained. Reporting the failure of `r` alone to separate A from B would be a
more honest and more useful result than a clean correlation.
