# Physics-informed point-set registration — findings

Work on the codebase accompanying *Biomechanics-informed Non-rigid Medical Image
Registration with Elasticity Theories* (Min et al., IEEE TMI 2026).

Everything below is measured, with code and raw results in this repo. Section 6
lists what is **not** established, which matters as much as the rest.

---

## Headline

Two things, and the second is the substantive one.

**No modification improved the published configuration.** Across a 15× range of
model size, three activations, two pooling schemes, three bottleneck widths and
two elasticity theories, every change either made results worse or fell below
the measured noise floor. ReLU / max-pool / trunk 1024 / dim_k 1024 / linear
elasticity — the settings inherited from PointNet (2017) and never justified in
the paper — beat or matched everything tested against them.

**The physics loss does not compute physics.** The strain term is contaminated
525× by cross-point coupling and correlates −0.015 with the true field gradient.
The shipped `Exx` is *exactly zero at 71% of points*, and the nonzero set is
precisely the points that win a max-pool channel — it is a mask, not a strain
field. Correcting it makes the measurement right and makes training fail, which
means the wrong quantity was doing useful optimisation work.

---

## 1. The noise floor (measure this before believing anything else)

Identical configuration, identical data, only the random initialisation differs:

| condition | non-affine | linear spread | nonlinear spread |
|---|---|---|---|
| `uniform_pressure` | 0.000 | **6.26×** | 1.11× |
| `two_zone_material` | 0.464 | 1.45× | **3.90×** |
| `two_point_poles` | 0.769 | 1.04× | 1.07× |
| `four_points_poles_equator` | 0.885 | 1.02× | 1.01× |

Results vary by up to **6× from the random seed alone**. Instability is
condition-dependent, not model-dependent — linear is wild on uniform pressure,
nonlinear is wild on the two-zone material. The two unstable cases have obvious
causes: a stiffness discontinuity at r = 0.5 in one, and a problem so easy the
optimiser stops in inconsistent places in the other.

**Every architecture measurement below was taken on `two_point_poles`, where the
floor is 4–7%.** That is the bar those numbers must clear.

---

## 2. Architecture: only the activation matters

Ten runs on `two_point_poles`, scored by RMSE against ground-truth displacement
on a held-out sample. Predict-zero baseline 8.334e-03.

| trunk | dim_k | act | pool | params | seen | **unseen** |
|---|---|---|---|---|---|---|
| 1024 | 1024 | relu | max | 25.2M | 1.377e-03 | **1.644e-03** |
| 1024 | 512 | relu | max | 17.7M | 1.310e-03 | 1.671e-03 |
| 1024 | 1024 | relu | attention | 25.2M | 1.292e-03 | 1.694e-03 |
| 2048 | 1024 | relu | max | 54.5M | 1.278e-03 | 1.708e-03 |
| 1024 | 2048 | relu | max | 40.1M | 1.529e-03 | 1.709e-03 |
| 512 | 1024 | relu | max | 14.2M | 1.771e-03 | 1.808e-03 |
| 4096 | 1024 | relu | max | 142.5M | 1.386e-03 | 1.845e-03 |
| 256 | 1024 | relu | max | 9.6M | 1.795e-03 | 1.906e-03 |
| 1024 | 1024 | **sin** | max | 25.2M | 3.953e-03 | 3.592e-03 |
| 1024 | 1024 | **tanh** | max | 25.2M | 4.328e-03 | 3.942e-03 |

Graded against the 7% floor:

```
activation (relu vs tanh)   140%   >> 7%   REAL
trunk width (9.6M-142M)      16%   >  7%   probably real
dim_k (512-2048)              4%   <  7%   NOT distinguishable
pooling (max vs attention)    3%   <  7%   NOT distinguishable
```

`sin` losing by 2.2× is the surprise — it is the SIREN activation, designed for
smooth continuous fields, which is exactly what a displacement field is. Caveat:
the script auto-scales its learning rate to 0.1×, so part of that gap may be
undertraining rather than the activation.

### Trunk width is a U-curve centred on the inherited default

```
  9.6M   1.906e-03   underfits
 14.2M   1.808e-03
 25.2M   1.644e-03   <- best, and the paper's value
 54.5M   1.708e-03
142.5M   1.845e-03   overfits
```

Overfitting gap rises monotonically with size: 6%, 19%, 34%, 33%.

The trunk is a halving pyramid off the 2051-wide input (2×1024 global feature +
3 coordinates): 2051 → 1024 → 512 → 256 → 128 → 64 → 3. Width 1024 is the only
setting that halves the input; 4096 *expands* to 2× the input before contracting;
256 compresses 8× in one step. Compression ratio tracks the overfitting gap
monotonically, though two configurations reaching the same ratio by different
routes gave 12% and 2% — so ratio explains much but is not the whole mechanism.

---

## 3. Linear vs Neo-Hookean elasticity: no difference

24 runs — 4 conditions × 2 models × 3 seeds, 1000 epochs, trunk 1024 / relu /
max, on an A100 in 4.5 hours.

| condition | non-affine | linear | nonlinear | diff | p | vs predict-zero |
|---|---|---|---|---|---|---|
| `uniform_pressure` | 0.000 | 6.630e-04 | 5.829e-04 | −12.1% | 1.00 | 77× |
| `two_zone_material` | 0.464 | 1.379e-03 | 1.367e-03 | −0.9% | 1.00 | 31× |
| `two_point_poles` | 0.769 | 1.715e-03 | 1.680e-03 | −2.0% | 0.50 | 5× |
| `four_points_poles_equator` | 0.885 | 2.608e-03 | 2.659e-03 | +1.9% | 0.10 | 3× |

Nonlinear wins 3 of 4, **nothing reaches significance**, and differences are
smaller than the seed noise. Exact permutation test; note that with 3 seeds per
group the minimum achievable p is 0.10, so no result at this sample size *can*
reach p < 0.05.

Every condition beat predict-zero by 3–77×, so this is a real null, not a
broken-pipeline null.

**This was predicted before the runs.** At 5% strain the two theories differ by
4.67% in stress (measured: 0.00% at e=1e-6, 0.95% at e=1e-2, 9.13% at e=0.10),
which is below the noise floor. The result is consistent with the paper's own
finding at their larger deformations.

---

## 4. The physics loss

The shipped model computes strain as

```python
torch.autograd.grad(Ux, x, grad_outputs=torch.ones_like(Ux))
```

which returns the **sum over all output points** of ∂(Ux_i)/∂(x_j), not point
*j*'s own derivative. That equals the wanted diagonal only if no point's output
depends on another point's coordinates. Here it fails twice: the per-point head
is fed a tensor with no gradient path to the coordinate leaves, and every point's
displacement depends on the max-pooled global feature — a function of all 1024
points.

Measured on an untrained network, 24 probe points, `uniform_pressure`:

| quantity | value |
|---|---|
| mean true diagonal ∂(Ux_i)/∂(x_i) | 5.19e-07 |
| mean value `grad_outputs=ones` returns | 2.73e-04 — **525×** |
| correlation with true field gradient | **−0.015** |
| fraction of points where `Exx` is exactly 0 | **71.1%** |

The 296/1024 nonzero points are *precisely* the set winning at least one
max-pool channel (set equality verified).

Consistent with this, the physics loss reaches ~0 in every run — 3 of 10 at
exactly 0.0 — regardless of architecture. It costs nothing to satisfy because it
constrains nothing.

### The correction works, and breaks training

Feeding the head gradient-carrying coordinates and holding the global feature
fixed while differentiating fixes the measurement. Validated against
`uniform_pressure`, whose exact answer is a uniform contraction with all normal
strains −0.05 and zero shear. Both variants read off **identical weights** fitted
to the exact field (0.22% field error):

| variant | normal strain | rel. error | shear RMS | % exactly zero |
|---|---|---|---|---|
| as shipped | −0.000329 | **99.3%** | 9.87e-03 | 73.9% |
| corrected | −0.049505 | **1.0%** | 1.40e-03 | 0.0% |
| exact | −0.050000 | — | 0 | — |

Correlation with the true field gradient: −0.015 → **+0.891**, stable across
neighbour counts k = 10–80 and three seeds.

**But the corrected model does not learn.** Cutting the coordinate path also cuts
the physics loss off from the PointNet backbone (gradient 3.2e+02 → 1.7e+00). On
`uniform_pressure` it scored **1.0× better than predicting no movement at all**,
against 122× for the shipped model; final training loss 1.9e-02 vs 1.0e-06.

The contaminated gradient is **load-bearing for optimisation even though it is
not strain**. That tension is the most novel result here and is unexplained.

It is architectural, not a typo — any PointNet-style PINN using
`grad_outputs=ones` inherits it.

---

## 5. Synthetic benchmark and code defects

Seven boundary conditions on a 1024-point sphere (512 Fibonacci surface points,
512 rejection-sampled interior), 5 seeded variants each, all scaled to 5% peak
displacement so the boundary condition is the only variable. Complexity measured
as the **non-affine fraction**: best affine fit source→target, max residual over
max displacement.

FEM via `scipy.spatial.Delaunay` + `scikit-fem`, pure Python on native Windows.
Validated against the analytic sphere to 0.7%, converging 1.35e-2 → 6.8e-3 →
1.9e-3 under refinement; energy identity ½uᵀKu = ½fᵀu exact to 9 digits. Free
point loads leave the stiffness matrix singular (6 rigid modes), handled by
inertia relief rather than pinning nodes.

**Load count does not control complexity.** n = 1, 2, 4, 8, 16, 32 random point
loads give 0.858, 0.852, 0.866, 0.887, 0.863, 0.803 — flat, then declining. Many
random inward loads average toward isotropic pressure, which scores 0.000.
Complexity is set by the loading *mechanism*, not the count.

**`stiffer_material` is byte-identical to `uniform_pressure`.** Matching strain
across two materials means scaling pressure by the stiffness ratio, reproducing
the same clouds.

### Defects found in the existing code

**Checkpoints 199 epochs stale.** Saves on `epoch % save_every == 0` with no
final save; at `save_every=200, epochs=2000` the last write is epoch 1800.
Re-running `two_point_poles`/4096 with the fix: 1.845e-03 → 1.663e-03, ~10%.

**One sample never trains.** `batch_size=2, drop_last=True, shuffle=False` over 5
files silently discards the 5th every epoch. Repurposed here as a held-out set,
but it was not intentional, and the original `rmse_mean` mixed train and test
error (unseen worse in 8/10 runs, mean ratio 1.147).

**Loss logged at 6 decimal places** while the plateau sits near 2e-05, so
converged values round to 0.000000.

**Convergence later than it looks.** Width 1024 reaches within 20% of final at
epoch 890 and within 5% only at 1828.

---

## 6. Not established

- **Everything in section 2 rests on a single seed.** The noise floor from
  section 1 grades them, but the width result in particular would be firmer with
  3 seeds per width (~4 GPU-hours).
- **All synthetic.** Whether the section-4 finding changes anything on the real
  MRI/ultrasound prostate data is untested — and it is the first question a
  reviewer will ask. The paper reported TRE improvements with PINNs on clinical
  data; if the physics term is inert, something else produced those gains, most
  likely the stress branch acting as a generic regulariser. **Do not claim
  otherwise without their data.**
- **The training-failure result is one measurement**, not a characterised
  phenomenon. The gradient collapse is measured; *why* the contaminated signal
  helps is not explained.
- **Four near-duplicate training samples, weak held-out test.** The held-out file
  is the same deformation resampled, not a new one.
- **The strain correction is an approximation.** Holding the global feature fixed
  discards a real gradient path — defensible, but not the total derivative.
- **Stage 2 ran at 1000 epochs**, where width 1024 sits ~1.20× above its final
  loss. Both models had identical budgets so the comparison is fair, but neither
  was trained to convergence.
