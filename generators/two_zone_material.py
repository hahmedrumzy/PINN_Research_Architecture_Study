"""
Condition: two_zone_material

Closed-form (NOT FEM) piecewise linear-elastic solution for two concentric
spherical zones under a uniform external pressure P applied on r = R.

Geometry / materials
--------------------
    inner core   r <= a          : transition-zone material  (LAMBDA_STIFF, MU_STIFF)
    outer shell  a <= r <= R     : peripheral-zone material  (LAMBDA_SOFT,  MU_SOFT)
    a = 0.5, R = 1.0, P = 0.1

    material 1 (core) : lambda = 8221.48, mu = 167.78  ->  K1 = lambda + 2mu/3 = 8333.33
    material 2 (shell): lambda =   82.21, mu =   1.68  ->  K2 = lambda + 2mu/3 =   83.33

Solution
--------
The general spherically-symmetric radial displacement of a linear elastic solid is

    u_r(r)      = A*r + B/r**2
    sigma_rr(r) = 3*K*A - 4*mu*B/r**3          with  K = lambda + 2*mu/3

Regularity at the origin forces B1 = 0 in the core, so u_r = A1*r there.  The
three remaining conditions

    (i)   sigma_rr(R) = -P                     (outer surface traction)
    (ii)  u_r(a-) = u_r(a+)                    (displacement continuity)
    (iii) sigma_rr(a-) = sigma_rr(a+)          (traction continuity)

close the system.  With

    gamma = 3*(K2 - K1) / (3*K1 + 4*mu2)

    A2 = -P / (3*K2 - 4*mu2*gamma*(a/R)**3)
    B2 = gamma * a**3 * A2
    A1 = (1 + gamma) * A2

so

    u_r(r) = A1*r                r <= a
    u_r(r) = A2*r + B2/r**2      r >  a

and the displacement vector is u = u_r(r) * r_hat.

Degeneracy: setting material 1 == material 2 gives gamma = 0, hence B2 = 0 and
A1 = A2 = -P/(3K), i.e. the formula collapses exactly onto the homogeneous
uniform-pressure solution u_r = -P*r/(3K).  This is asserted below.

CAVEAT -- stiffness map mismatch with the trainer
-------------------------------------------------
experiment_architecture_combined.py's get_lame_vectors() partitions the point
cloud along the Z AXIS at 2/3 of the z-extent (split_axis=2, split_frac=2/3),
assigning the "stiff" Lame pair below the threshold and the "soft" pair above.
This dataset, by contrast, is zoned RADIALLY (a stiff ball inside a soft shell).
The two partitions do not coincide.  Therefore, if this condition is ever trained
with non-default --lame1-stiff / --lame2-stiff, the physics (linear elasticity)
loss will enforce a stiffness map that does NOT match how the data was generated,
and the residual will be inconsistent with the ground truth.  At the trainer's
defaults (lame*_stiff == lame*_soft == the soft/peripheral pair) the map is
uniform and the mismatch is harmless.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _common  # noqa: E402
from _common import (  # noqa: E402
    LAMBDA_SOFT, MU_SOFT, LAMBDA_STIFF, MU_STIFF, RADIUS, SEEDS,
    assert_only_npy, bulk_modulus, radial_unit, report, save_pair, TARGET_MAX_U,
)

CONDITION = "two_zone_material"

A_INNER = 0.5          # core radius
R_OUTER = RADIUS       # 1.0


def _pressure_for_target():
    """Peak |u| is at r=R and linear in P, so one unit solve fixes the scale."""
    g, A1, A2, B2, _, _ = _coeffs_at_P(1.0)
    peak = abs(A2 * R_OUTER + B2 / R_OUTER ** 2)
    return TARGET_MAX_U / peak


def _coeffs_at_P(P):
    K1 = bulk_modulus(LAMBDA_STIFF, MU_STIFF)
    K2 = bulk_modulus(LAMBDA_SOFT, MU_SOFT)
    mu2 = MU_SOFT
    gamma = 3.0 * (K2 - K1) / (3.0 * K1 + 4.0 * mu2)
    A2 = -P / (3.0 * K2 - 4.0 * mu2 * gamma * (A_INNER / R_OUTER) ** 3)
    B2 = gamma * A_INNER ** 3 * A2
    return gamma, (1.0 + gamma) * A2, A2, B2, K1, K2


PRESSURE = _pressure_for_target()   # ~14.31 kPa -> peak |u| = TARGET_MAX_U


def coefficients(lam1, mu1, lam2, mu2, a=A_INNER, R=R_OUTER, P=PRESSURE):
    """Return (gamma, A1, A2, B2, K1, K2) for the two-zone problem."""
    K1 = bulk_modulus(lam1, mu1)
    K2 = bulk_modulus(lam2, mu2)

    gamma = 3.0 * (K2 - K1) / (3.0 * K1 + 4.0 * mu2)
    A2 = -P / (3.0 * K2 - 4.0 * mu2 * gamma * (a / R) ** 3)
    B2 = gamma * a ** 3 * A2
    A1 = (1.0 + gamma) * A2
    return gamma, A1, A2, B2, K1, K2


def radial_displacement(r, A1, A2, B2, a=A_INNER):
    """u_r(r): A1*r inside the core, A2*r + B2/r^2 in the shell."""
    r = np.asarray(r, dtype=np.float64)
    inner = r <= a
    r_safe = np.where(r == 0.0, 1e-30, r)
    return np.where(inner, A1 * r, A2 * r + B2 / r_safe ** 2)


def displacement_field(points, A1, A2, B2, a=A_INNER):
    r, rhat = radial_unit(points)
    return radial_displacement(r[:, 0], A1, A2, B2, a)[:, None] * rhat


def sigma_rr_shell(r, A2, B2, K2, mu2):
    return 3.0 * K2 * A2 - 4.0 * mu2 * B2 / r ** 3


def run_assertions():
    """STEP 3 -- every check here must actually execute."""
    gamma, A1, A2, B2, K1, K2 = coefficients(
        LAMBDA_STIFF, MU_STIFF, LAMBDA_SOFT, MU_SOFT)
    mu2 = MU_SOFT
    a, R, P = A_INNER, R_OUTER, PRESSURE

    print("  coefficients:")
    print(f"      K1 = {K1!r}   K2 = {K2!r}")
    print(f"      gamma = {gamma!r}")
    print(f"      A1    = {A1!r}")
    print(f"      A2    = {A2!r}")
    print(f"      B2    = {B2!r}")

    # (a) DEGENERACY: material 1 == material 2 must collapse to single-material.
    g0, A1_0, A2_0, B2_0, K1_0, K2_0 = coefficients(
        LAMBDA_SOFT, MU_SOFT, LAMBDA_SOFT, MU_SOFT)
    K_soft = bulk_modulus(LAMBDA_SOFT, MU_SOFT)
    u_single = -P / (3.0 * K_soft)
    assert abs(g0) < 1e-12, f"(a) gamma != 0 in degenerate case: {g0!r}"
    assert abs(B2_0) < 1e-12, f"(a) B2 != 0 in degenerate case: {B2_0!r}"
    assert abs(A1_0 - u_single) <= 1e-12 * abs(u_single), \
        f"(a) A1 != -P/(3K): {A1_0!r} vs {u_single!r}"
    assert abs(A2_0 - u_single) <= 1e-12 * abs(u_single), \
        f"(a) A2 != -P/(3K): {A2_0!r} vs {u_single!r}"
    assert abs(A1_0 - A2_0) <= 1e-12 * abs(u_single), "(a) A1 != A2 in degenerate case"
    print(f"  [a] degeneracy OK: gamma={g0!r}, B2={B2_0!r}, "
          f"A1=A2={A1_0!r} == -P/(3K)={u_single!r}")

    # (b) DISPLACEMENT CONTINUITY at r = a.
    u_in = A1 * a
    u_out = A2 * a + B2 / a ** 2
    rel_u = abs(u_in - u_out) / abs(u_in)
    assert rel_u < 1e-12, f"(b) displacement jump at r=a: rel={rel_u!r}"
    print(f"  [b] displacement continuity OK: u(a-)={u_in!r}, u(a+)={u_out!r}, "
          f"rel err = {rel_u:.3e}")

    # (c) TRACTION CONTINUITY at r = a  (core has B1 = 0).
    s_in = 3.0 * K1 * A1
    s_out = 3.0 * K2 * A2 - 4.0 * mu2 * B2 / a ** 3
    rel_s = abs(s_in - s_out) / abs(s_in)
    assert rel_s < 1e-12, f"(c) traction jump at r=a: rel={rel_s!r}"
    print(f"  [c] traction continuity OK: s_rr(a-)={s_in!r}, s_rr(a+)={s_out!r}, "
          f"rel err = {rel_s:.3e}")

    # (d) OUTER BC: sigma_rr(R) = -P.
    s_R = sigma_rr_shell(R, A2, B2, K2, mu2)
    rel_bc = abs(s_R + P) / P
    assert rel_bc < 1e-12, f"(d) outer BC violated: sigma_rr(R)={s_R!r}"
    print(f"  [d] outer BC OK: sigma_rr(R)={s_R!r}, rel err = {rel_bc:.3e}")

    return gamma, A1, A2, B2


def main():
    print(f"[{CONDITION}] closed-form two-zone (stiff core / soft shell) solution")
    gamma, A1, A2, B2 = run_assertions()

    n_core_total = n_shell_total = 0
    for i, seed in enumerate(SEEDS):
        source = _common.source_cloud(seed)
        u = displacement_field(source, A1, A2, B2)
        save_pair(CONDITION, i + 1, source, u)

        r = np.linalg.norm(source, axis=1)
        n_core = int((r <= A_INNER).sum())
        n_shell = int(r.shape[0] - n_core)
        n_core_total += n_core
        n_shell_total += n_shell
        report(CONDITION, source, u,
               extra={"seed": seed,
                      "core points (r<=a)": n_core,
                      "shell points (r>a)": n_shell})

    print(f"  totals over {len(SEEDS)} seeds: core={n_core_total}, "
          f"shell={n_shell_total}, points/seed=1024")

    # (e) directory hygiene -- the trainer np.loads every entry.
    assert_only_npy(CONDITION)
    print(f"  [e] assert_only_npy('{CONDITION}') OK")


if __name__ == "__main__":
    main()
