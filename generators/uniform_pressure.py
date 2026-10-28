"""
Condition: uniform_pressure

Closed-form Lame solution for a solid, homogeneous, isotropic elastic sphere of
radius R under a uniform external pressure P applied over its whole boundary.

The loading is hydrostatic, so the interior stress state is pure hydrostatic
compression sigma_ij = -P delta_ij everywhere and the strain field is uniform:

    eps_xx = eps_yy = eps_zz = -P / (3K),   eps_xy = eps_yz = eps_zx = 0
    u_r(r) = -P * r / (3K)
    u(x)   = u_r(r) * r_hat = -(P / (3K)) * x

with K = lambda + 2*mu/3 the bulk modulus.

Material (peripheral / soft zone, paper Sec. IV-C.2):
    lambda = 82.21, mu = 1.68  ->  K = 83.33
    P = 0.1  ->  u_r / r = -P/(3K) = -4.0e-4
             ->  max|u| at r = R = 1.0 is 4.0e-4
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _common
from _common import (SEEDS, RADIUS, LAMBDA_SOFT, MU_SOFT, bulk_modulus, TARGET_MAX_U,
                     source_cloud, radial_unit, save_pair, assert_only_npy,
                     report)

CONDITION = "uniform_pressure"

K = bulk_modulus(LAMBDA_SOFT, MU_SOFT)          # 82.21 + 1.12 = 83.33
# P is derived from the shared target peak displacement: max|u| = P*R/(3K).
PRESSURE = 3.0 * K * TARGET_MAX_U / RADIUS      # 12.4995 kPa -> 5% strain
STRAIN = -PRESSURE / (3.0 * K)                  # -4.0e-4, all three normal strains
U_MAX_EXPECT = PRESSURE * RADIUS / (3.0 * K)    # +4.0e-4


def displacement(points):
    """u(x) = u_r(r) * r_hat with u_r(r) = -P r / (3K).  Returns (N, 3)."""
    r, r_hat = radial_unit(points)
    u_r = -PRESSURE * r / (3.0 * K)             # (N, 1)
    return u_r * r_hat


def check(points, u):
    r = np.linalg.norm(points, axis=1)
    mag = np.linalg.norm(u, axis=1)

    # (a) uniform strain: u_r / r is the same constant everywhere.
    #     Signed radial component so sign is part of the test.
    u_r = np.einsum("ij,ij->i", u, points) / np.where(r == 0, 1e-30, r)
    interior = r > 1e-12
    ratio = u_r[interior] / r[interior]
    rel_spread = np.abs(ratio - STRAIN).max() / abs(STRAIN)
    assert rel_spread <= 1e-12, (
        f"u_r/r not constant: relative spread {rel_spread:.3e} > 1e-12 "
        f"(min {ratio.min():.16e}, max {ratio.max():.16e}, expected {STRAIN:.16e})")

    # off-diagonal strain is identically zero: u is exactly parallel to x
    # tolerance must be RELATIVE: |u x x| scales with |u|*|x|, so an absolute
    # bound calibrated at one amplitude silently breaks at another.
    cross = np.linalg.norm(np.cross(u, points), axis=1).max()
    scale = np.linalg.norm(u, axis=1).max() * np.linalg.norm(points, axis=1).max()
    assert cross <= 1e-14 * scale, (
        f"u not radial: max|u x x| = {cross:.3e} > {1e-14 * scale:.3e} "
        f"(relative {cross / scale:.3e})")

    # (b) max|u| at r = R equals P*R/(3K)
    surface = np.isclose(r, RADIUS, rtol=0, atol=1e-12)
    assert surface.any(), "no points found at r = R"
    err_surface = np.abs(mag[surface] - U_MAX_EXPECT).max()
    assert err_surface <= 1e-9, (
        f"|u| at r=R is {mag[surface].max():.16e}, expected {U_MAX_EXPECT:.16e} "
        f"(abs err {err_surface:.3e} > 1e-9)")
    err_max = abs(mag.max() - U_MAX_EXPECT)
    assert err_max <= 1e-9, (
        f"max|u| = {mag.max():.16e}, expected {U_MAX_EXPECT:.16e} "
        f"(abs err {err_max:.3e} > 1e-9)")

    return rel_spread, err_max


def main():
    assert abs(K - 83.33) < 1e-9, f"K = {K}, expected 83.33"
    # The quoted -4.0e-4 / 4.0e-4 are the ROUNDED nominal values: K = 83.33 gives
    # 3K = 249.99, not 250, so the exact closed form is -4.00016...e-4. The strict
    # 1e-9 checks in check() are made against the exact P*r/(3K), and we only
    # sanity-check agreement with the nominal figure to its stated precision.
    assert abs(STRAIN + TARGET_MAX_U) < 1e-9, f"strain = {STRAIN}, expected ~ {-TARGET_MAX_U}"
    print(f"[{CONDITION}] note: exact -P/(3K) = {STRAIN:.16e}; "
          f"nominal {-TARGET_MAX_U} differs by {abs(STRAIN + TARGET_MAX_U):.3e} "
          f"(K = 83.33 => 3K = 249.99, not 250)")

    print(f"[{CONDITION}] lambda={LAMBDA_SOFT} mu={MU_SOFT} K={K} P={PRESSURE}")
    print(f"[{CONDITION}] uniform normal strain = {STRAIN:.8e}, "
          f"|u| at r=R = {U_MAX_EXPECT:.8e}")

    worst_spread, worst_err = 0.0, 0.0
    all_max, all_mean = [], []

    for i, seed in enumerate(SEEDS):
        source = source_cloud(seed)
        u = displacement(source)

        spread, err = check(source, u)
        worst_spread = max(worst_spread, spread)
        worst_err = max(worst_err, err)

        save_pair(CONDITION, i + 1, source, u)

        mag = np.linalg.norm(u, axis=1)
        all_max.append(mag.max())
        all_mean.append(mag.mean())
        report(CONDITION, source, u, extra={
            "seed": seed,
            "file": f"{CONDITION}_{i + 1}.npy",
            "u_r/r": f"{STRAIN:.16e}",
        })

    assert_only_npy(CONDITION)

    print(f"[{CONDITION}] OK -- {len(SEEDS)} samples")
    print(f"[{CONDITION}] worst relative spread of u_r/r : {worst_spread:.3e} (tol 1e-12)")
    print(f"[{CONDITION}] worst |max|u| - P*R/(3K)|      : {worst_err:.3e} (tol 1e-9)")
    print(f"[{CONDITION}] max|u| over all samples        : {max(all_max):.8e}")
    print(f"[{CONDITION}] mean|u| over all samples       : {float(np.mean(all_mean)):.8e}")


if __name__ == "__main__":
    main()
