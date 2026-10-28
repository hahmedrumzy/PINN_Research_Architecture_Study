"""
Condition: stiffer_material  --  uniform external pressure on a STIFF sphere.

Physics
-------
Identical boundary-value problem to the `uniform_pressure` condition: a solid
elastic sphere of radius R under uniform external pressure P deforms purely
radially with the closed-form solution

    u_r(r) = -P * r / (3 K),        K = lambda + 2 mu / 3   (bulk modulus)

so the displacement field is exactly u(x) = -(P / 3K) * x  -- a pure isotropic
contraction. The ONLY difference from `uniform_pressure` is the material:

    soft  (peripheral zone) : lambda = 82.21   mu = 1.68     -> K_soft  =   83.33
    stiff (transition zone) : lambda = 8221.48 mu = 167.78   -> K_stiff = 8333.33

(paper Sec. IV-C.2, E = 5 kPa vs 500 kPa at nu = 0.49).

Why the pressure is rescaled
----------------------------
Reusing P = 0.1 on the stiff material would give max|u| = 0.1/(3*8333.33)
= 4.0e-6 on a unit sphere. `save_pair` stores the point clouds as float32, whose
absolute resolution near coordinate magnitude 1.0 is ~6e-8 (one ulp = 2^-23).
A 4e-6 displacement therefore survives with barely one to two significant
digits, and `target = source + u` degenerates into a near-rounded copy of
`source`: the network would be learning quantisation noise, not elasticity.

To keep the two conditions comparable we scale the load with the stiffness so
that the STRAIN, not the pressure, is held fixed:

    P_stiff = 0.1 * K_stiff / K_soft  ~= 10.0     ->  u_r/r = -P/(3K) = -4.0e-4

The ablation then varies material stiffness (100x) while the deformation
signal magnitude stays at the soft-case level, so any change in registration
accuracy is attributable to the PINN elasticity term and not to a shrinking
target.

Numerical footnote (honest bookkeeping)
---------------------------------------
K_stiff / K_soft = 8333.3333.../83.33 = 100.00400016..., NOT exactly 100,
because LAMBDA_SOFT/MU_SOFT are rounded 2-decimal constants in _common.py.
Consequently the two statements

    (i)  P_stiff = 10.0 exactly       ->  |u_r/r| = 4.000000e-04
    (ii) P_stiff = 0.1*K_stiff/K_soft ->  |u_r/r| = 4.000160e-04  (== soft case)

differ by 1.6e-08 absolute / 4.0e-05 relative. We adopt (i), P_stiff = 10.0,
because the round number reproduces the nominal design target 4.0e-4 exactly
and the residual 1.6e-08 discrepancy versus the soft condition is ~4x SMALLER
than the float32 storage resolution (6e-08) of the saved point clouds -- i.e.
it is invisible in the data the trainer actually consumes.
"""

import numpy as np

import _common as C


# --- material / load -------------------------------------------------------
K_SOFT = C.bulk_modulus(C.LAMBDA_SOFT, C.MU_SOFT)      #   83.33
K_STIFF = C.bulk_modulus(C.LAMBDA_STIFF, C.MU_STIFF)   # 8333.33

PRESSURE_SOFT = 0.1                                    # uniform_pressure load
PRESSURE = 3.0 * K_STIFF * C.TARGET_MAX_U / C.RADIUS   # 1250.0 kPa -> 5% strain

TARGET_STRAIN = -C.TARGET_MAX_U                        # u_r / r
CONDITION = "stiffer_material"


def displacement(points):
    """u_r(r) = -P r / (3K)  ->  u(x) = -(P / 3K) x  (pure radial contraction)."""
    r, r_hat = C.radial_unit(points)
    u_r = -PRESSURE * r / (3.0 * K_STIFF)              # (N, 1)
    return u_r * r_hat


def main():
    print(f"[{CONDITION}] K_soft = {K_SOFT:.4f}   K_stiff = {K_STIFF:.4f}   "
          f"ratio = {K_STIFF / K_SOFT:.6f}")
    print(f"[{CONDITION}] P = {PRESSURE}   ->  u_r/r = {-PRESSURE / (3 * K_STIFF):.6e}")

    for i, seed in enumerate(C.SEEDS):
        source = C.source_cloud(seed)
        u = displacement(source)

        r, r_hat = C.radial_unit(source)
        mag = np.linalg.norm(u, axis=1)

        # (a) peak displacement == the shared target (surface points sit at r = RADIUS)
        max_u = mag.max()
        assert abs(max_u - C.TARGET_MAX_U) < 1e-9, \
            f"seed {seed}: max|u| = {max_u!r}, expected {C.TARGET_MAX_U} (+-1e-9)"
        assert abs(max_u - PRESSURE * C.RADIUS / (3.0 * K_STIFF)) < 1e-12, \
            f"seed {seed}: max|u| does not match the closed form"

        # (b) u_r / r constant everywhere (isotropic contraction), ~1e-12 relative
        u_r = np.sum(u * r_hat, axis=1)                # signed radial component
        ratio = u_r / r[:, 0]
        rel_spread = np.abs(ratio - TARGET_STRAIN).max() / abs(TARGET_STRAIN)
        assert rel_spread < 1e-12, \
            f"seed {seed}: u_r/r not constant, max relative spread = {rel_spread!r}"

        # displacement must be purely radial (no tangential component)
        tangential = np.linalg.norm(u - u_r[:, None] * r_hat, axis=1).max()
        assert tangential < 1e-15, f"seed {seed}: tangential component {tangential!r}"

        C.save_pair(CONDITION, i + 1, source, u)
        C.report(f"{CONDITION}_{i + 1}", source, u,
                 extra={"seed": seed,
                        "u_r/r": f"{ratio.mean():.9e}",
                        "u_r/r rel. spread": f"{rel_spread:.3e}"})

    # (c) the trainer np.loads every directory entry -- nothing but .npy allowed
    C.assert_only_npy(CONDITION)
    print(f"[{CONDITION}] all assertions passed; {len(C.SEEDS)} samples written.")


if __name__ == "__main__":
    main()
