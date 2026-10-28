"""
Condition 4 -- two_point_poles.

Point loads at two opposite poles of the sphere, both directed inward. First
asymmetric condition: no closed-form solution, so this is solved with FEM
(scipy.spatial.Delaunay + scikit-fem, see generators/_fem.py).

The load set is self-equilibrated by construction (equal, opposite, collinear
through the centre => zero net force AND zero net torque), so the inertia-relief
projection in _fem.solve_free is a no-op here and the static solution is exact
up to discretisation.

Load magnitude is rescaled so max|u| matches the analytic conditions
(4.0e-4 on a unit sphere), keeping all seven datasets comparable in signal size.

Accuracy note: 1024 nodes is a coarse mesh and a point load is singular in the
continuum limit, so displacement immediately next to a load node is
mesh-dependent. Fine for a toy benchmark; do not read it as ground truth physics.
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C
import _fem as F

CONDITION = "two_point_poles"
TARGET_MAX_U = C.TARGET_MAX_U


def load_points(seed=None):
    return np.array([[0.0, 0.0, C.RADIUS], [0.0, 0.0, -C.RADIUS]])


def main():
    print(f"=== {CONDITION} ===")
    for i, seed in enumerate(C.SEEDS):
        src = C.source_cloud(seed)
        u, info = F.solve_point_loads(src, load_points(seed),
                                      C.LAMBDA_SOFT, C.MU_SOFT, TARGET_MAX_U)

        # --- physical checks (mesh-independent; the Fibonacci lattice is NOT
        # mirror-symmetric, so exact point-pair comparison is not available) ---
        z = src[:, 2]
        rho_hat = src[:, :2] / np.maximum(np.linalg.norm(src[:, :2], axis=1, keepdims=True), 1e-30)
        u_rho = np.einsum("ij,ij->i", u[:, :2], rho_hat)
        umax = np.abs(u).max()
        uz_top, uz_bot = u[z > 0.3, 2].mean(), u[z < -0.3, 2].mean()
        checks = {
            "polar compression (u_z<0 above, >0 below)": (uz_top < 0) and (uz_bot > 0),
            "equatorial bulge (mean u_rho > 0)": u_rho[np.abs(z) < 0.3].mean() > 0,
        }
        antisym = abs(uz_top + uz_bot) / umax

        assert all(checks.values()), f'physical checks failed: {checks}'
        C.save_pair(CONDITION, i + 1, src, u)
        if i == 0:
            print(f"  tets={info['n_tets']}  load nodes={info['load_nodes']}  "
                  f"snap dist={info['snap_dist_max']:.3e}")
            print(f"  unbalanced load fraction = {info['unbalanced_fraction']:.3e} (want ~0)")
            print(f"  energy/work ratio        = {info['energy_uKu']/info['work_fu']:.9f}")
            for k, v in checks.items():
                print(f"  {'PASS' if v else 'FAIL'}  {k}")
            print(f"  u_z antisymmetry residual = {antisym:.3e} (want << 1)")
            print(f"  mean u_z above/below      = {uz_top:.3e} / {uz_bot:.3e}")
        C.report(f"{CONDITION}[{i+1}]", src, u)

    C.assert_only_npy(CONDITION)
    print(f"  assert_only_npy OK")


if __name__ == "__main__":
    main()
