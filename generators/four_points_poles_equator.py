"""
Condition 5 -- four_points_poles_equator.

Inward point loads at the two poles PLUS two opposite equator points.
Still self-equilibrated by construction (two antipodal pairs), but the load is
now biaxial rather than uniaxial, so the response has no axis of symmetry.
Same FEM scaffolding as condition 4.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C, _fem as F

CONDITION = "four_points_poles_equator"
TARGET_MAX_U = C.TARGET_MAX_U


def load_points(seed=None):
    R = C.RADIUS
    return np.array([[0, 0, R], [0, 0, -R], [R, 0, 0], [-R, 0, 0]], dtype=float)


def main():
    print(f"=== {CONDITION} ===")
    for i, seed in enumerate(C.SEEDS):
        src = C.source_cloud(seed)
        u, info = F.solve_point_loads(src, load_points(seed),
                                      C.LAMBDA_SOFT, C.MU_SOFT, TARGET_MAX_U)
        C.save_pair(CONDITION, i + 1, src, u)
        if i == 0:
            print(f"  tets={info['n_tets']}  nodes={info['load_nodes']}  snap={info['snap_dist_max']:.3e}")
            print(f"  unbalanced fraction = {info['unbalanced_fraction']:.3e}")
            print(f"  energy/work ratio   = {info['energy_uKu']/info['work_fu']:.9f}")
        C.report(f"{CONDITION}[{i+1}]", src, u)
    C.assert_only_npy(CONDITION)
    print("  assert_only_npy OK")


if __name__ == "__main__":
    main()
