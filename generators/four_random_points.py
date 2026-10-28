"""
Condition 6 -- four_random_points.

Inward point loads at four RANDOMLY chosen surface points, re-drawn per seed.

Unlike conditions 4 and 5 this load set is NOT self-equilibrated -- a free body
under an unbalanced force set has no static solution at all. _fem.solve_free
handles it by inertia relief: the net force and net torque are projected out
(uniform lumped mass), which is the d'Alembert equilibrium of the free body.
The reported 'unbalanced fraction' is how much of the load that removed, and is
expected to be O(0.1-1) here rather than ~0.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C, _fem as F

CONDITION = "four_random_points"
TARGET_MAX_U = C.TARGET_MAX_U
N_LOADS = 4


def load_points(seed, n=N_LOADS):
    """n random points ON the sphere surface, reproducible per seed."""
    rng = np.random.default_rng(seed + 100000)
    v = rng.normal(size=(n, 3))
    return C.RADIUS * v / np.linalg.norm(v, axis=1, keepdims=True)


def main():
    print(f"=== {CONDITION} ===")
    for i, seed in enumerate(C.SEEDS):
        src = C.source_cloud(seed)
        u, info = F.solve_point_loads(src, load_points(seed),
                                      C.LAMBDA_SOFT, C.MU_SOFT, TARGET_MAX_U)
        C.save_pair(CONDITION, i + 1, src, u)
        print(f"  [{i+1}] unbalanced={info['unbalanced_fraction']:.3e}  "
              f"E/W={info['energy_uKu']/info['work_fu']:.6f}  "
              f"snap={info['snap_dist_max']:.2e}  max|u|={np.linalg.norm(u,axis=1).max():.3e}  "
              f"mean|u|={np.linalg.norm(u,axis=1).mean():.3e}")
    C.assert_only_npy(CONDITION)
    print("  assert_only_npy OK")


if __name__ == "__main__":
    main()
