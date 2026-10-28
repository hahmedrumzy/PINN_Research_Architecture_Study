"""
Condition 7 -- n_random_points.

Generalisation of condition 6 to a configurable number of inward point loads.
Same unbalanced-load handling (inertia relief). Default n=8; override with
    python generators/n_random_points.py --n 16
Datasets land in toy_datasets/n<N>_random_points/ so several N can coexist.
"""
import sys, os, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C, _fem as F
from four_random_points import load_points

TARGET_MAX_U = C.TARGET_MAX_U


def main(n=8):
    condition = f"n{n}_random_points"
    print(f"=== {condition} ===")
    for i, seed in enumerate(C.SEEDS):
        src = C.source_cloud(seed)
        u, info = F.solve_point_loads(src, load_points(seed, n),
                                      C.LAMBDA_SOFT, C.MU_SOFT, TARGET_MAX_U)
        C.save_pair(condition, i + 1, src, u)
        print(f"  [{i+1}] unbalanced={info['unbalanced_fraction']:.3e}  "
              f"E/W={info['energy_uKu']/info['work_fu']:.6f}  "
              f"max|u|={np.linalg.norm(u,axis=1).max():.3e}  "
              f"mean|u|={np.linalg.norm(u,axis=1).mean():.3e}")
    C.assert_only_npy(condition)
    print("  assert_only_npy OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    main(ap.parse_args().n)
