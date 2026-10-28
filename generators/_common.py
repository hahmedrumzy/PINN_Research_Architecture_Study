"""
Shared scaffolding for every synthetic sphere-deformation dataset.

Design note: all conditions reuse the SAME point clouds (same seeds -> same
Fibonacci surface + same rejection-sampled interior). Only the displacement
field differs between conditions. That makes the boundary condition the single
independent variable in the architecture ablation.

Layout enforced by experiment_architecture_combined.py:
    toy_datasets/<condition>/   -> ONLY .npy sample files (load_dataset does a
                                   flat os.listdir + np.load on every entry)
    ground_truth/<condition>/   -> ground-truth displacement fields
    generators/<condition>.py   -> the generator itself
"""

import os
import numpy as np

# --- material constants (paper Sec. IV-C.2: E=5 kPa / 500 kPa at nu=0.49) ---
LAMBDA_SOFT, MU_SOFT = 82.21, 1.68        # peripheral zone
LAMBDA_STIFF, MU_STIFF = 8221.48, 167.78  # transition zone

# Peak displacement every condition is scaled to, so the boundary condition is
# the only variable in the ablation. 0.05 = 5% of the unit radius: still firmly
# small-strain (linear elasticity stays valid) and ~125x the previous 4.0e-4,
# which was six orders of magnitude below the untrained network's own error and
# made all seven datasets indistinguishable.
TARGET_MAX_U = 0.05

N_SURFACE, N_VOLUME = 512, 512
N_POINTS = N_SURFACE + N_VOLUME
RADIUS = 1.0
SEEDS = [42, 123, 256, 512, 999]

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def bulk_modulus(lam, mu):
    return lam + 2.0 * mu / 3.0


def sample_sphere_surface(n=N_SURFACE, radius=RADIUS):
    """Fibonacci lattice -- deterministic, identical across all seeds."""
    golden = (1 + np.sqrt(5)) / 2
    i = np.arange(n)
    theta = np.arccos(1 - 2 * (i + 0.5) / n)
    phi = 2 * np.pi * i / golden
    return np.stack([radius * np.sin(theta) * np.cos(phi),
                     radius * np.sin(theta) * np.sin(phi),
                     radius * np.cos(theta)], axis=1).astype(np.float64)


def sample_sphere_volume(n=N_VOLUME, radius=RADIUS, seed=0):
    """Rejection sampling inside the ball -- varies with seed."""
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        cand = rng.uniform(-radius, radius, (n * 2, 3))
        cand = cand[np.linalg.norm(cand, axis=1) <= radius]
        pts.extend(cand.tolist())
    return np.array(pts[:n], dtype=np.float64)


def source_cloud(seed):
    """(1024, 3) -- surface points FIRST, then interior. Convention preserved."""
    return np.vstack([sample_sphere_surface(), sample_sphere_volume(seed=seed)])


def radial_unit(points):
    r = np.linalg.norm(points, axis=1, keepdims=True)
    r_safe = np.where(r == 0, 1e-30, r)
    return r, points / r_safe


def save_pair(condition, index, source, u):
    """Write toy_datasets/<condition>/<condition>_<i>.npy and its ground truth."""
    data_dir = os.path.join(REPO, "toy_datasets", condition)
    gt_dir = os.path.join(REPO, "ground_truth", condition)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    target = source + u
    dataset = np.ascontiguousarray(
        np.array([source, target], dtype=np.float32))          # (2, 1024, 3)

    validate(dataset)
    np.save(os.path.join(data_dir, f"{condition}_{index}.npy"), dataset)
    np.save(os.path.join(gt_dir, f"{condition}_{index}_ground_truth.npy"),
            u.astype(np.float32))
    return dataset


def validate(dataset):
    assert dataset.shape == (2, N_POINTS, 3), f"bad shape {dataset.shape}"
    assert dataset.dtype == np.float32, f"bad dtype {dataset.dtype}"
    assert np.isfinite(dataset).all(), "non-finite values present"


def assert_only_npy(condition):
    """load_dataset() np.loads every directory entry -- nothing else may live here."""
    d = os.path.join(REPO, "toy_datasets", condition)
    bad = [f for f in os.listdir(d) if not f.endswith(".npy")]
    assert not bad, f"non-.npy entries in {d}: {bad} -- the trainer will crash"


def report(condition, source, u, extra=None):
    mag = np.linalg.norm(u, axis=1)
    print(f"  {condition}: max|u| = {mag.max():.6e}   mean|u| = {mag.mean():.6e}")
    if extra:
        for k, v in extra.items():
            print(f"      {k}: {v}")
