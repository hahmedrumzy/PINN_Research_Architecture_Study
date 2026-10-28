"""
Toy Dataset Generation for PINN Point Set Registration
=======================================================
Generates multiple sphere datasets under uniform radial pressure.
Each dataset uses a different random seed for point sampling,
giving slightly different point cloud configurations of the same geometry.

Uses analytic linear elasticity solution — no FEA library needed.
Output: N sphere_uniform_pressure_X.npy files of shape (2, 1024, 3)
  - index 0 = source (undeformed sphere)
  - index 1 = target (deformed sphere)
Ground truth displacements saved separately in ground_truth/ folder.
Compatible with the existing prostateset_v2 dataloader.

Material properties (uniform, linear elastic):
    lame1 (lambda) = 82.21  kPa  (soft tissue, PZ values from paper)
    lame2 (mu)     = 1.68   kPa
"""

import numpy as np
import os

# ------------------------------------------------------------------ #
#  Material properties — matching the paper's PZ values
# ------------------------------------------------------------------ #
LAMBDA = 82.21   # Lame parameter lambda (kPa)
MU     = 1.68    # Lame parameter mu / shear modulus (kPa)

# Derived quantities
E  = MU * (3*LAMBDA + 2*MU) / (LAMBDA + MU)   # Young's modulus
NU = LAMBDA / (2*(LAMBDA + MU))                # Poisson's ratio

print(f"Material properties:")
print(f"  lambda = {LAMBDA:.2f} kPa")
print(f"  mu     = {MU:.2f} kPa")
print(f"  E      = {E:.2f} kPa")
print(f"  nu     = {NU:.4f}")

# ------------------------------------------------------------------ #
#  Parameters
# ------------------------------------------------------------------ #
N_POINTS    = 1024          # number of points per point cloud
R           = 1.0           # sphere radius
P           = 0.1           # applied pressure magnitude (kPa)
N_DATASETS  = 5             # number of datasets to generate
SEEDS       = [42, 123, 256, 512, 999]   # one seed per dataset
OUTPUT_DIR  = "toy_datasets"
GT_DIR      = "ground_truth"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GT_DIR, exist_ok=True)

# ------------------------------------------------------------------ #
#  Point sampling
# ------------------------------------------------------------------ #

def sample_sphere_surface(n, radius=1.0):
    """ 
    Uniformly sample n points on sphere surface using Fibonacci lattice.
    Deterministic — same result regardless of random seed.
    Returns array of shape (n, 3).
    """
    golden = (1 + np.sqrt(5)) / 2          # golden ratio ~ 1.618
    i      = np.arange(n)                  # indices [0, 1, ..., n-1]

    # polar angle: maps indices evenly from north pole to south pole
    theta = np.arccos(1 - 2*(i+0.5)/n)

    # azimuthal angle: golden ratio spacing prevents clustering
    phi = 2 * np.pi * i / golden

    # convert spherical to cartesian coordinates
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    return np.stack([x, y, z], axis=1).astype(np.float32)     # (n, 3)


def sample_sphere_volume(n, radius=1.0):
    """
    Uniformly sample n points inside sphere volume using rejection sampling.
    Random — different results per seed.
    Returns array of shape (n, 3).
    """
    points = []
    while len(points) < n:
        # generate random points in bounding cube
        pts  = np.random.uniform(-radius, radius, (n*2, 3))
        # keep only points inside the sphere
        mask = np.linalg.norm(pts, axis=1) <= radius
        points.extend(pts[mask].tolist())
    return np.array(points[:n], dtype=np.float32)              # (n, 3)


# ------------------------------------------------------------------ #
#  Analytic displacement: uniform radial pressure
# ------------------------------------------------------------------ #
# Lame's solution for solid sphere under uniform external pressure P:
#
#   u_r(r) = -P * r / (3K)
#
# where K = lambda + 2*mu/3 is the bulk modulus.
# Derived by solving the linear elasticity PDEs with spherical symmetry.
# Every point moves radially inward proportional to its distance from origin.
# ------------------------------------------------------------------ #

def sphere_uniform_pressure_displacement(points, P, lam, mu):
    """
    Compute analytic displacement field for sphere under uniform external pressure.

    Args:
        points: (N, 3) float32 array of point coordinates
        P:      applied pressure magnitude (kPa)
        lam:    Lame parameter lambda (kPa)
        mu:     Lame parameter mu / shear modulus (kPa)

    Returns:
        u: (N, 3) float32 array of displacement vectors
    """
    K     = lam + 2*mu/3                                        # bulk modulus

    r     = np.linalg.norm(points, axis=1, keepdims=True)       # (N,1) distance from origin
    r     = np.where(r == 0, 1e-10, r)                          # avoid division by zero at origin

    r_hat = points / r                                           # (N,3) unit radial vector

    u_r   = -P * r / (3 * K)                                    # (N,1) radial displacement
                                                                 # negative = moves inward
    u     = (u_r * r_hat).astype(np.float32)                    # (N,3) displacement vector
    return u


# ------------------------------------------------------------------ #
#  Generate N datasets with different random seeds
# ------------------------------------------------------------------ #

# Surface points are deterministic (Fibonacci lattice) — same for all datasets
# Volume points change with each seed — gives different interior point configurations
sphere_surface = sample_sphere_surface(512, R)   # (512, 3) — fixed across all datasets

print(f"\nGenerating {N_DATASETS} datasets...")

for idx, seed in enumerate(SEEDS):
    np.random.seed(seed)

    # sample interior points with this seed
    sphere_volume = sample_sphere_volume(512, R)                          # (512, 3)

    # combine surface and volume points
    sphere_source = np.vstack([sphere_surface, sphere_volume])            # (1024, 3)

    # compute analytic displacements
    u_sphere = sphere_uniform_pressure_displacement(sphere_source, P, LAMBDA, MU)

    # target = source + displacement
    sphere_target = (sphere_source + u_sphere).astype(np.float32)

    # stack into (2, 1024, 3) — format expected by prostateset_v2
    dataset = np.ascontiguousarray(
        np.array([sphere_source, sphere_target], dtype=np.float32)
    )                                                                      # (2, 1024, 3)

    # save dataset to toy_datasets/
    dataset_filename = f"sphere_uniform_pressure_{idx+1}.npy"
    dataset_path     = os.path.join(OUTPUT_DIR, dataset_filename)
    np.save(dataset_path, dataset)

    # save ground truth displacements to ground_truth/
    gt_filename = f"sphere_uniform_pressure_{idx+1}_ground_truth.npy"
    gt_path     = os.path.join(GT_DIR, gt_filename)
    np.save(gt_path, u_sphere)

    print(f"  Dataset {idx+1} (seed={seed}):")
    print(f"    Shape:            {dataset.shape}")
    print(f"    Max displacement: {np.max(np.linalg.norm(u_sphere, axis=1)):.6f}")
    print(f"    Saved to:         {dataset_path}")
    print(f"    Ground truth:     {gt_path}")

# ------------------------------------------------------------------ #
#  Summary
# ------------------------------------------------------------------ #
print(f"\nAll datasets generated.")
print(f"\ntoy_datasets/ contains:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  {f}")
print(f"\nground_truth/ contains:")
for f in sorted(os.listdir(GT_DIR)):
    print(f"  {f}")
print(f"\nTo use in the PINN pipeline:")
print(f"  python non_rigid_registration_pinns_modified_v10.py \\")
print(f"      --outfile ./output \\")
print(f"      --traindata-path ./toy_datasets \\")
print(f"      --testdata-path ./toy_datasets")  