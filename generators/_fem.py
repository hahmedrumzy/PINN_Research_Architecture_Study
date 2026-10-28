"""
Shared FEM scaffolding for the point-load sphere conditions (tasks 4-7).

Native Windows, pure Python: scipy.spatial.Delaunay for tetrahedralisation +
scikit-fem for assembly. No Gmsh, no FEniCS.

Handling the singular system
----------------------------
These conditions have NO fixed supports -- the sphere floats free and is loaded
only by point forces. The stiffness matrix therefore has a 6-dimensional null
space (3 translations + 3 rotations) and a plain spsolve fails.

Pinning nodes would be wrong here: the sphere genuinely wants to move at every
point, so clamping a node injects a spurious reaction force and a fake stress
concentration.

Instead we do the standard free-body treatment (inertia relief with uniform
lumped mass):
  1. project the load onto the self-equilibrated subspace, f <- f - R(R^T f),
     which removes exactly the net force and net torque that no static solution
     could ever balance;
  2. solve (K + alpha*R R^T) u = f, which is non-singular and leaves the
     deformation untouched while fixing the rigid component at zero;
  3. Procrustes out any residual rigid motion at the end.
For an already-balanced load set (e.g. two opposite poles) step 1 is a no-op.
"""

import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from skfem import MeshTet, Basis, ElementVector, ElementTetP1, asm, FacetBasis, LinearForm
from skfem.helpers import dot
from skfem.models.elasticity import linear_elasticity


def build_mesh(points, min_vol=1e-12):
    """Delaunay-tetrahedralise a convex point cloud; drop degenerate tets."""
    t = Delaunay(points).simplices
    v0, v1, v2, v3 = (points[t[:, i]] for i in range(4))
    vol = np.einsum("ij,ij->i", np.cross(v1 - v0, v2 - v0), v3 - v0) / 6.0
    t = t[np.abs(vol) > min_vol]
    orphans = np.setdiff1d(np.arange(len(points)), np.unique(t))
    if orphans.size:
        raise RuntimeError(f"{orphans.size} nodes orphaned by sliver filtering")
    return MeshTet(np.ascontiguousarray(points.T), np.ascontiguousarray(t.T))


def assemble(points, lam, mu):
    mesh = build_mesh(points)
    basis = Basis(mesh, ElementVector(ElementTetP1()))
    K = asm(linear_elasticity(lam, mu), basis)
    return mesh, basis, K


def rigid_modes(points):
    """Orthonormal basis (3N, 6) of the rigid-body null space, DOF-interleaved."""
    n = len(points)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    R = np.zeros((3 * n, 6))
    for i in range(3):                       # translations
        R[i::3, i] = 1.0
    R[1::3, 3], R[2::3, 3] = -z, y           # rotation about x
    R[0::3, 4], R[2::3, 4] = z, -x           # rotation about y
    R[0::3, 5], R[1::3, 5] = -y, x           # rotation about z
    return np.linalg.qr(R)[0]


def point_load_vector(points, basis, load_points, forces):
    """Place each force on the mesh node nearest its requested location."""
    f = np.zeros(basis.N)
    used = []
    for lp, force in zip(np.atleast_2d(load_points), np.atleast_2d(forces)):
        node = int(np.argmin(np.linalg.norm(points - lp, axis=1)))
        used.append(node)
        for c in range(3):
            f[basis.nodal_dofs[c, node]] += force[c]
    return f, used


def pressure_load_vector(mesh, basis, P):
    """Uniform external pressure P on the boundary: t = -P n. Used to validate."""
    fb = FacetBasis(mesh, basis.elem)

    @LinearForm
    def traction(v, w):
        return -P * dot(w.n, v)

    return asm(traction, fb)


def solve_free(K, f, points, basis):
    """Free-floating static solve. Returns (u_nodal (N,3), diagnostics dict)."""
    R = rigid_modes(points)
    # reorder rigid modes into skfem's DOF numbering
    perm = np.empty(3 * len(points), dtype=int)
    for c in range(3):
        perm[c::3] = basis.nodal_dofs[c]
    Rd = np.zeros_like(R)
    Rd[perm] = R

    f_rigid = Rd @ (Rd.T @ f)
    f_bal = f - f_rigid

    alpha = K.diagonal().mean()
    Kreg = (K + csr_matrix(alpha * (Rd @ Rd.T))).tocsr()
    u = spsolve(Kreg, f_bal)

    u_nodal = np.stack([u[basis.nodal_dofs[c]] for c in range(3)], axis=1)
    diag = {
        "unbalanced_fraction": float(np.linalg.norm(f_rigid) / max(np.linalg.norm(f), 1e-300)),
        "energy_uKu": float(0.5 * u @ (K @ u)),
        "work_fu": float(0.5 * f_bal @ u),
    }
    return u_nodal, diag


def remove_rigid_motion(source, u):
    """Procrustes out the best-fit rigid transform from a displacement field."""
    A = source - source.mean(0)
    B = (source + u) - (source + u).mean(0)
    U, _, Vt = np.linalg.svd(B.T @ A)
    d = np.sign(np.linalg.det(U @ Vt))
    Rot = U @ np.diag([1.0, 1.0, d]) @ Vt
    fitted = (source - source.mean(0)) @ Rot.T + (source + u).mean(0)
    return (source + u) - fitted


# ------------------------------------------------------------------ #
#  Reusable driver for the point-load conditions (tasks 4-7)
# ------------------------------------------------------------------ #

def inward_forces(load_points):
    """Unit forces pointing at the sphere centre from each load location."""
    lp = np.atleast_2d(load_points)
    n = np.linalg.norm(lp, axis=1, keepdims=True)
    return -lp / np.maximum(n, 1e-30)


def solve_point_loads(points, load_points, lam, mu, target_max_u=None):
    """
    Assemble, apply inward unit point loads, solve free-floating, then rescale
    linearly so max|u| hits target_max_u (linear elasticity => u is linear in F,
    so one solve plus a scalar is exact). Returns (u, info).
    """
    mesh, basis, K = assemble(points, lam, mu)
    # snap first, then take force directions from the ACTUAL loaded nodes, so
    # each force is exactly inward at the place it is really applied
    lp = np.atleast_2d(load_points)
    nodes = [int(np.argmin(np.linalg.norm(points - q, axis=1))) for q in lp]
    forces = inward_forces(points[nodes])
    f, nodes = point_load_vector(points, basis, points[nodes], forces)
    u, diag = solve_free(K, f, points, basis)
    u = remove_rigid_motion(points, u)

    raw_max = np.linalg.norm(u, axis=1).max()
    scale = 1.0 if target_max_u is None else target_max_u / raw_max
    u = u * scale

    snap = np.linalg.norm(points[nodes] - np.atleast_2d(load_points), axis=1)
    info = dict(diag)
    info.update(n_tets=mesh.t.shape[1], load_nodes=nodes,
                snap_dist_max=float(snap.max()), force_scale=float(scale),
                raw_max_u=float(raw_max))
    return u, info
