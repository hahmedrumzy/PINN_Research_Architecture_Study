"""
experiment_full_study.py
========================
Three-stage study built ON TOP of experiment_architecture_combined.py, which is
imported and NEVER modified.

  --stage 0   confirm the strain bug, fix it, validate against the analytic case
  --stage 1   does optimal trunk width depend on deformation complexity?
  --stage 2   linear vs nonlinear (Neo-Hookean) elasticity

THE STRAIN BUG
--------------
The shipped forward pass computes strain as

    torch.autograd.grad(Ux, x, grad_outputs=torch.ones_like(Ux))

which returns, for each coordinate j,   sum_i d(Ux_i)/d(x_j)  -- a sum over ALL
output points, not point j's own derivative. That is only equal to the diagonal
d(Ux_j)/d(x_j) if no output point depends on any other point's coordinate.

In this architecture that condition fails twice over:
  1. the per-point head is fed `source`, a tensor with no gradient path back to
     the x/y/z leaves at all, so the diagonal term is ~0 to begin with; and
  2. every point's displacement depends on the max-pooled global feature, which
     is a function of ALL 1024 points -- so the 1023 off-diagonal terms are
     nonzero and dominate the sum.

THE FIX
-------
The predicted displacement field is, for a fixed source/target pair,

    u(x) = head( g , x )        g = global feature of the pair

Feed the head `source_used` (which does carry a gradient path to x/y/z) and hold
`g` fixed via .detach() when differentiating. The head is a stack of 1x1 convs,
i.e. purely pointwise, so with g held constant  d(u_i)/d(x_j) = 0 for i != j
exactly, and grad_outputs=ones then returns precisely the diagonal.

THIS IS AN APPROXIMATION -- stated plainly, not exact.

The true total derivative of the predicted displacement w.r.t. a point's
coordinate includes a second path: moving that point changes the max-pooled
global feature g, which changes every point's displacement. Detaching g
discards that path. What is computed is the PARTIAL derivative of the per-point
head at fixed g, not the total derivative of the network's output.

The approximation is defensible -- g is a property of the input pair rather than
of where the field is evaluated, and the discarded path is exactly the
cross-point coupling that made the shipped version meaningless -- but it is an
approximation, and any result built on it inherits that caveat.

Detaching g would stop the PointNet backbone receiving gradient from the
registration loss, so the forward pass returns BOTH: `disp` (full graph, used
for chamfer and as the model output) and a numerically identical `disp_pw`
(global feature detached) used only to differentiate. Same values, different
graph -- the DISPLACEMENT backbone still trains.

MEASURED, NOT ASSERTED  (all on uniform_pressure sample 1, dim_k=1024)
---------------------------------------------------------------------
Size of the bug -- untrained net, trunk [1024,512,256,128,64], true diagonal
obtained by looping autograd one output element at a time over 24 probe points:
  mean |true d(Ux_i)/d(x_i)|            5.19e-07
  mean |what grad_outputs=ones returns| 2.73e-04     (525x the real signal)
  mean sum_{j!=i} |d(Ux_j)/d(x_i)|      3.78e-04
  With g detached AND the head fed `source` (as shipped) the derivative is
  None -- literally no gradient path -- so 100% of the shipped number arrives
  through the global feature.
  The shipped "strain" is EXACTLY ZERO at 71.1% of points; the 296/1024 points
  where it is nonzero are exactly the points that win at least one max-pool
  channel. It is a max-pool-winner mask, not a strain field.

Size of the approximation the fix makes, restricted to the max-pool winners
(the only points where the discarded path is nonzero at all), trunk
[256,128,64,32,16]:
  untrained network   mean |discarded| / mean |partial| = 0.008,  corr 0.9996
  field fitted to the
  exact analytic field  ... = 0.0001, corr 0.999995
So the caveat above is real in principle but numerically negligible here.

KNOWN DEFECT IN FixedStrainRegistration
---------------------------------------
Detaching `g_s_rep` removes d(g_s)/d(x), which is wanted, but it also removes
d(g_s)/d(theta), which is not. The stress heads are the only consumer of
`ptfeatures_pinns`, so under FixedStrainRegistration that backbone receives
EXACTLY ZERO gradient and never trains -- measured, sum|grad| = 0.0 against
3.08e-01 in the shipped model. The `disp`/`disp_pw` double-forward works around
this for `ptfeatures` but there is no equivalent for the stress branch. Note
also that `ptfeatures` now gets gradient only from chamfer, not from the physics
loss (sum|grad| 6.7e+01 -> 4.0e-01).

FixedStrainRegistrationV2 avoids both problems by detaching the backbone's
COORDINATE INPUT rather than its output feature. It is bitwise-identical on
every returned tensor and restores the stress backbone's gradient. PREFER IT.
"""

import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment_architecture_combined as X


class FixedStrainRegistration(X.PhysicsInformedRegistration):
    """Identical network; only the differentiation path is repaired."""

    def forward(self, data, pointwise_strain=True):
        device = data.device
        source = data[:, 0, :, :]
        target = data[:, 1, :, :]

        x = source[:, :, 0].reshape(-1, 1)
        y = source[:, :, 1].reshape(-1, 1)
        z = source[:, :, 2].reshape(-1, 1)
        x.requires_grad_(True); y.requires_grad_(True); z.requires_grad_(True)

        B, N = data.shape[0], source.shape[1]
        source_used = torch.zeros(B, N, 3, device=device)
        source_used[:, :, 0] = x.reshape(B, -1)
        source_used[:, :, 1] = y.reshape(B, -1)
        source_used[:, :, 2] = z.reshape(B, -1)

        source_four = X._homogeneous(source_used)
        target_four = X._homogeneous(target)

        g = torch.cat((self.ptfeatures(source_four), self.ptfeatures(target_four)), -1)
        g_rep = g.unsqueeze(1).repeat(1, N, 1)

        def head(gr, coords):
            cat = torch.cat((gr, coords), -1).permute(0, 2, 1)
            return self.last_layer(self.list_layers2(cat)).permute(0, 2, 1)

        # full-graph output: backbone still trains from the registration loss
        disp = head(g_rep, source_used)
        deformed_source = source + disp

        # numerically identical, but g held fixed => purely pointwise derivative
        disp_pw = head(g_rep.detach(), source_used) if pointwise_strain else disp

        Ux, Uy, Uz = (disp_pw[:, :, i].reshape(-1, 1) for i in range(3))

        sb = self._stress_backbone()
        g_s = torch.cat((sb(source_four), sb(target_four)), -1)
        g_s_rep = g_s.unsqueeze(1).repeat(1, N, 1)
        if pointwise_strain:
            g_s_rep = g_s_rep.detach()
        g_s_cat = torch.cat((g_s_rep, source_used), -1).permute(0, 2, 1)

        S = {}
        for k in ["sxx", "syy", "szz", "sxy", "sxz", "syz"]:
            S[k] = getattr(self, f"pinns_{k}2")(
                getattr(self, f"pinns_{k}1")(g_s_cat)).reshape(-1, 1)

        def d(out, wrt):
            return X.check_derivative(torch.autograd.grad(
                out, wrt, torch.ones_like(out), create_graph=True,
                allow_unused=True)[0], wrt)

        Exx, Eyy, Ezz = d(Ux, x), d(Uy, y), d(Uz, z)
        Exy = 0.5 * (d(Ux, y) + d(Uy, x))
        Exz = 0.5 * (d(Ux, z) + d(Uz, x))
        Eyz = 0.5 * (d(Uy, z) + d(Uz, y))
        mb1 = d(S["sxx"], x) + d(S["sxy"], y) + d(S["sxz"], z)
        mb2 = d(S["sxy"], x) + d(S["syy"], y) + d(S["syz"], z)
        mb3 = d(S["sxz"], x) + d(S["syz"], y) + d(S["szz"], z)

        return {"deformed_source": deformed_source,
                "Ux": Ux, "Uy": Uy, "Uz": Uz,
                "Sxx": S["sxx"], "Syy": S["syy"], "Szz": S["szz"],
                "Sxy": S["sxy"], "Sxz": S["sxz"], "Syz": S["syz"],
                "Exx": Exx, "Eyy": Eyy, "Ezz": Ezz,
                "Exy": Exy, "Exz": Exz, "Eyz": Eyz,
                "mb1": mb1, "mb2": mb2, "mb3": mb3,
                "_disp": disp, "_x": x, "_y": y, "_z": z}


class FixedStrainRegistrationV2(X.PhysicsInformedRegistration):
    """Same repair, done by detaching the backbone's COORDINATE INPUT instead of
    its output feature.

    FixedStrainRegistration detaches g itself. That kills d(g)/d(x) -- which is
    what we want -- but it also kills d(g)/d(theta), so any backbone whose only
    route to the loss runs through a detached g stops training altogether. That
    is exactly what happens to `ptfeatures_pinns`: under FixedStrainRegistration
    it receives EXACTLY ZERO gradient (measured), because `g_s_rep` is detached
    and the stress heads are its only consumer. FixedStrainRegistration works
    around the same problem for `ptfeatures` by running the displacement head
    twice, but no such workaround exists for the stress branch.

    Detaching the INPUT instead --  g = backbone(source_used.detach()) --  gives
    d(g)/d(x) = 0 while leaving d(g)/d(theta) fully intact. The coordinates are
    network inputs, not parameters, so nothing that matters for training is
    lost. Consequences:
      * identical strain values to FixedStrainRegistration (verified bitwise),
      * `ptfeatures_pinns` trains again,
      * one head forward instead of two, so it is also cheaper.
    The same caveat about it being a PARTIAL derivative at fixed g still applies
    -- that is inherent to treating g as a property of the pair, not a defect of
    either implementation.
    """

    def forward(self, data, pointwise_strain=True):
        device = data.device
        source = data[:, 0, :, :]
        target = data[:, 1, :, :]

        x = source[:, :, 0].reshape(-1, 1)
        y = source[:, :, 1].reshape(-1, 1)
        z = source[:, :, 2].reshape(-1, 1)
        x.requires_grad_(True); y.requires_grad_(True); z.requires_grad_(True)

        B, N = data.shape[0], source.shape[1]
        source_used = torch.zeros(B, N, 3, device=device)
        source_used[:, :, 0] = x.reshape(B, -1)
        source_used[:, :, 1] = y.reshape(B, -1)
        source_used[:, :, 2] = z.reshape(B, -1)

        # the ONLY difference from the shipped model: the backbone sees the
        # coordinates with the graph cut, so g cannot leak cross-point coupling
        # into the derivative -- while its parameters still get full gradient.
        bb_in = source_used.detach() if pointwise_strain else source_used
        source_four_bb = X._homogeneous(bb_in)
        target_four = X._homogeneous(target)

        g = torch.cat((self.ptfeatures(source_four_bb),
                       self.ptfeatures(target_four)), -1)
        g_rep = g.unsqueeze(1).repeat(1, N, 1)
        cat = torch.cat((g_rep, source_used), -1).permute(0, 2, 1)
        disp = self.last_layer(self.list_layers2(cat)).permute(0, 2, 1)
        deformed_source = source + disp

        Ux, Uy, Uz = (disp[:, :, i].reshape(-1, 1) for i in range(3))

        sb = self._stress_backbone()
        g_s = torch.cat((sb(source_four_bb), sb(target_four)), -1)
        g_s_cat = torch.cat((g_s.unsqueeze(1).repeat(1, N, 1), source_used),
                            -1).permute(0, 2, 1)

        S = {}
        for k in ["sxx", "syy", "szz", "sxy", "sxz", "syz"]:
            S[k] = getattr(self, f"pinns_{k}2")(
                getattr(self, f"pinns_{k}1")(g_s_cat)).reshape(-1, 1)

        def d(out, wrt):
            return X.check_derivative(torch.autograd.grad(
                out, wrt, torch.ones_like(out), create_graph=True,
                allow_unused=True)[0], wrt)

        Exx, Eyy, Ezz = d(Ux, x), d(Uy, y), d(Uz, z)
        Exy = 0.5 * (d(Ux, y) + d(Uy, x))
        Exz = 0.5 * (d(Ux, z) + d(Uz, x))
        Eyz = 0.5 * (d(Uy, z) + d(Uz, y))
        mb1 = d(S["sxx"], x) + d(S["sxy"], y) + d(S["sxz"], z)
        mb2 = d(S["sxy"], x) + d(S["syy"], y) + d(S["syz"], z)
        mb3 = d(S["sxz"], x) + d(S["syz"], y) + d(S["szz"], z)

        return {"deformed_source": deformed_source,
                "Ux": Ux, "Uy": Uy, "Uz": Uz,
                "Sxx": S["sxx"], "Syy": S["syy"], "Szz": S["szz"],
                "Sxy": S["sxy"], "Sxz": S["sxz"], "Syz": S["syz"],
                "Exx": Exx, "Eyy": Eyy, "Ezz": Ezz,
                "Exy": Exy, "Exz": Exz, "Eyz": Eyz,
                "mb1": mb1, "mb2": mb2, "mb3": mb3,
                "_disp": disp, "_x": x, "_y": y, "_z": z}


# ===================================================================== #
#  STAGE 0 -- confirm the bug, apply the fix, validate
# ===================================================================== #

def local_gradient(points, disp, k=20):
    """True d(u)/d(x) of a field, by least-squares fit over k nearest points."""
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k)
    G = np.empty((len(points), 3, 3))
    for i in range(len(points)):
        P = points[idx[i]] - points[i]
        D = disp[idx[i]] - disp[i]
        sol, *_ = np.linalg.lstsq(np.hstack([P, np.ones((k, 1))]), D, rcond=None)
        G[i] = sol[:3].T                      # G[i][a][b] = d(u_a)/d(x_b)
    return G


def strain_check(model, batch, pointwise, label, k=20):
    """Compare autograd strain against the true gradient of the model's own field."""
    out = model(batch, pointwise_strain=pointwise) if pointwise is not None else model(batch)
    disp = (out["deformed_source"] - batch[:, 0, :, :]).detach().cpu().numpy()[0]
    src = batch[:, 0, :, :].detach().cpu().numpy()[0]
    n = src.shape[0]
    G = local_gradient(src, disp, k=k)

    res = {"label": label, "k": k}
    for name, comp, (a, b) in [("Exx", "Exx", (0, 0)), ("Eyy", "Eyy", (1, 1)),
                                ("Ezz", "Ezz", (2, 2))]:
        auto = out[comp].detach().cpu().numpy().reshape(-1)[:n]
        true = G[:, a, b]
        res[name] = {
            "corr": float(np.corrcoef(auto, true)[0, 1]),
            "auto_std": float(auto.std()), "true_std": float(true.std()),
            "magnitude_ratio": float(np.abs(auto).max() / max(np.abs(true).max(), 1e-30)),
        }
    return res


def reference_calibration(condition="uniform_pressure", ks=(10, 20, 40, 80)):
    """How good is the least-squares reference itself?

    Run it on the EXACT analytic field, whose gradient is known to be -0.05*I.
    Anything it gets wrong there is its own error, not the network's.

    CAVEAT: the analytic field is exactly LINEAR (u = -0.05 x), and a linear
    least-squares fit reproduces a linear field exactly at any k. So this
    bounds the reference's noise but does NOT measure how much it smooths a
    field with real curvature -- which is the case that matters. Treat it as a
    sanity check, not a full validation of the reference.
    """
    src = np.load(f"toy_datasets/{condition}/{condition}_1.npy")[0].astype(np.float64)
    u = np.load(f"ground_truth/{condition}/{condition}_1_ground_truth.npy").astype(np.float64)
    exact = -0.05
    print(f"\nreference calibration on the exact field (truth: diag {exact}, offdiag 0)")
    print(f"{'k':>4}{'mean diag':>13}{'bias':>12}{'std diag':>12}{'offdiag rms':>14}")
    out = {}
    for k in ks:
        G = local_gradient(src, u, k=k)
        d = np.stack([G[:, 0, 0], G[:, 1, 1], G[:, 2, 2]], 1)
        o = np.stack([G[:, 0, 1], G[:, 0, 2], G[:, 1, 2],
                      G[:, 1, 0], G[:, 2, 0], G[:, 2, 1]], 1)
        out[k] = dict(mean=float(d.mean()), bias=float(d.mean() - exact),
                      std=float(d.std()), offrms=float(np.sqrt((o ** 2).mean())))
        print(f"{k:>4}{d.mean():>13.6f}{d.mean()-exact:>+12.2e}{d.std():>12.3e}"
              f"{np.sqrt((o**2).mean()):>14.3e}")
    print("  (exact on a linear field at every k, as it must be -- see docstring)")
    return out


def stage0(device="cpu", condition="uniform_pressure", ks=(10, 20, 40, 80),
           hidden_dims=(1024, 512, 256, 128, 64)):
    print("=" * 74)
    print("STAGE 0 -- strain computation: confirm the bug, fix it, validate")
    print("=" * 74)

    ds = X.load_dataset(f"toy_datasets/{condition}")
    batch = torch.tensor(ds[0]).unsqueeze(0).float().to(device)
    hidden_dims = list(hidden_dims)

    torch.manual_seed(0)
    broken = X.PhysicsInformedRegistration(hidden_dims=hidden_dims).to(device)
    torch.manual_seed(0)
    fixed = FixedStrainRegistration(hidden_dims=hidden_dims).to(device)
    fixed.load_state_dict(broken.state_dict())      # identical weights
    torch.manual_seed(0)
    fixed2 = FixedStrainRegistrationV2(hidden_dims=hidden_dims).to(device)
    fixed2.load_state_dict(broken.state_dict())

    # --- the shipped 'strain' is a max-pool-winner mask, not a field ---
    ob = broken(batch)
    zero_frac = float((ob["Exx"].detach().cpu().numpy().reshape(-1) == 0.0).mean())
    print(f"\nthe shipped Exx is EXACTLY ZERO at {zero_frac:.1%} of points -- it is")
    print("nonzero only at the points that win a max-pool channel.")

    print("\nIdentical weights, identical displacement field. Only the")
    print("differentiation path differs. Comparing autograd strain against the")
    print("true gradient of each model's own predicted field:\n")

    variants = [("AS SHIPPED", broken, None),
                ("FIXED v1 (detach g)", fixed, True),
                ("FIXED v2 (detach coords)", fixed2, True),
                ("fixed model, fix disabled", fixed, False)]

    print(f"{'variant':<28}{'component':<7}{'corr vs true':>14}{'magnitude ratio':>18}")
    print("-" * 70)
    results = []
    for label, m, pw in variants:
        r = strain_check(m, batch, pw, label, k=20)
        results.append(r)
        for comp in ["Exx", "Eyy", "Ezz"]:
            print(f"{label if comp == 'Exx' else '':<28}{comp:<7}"
                  f"{r[comp]['corr']:>14.4f}{r[comp]['magnitude_ratio']:>18.2f}")
    print("-" * 70)
    print("corr 1.0 = autograd agrees with the true field gradient.")
    print("magnitude ratio 1.0 = same scale; >>1 means contamination.")

    before = np.mean([results[0][c]["corr"] for c in ["Exx", "Eyy", "Ezz"]])
    after = np.mean([results[1][c]["corr"] for c in ["Exx", "Eyy", "Ezz"]])
    print(f"\nmean correlation  before {before:+.4f}   after {after:+.4f}")

    # --- is that number an artefact of the neighbourhood size? ---
    print("\nsensitivity to the least-squares neighbourhood size k:")
    print(f"{'variant':<28}" + "".join(f"{'k=' + str(k):>10}" for k in ks))
    sweep = []
    for label, m, pw in variants:
        row = {"label": label}
        cells = []
        for k in ks:
            r = strain_check(m, batch, pw, label, k=k)
            v = float(np.mean([r[c]["corr"] for c in ["Exx", "Eyy", "Ezz"]]))
            row[f"k{k}"] = v
            cells.append(f"{v:>10.4f}")
        sweep.append(row)
        print(f"{label:<28}" + "".join(cells))

    calib = reference_calibration(condition, ks)

    print("\nCAVEAT ON THIS VALIDATION -- it is partly circular.")
    print("The least-squares reference measures how u varies BETWEEN neighbouring")
    print("points of one cloud, i.e. at fixed global feature g. The fix computes")
    print("the head's derivative at fixed g. Those are the same quantity by")
    print("construction, so agreement mostly confirms the head is smooth, not that")
    print("the strain is physically right. The trained analytic check (--analytic)")
    print("is the one that does not share this assumption.")

    os.makedirs("kaggle/results", exist_ok=True)
    json.dump({"before": before, "after": after, "shipped_frac_exactly_zero": zero_frac,
               "detail": results, "k_sweep": sweep, "reference_calibration": calib},
              open("kaggle/results/stage0_mechanism.json", "w"), indent=1)
    return before, after


def train_model(model, loader, device, epochs, chamfer_weight=10.0, lr=1e-3,
                lame=(82.21, 1.68), log_every=100, pointwise=True):
    """Minimal training loop; mirrors X.train_one_epoch but takes a model instance."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for ep in range(epochs):
        model.train()
        tot = cham = pin = 0.0
        n = 0
        for data in loader:
            data = data.float().to(device)
            opt.zero_grad()
            out = (model(data, pointwise_strain=pointwise)
                   if isinstance(model, (FixedStrainRegistration,
                                         FixedStrainRegistrationV2))
                   else model(data))
            lc = X.chamfer_loss(out["deformed_source"], data[:, 1, :, :], data.shape[2])
            flat = data[:, 0, :, :].reshape(-1, 3)
            l1, l2 = X.get_lame_vectors(flat, lame[0], lame[1], lame[0], lame[1])
            lp = X.linear_elasticity_loss(
                out["Sxx"], out["Syy"], out["Szz"], out["Sxy"], out["Sxz"], out["Syz"],
                out["Exx"], out["Eyy"], out["Ezz"], out["Exy"], out["Exz"], out["Eyz"],
                out["mb1"], out["mb2"], out["mb3"], l1, l2)[0]
            loss = chamfer_weight * lc + lp
            if torch.isnan(loss):
                raise RuntimeError(f"NaN at epoch {ep}")
            loss.backward(retain_graph=True)
            opt.step()
            tot += loss.item(); cham += lc.item(); pin += lp.item(); n += 1
        hist.append((tot / n, cham / n, pin / n))
        if log_every and ep % log_every == 0:
            print(f"    epoch {ep:5d}  total {tot/n:.6f}  chamfer {cham/n:.6f}  pinns {pin/n:.6f}",
                  flush=True)
    return hist



def _analytic_metrics(out, batch, exact, pw, u_gt):
    """Strain metrics + a SEPARATE measure of how good the displacement field is.

    The two must be reported side by side. A wrong strain can mean either that
    the derivative is being computed wrongly (the bug) or that the field itself
    is wrong (undertrained). `field_*` measures the field, `normal_*` the
    derivative, and `lsq_*` is the strain implied by the field itself.
    """
    def col(k):
        return out[k].detach().cpu().numpy().reshape(-1)

    normals = np.concatenate([col("Exx"), col("Eyy"), col("Ezz")])
    shears = np.concatenate([col("Exy"), col("Exz"), col("Eyz")])
    src = batch[:, 0, :, :].detach().cpu().numpy()[0]
    u_pred = (out["deformed_source"] - batch[:, 0, :, :]).detach().cpu().numpy()[0]

    # strain implied by the predicted FIELD (independent of any autograd path)
    G = local_gradient(src, u_pred.astype(np.float64), k=20)
    lsq_normal = np.concatenate([G[:, 0, 0], G[:, 1, 1], G[:, 2, 2]])
    lsq_shear = np.concatenate([0.5 * (G[:, 0, 1] + G[:, 1, 0]),
                                0.5 * (G[:, 0, 2] + G[:, 2, 0]),
                                0.5 * (G[:, 1, 2] + G[:, 2, 1])])

    gt_mag = float(np.linalg.norm(u_gt, axis=1).mean())
    return {
        "normal_mean": float(normals.mean()),
        "normal_median": float(np.median(normals)),
        "normal_std": float(normals.std()),
        "rel_err_vs_exact": float(abs(normals.mean() - exact) / abs(exact)),
        "frac_exactly_zero": float((normals == 0.0).mean()),
        "shear_rms": float(np.sqrt((shears ** 2).mean())),
        "shear_over_exact": float(np.sqrt((shears ** 2).mean()) / abs(exact)),
        # --- field quality, so a bad strain can be attributed correctly ---
        "field_rel_err": float(np.linalg.norm(u_pred - u_gt, axis=1).mean() / gt_mag),
        "field_max_u": float(np.linalg.norm(u_pred, axis=1).max()),
        "lsq_normal_mean": float(lsq_normal.mean()),
        "lsq_normal_std": float(lsq_normal.std()),
        "lsq_shear_rms": float(np.sqrt((lsq_shear ** 2).mean())),
    }


def stage0_analytic(device="cuda", epochs=500, hidden_dims=(1024, 512, 256, 128, 64),
                    dim_k=1024, eval_every=0):
    """Train on uniform_pressure and compare computed strain to the exact answer.

    uniform_pressure is the one condition with a closed form: a uniform
    contraction, Exx=Eyy=Ezz=-P/(3K) and every shear component identically
    zero. With TARGET_MAX_U=0.05 and RADIUS=1.0, P=3K*0.05 so the exact strain
    is -0.05 exactly.

    This is the only check in stage 0 that does NOT use the least-squares
    reference field gradient, so it is independent of the fix's own assumptions.
    """
    import generators._common as C
    cond = "uniform_pressure"
    K = C.bulk_modulus(C.LAMBDA_SOFT, C.MU_SOFT)
    P = 3.0 * K * C.TARGET_MAX_U / C.RADIUS
    exact = -P / (3.0 * K)
    hidden_dims = list(hidden_dims)
    print()
    print(f"analytic strain for {cond}: Exx=Eyy=Ezz={exact:+.6f}, all shear = 0")
    print(f"budget: {epochs} epochs, dim_k={dim_k}, hidden_dims={hidden_dims}, "
          f"device={device}")
    print()

    ds = X.load_dataset("toy_datasets/" + cond)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False,
                                         num_workers=0, drop_last=True)
    batch = torch.tensor(ds[0]).unsqueeze(0).float().to(device)
    u_gt = np.load(f"ground_truth/{cond}/{cond}_1_ground_truth.npy").astype(np.float64)

    rows = []
    variants = [("AS SHIPPED", X.PhysicsInformedRegistration, False),
                ("FIXED", FixedStrainRegistration, True),
                ("FIXED v2", FixedStrainRegistrationV2, True)]
    for label, cls, pw in variants:
        print("  training " + label + " ...", flush=True)
        torch.manual_seed(0)
        m = cls(dim_k=dim_k, hidden_dims=hidden_dims).to(device)
        trace = []
        done = 0
        step = eval_every if eval_every else epochs
        while done < epochs:
            chunk = min(step, epochs - done)
            train_model(m, loader, device, chunk, log_every=0, pointwise=pw)
            done += chunk
            out = m(batch, pointwise_strain=True) if pw else m(batch)
            mt = _analytic_metrics(out, batch, exact, pw, u_gt)
            mt["epoch"] = done
            trace.append(mt)
            print(f"    ep {done:5d}  normal {mt['normal_mean']:+.6f} "
                  f"(exact {exact:+.6f})  shear rms {mt['shear_rms']:.3e}  "
                  f"field rel err {mt['field_rel_err']:.3f}  "
                  f"lsq normal {mt['lsq_normal_mean']:+.6f}", flush=True)
        row = dict(trace[-1]); row["label"] = label; row["trace"] = trace
        rows.append(row)

    print()
    hdr = ("variant".ljust(14) + "normal strain".rjust(15) + "exact".rjust(11)
           + "rel err".rjust(11) + "shear rms".rjust(13) + "field err".rjust(12)
           + "lsq normal".rjust(13))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(r["label"].ljust(14)
              + format(r["normal_mean"], "15.6f")
              + format(exact, "11.6f")
              + format(r["rel_err_vs_exact"], "10.1%")
              + format(r["shear_rms"], "13.3e")
              + format(r["field_rel_err"], "12.3f")
              + format(r["lsq_normal_mean"], "13.6f"))
    print("-" * len(hdr))
    print("exact answer is a uniform contraction: equal normal strains, zero shear.")
    print("'field err' is the mean relative error of the predicted displacement")
    print("field itself and 'lsq normal' the strain that field actually implies --")
    print("both are independent of the autograd path, so they separate 'the")
    print("derivative is wrong' from 'the field is undertrained'.")
    print("NOTE: evaluated on a TRAINING sample; this is a fit check, not")
    print("generalisation.")

    os.makedirs("kaggle/results", exist_ok=True)
    json.dump({"exact": exact, "epochs": epochs, "dim_k": dim_k,
               "hidden_dims": hidden_dims, "device": device, "rows": rows},
              open("kaggle/results/stage0_analytic.json", "w"), indent=1)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--analytic", action="store_true",
                    help="stage 0: also run the trained analytic check")
    ap.add_argument("--skip-mechanism", action="store_true",
                    help="stage 0: skip the untrained mechanism check")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--dim-k", type=int, default=1024)
    ap.add_argument("--hidden-dims", default="1024,512,256,128,64",
                    help="comma-separated trunk widths; use a narrow trunk on CPU")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="stage 0 analytic: report metrics every N epochs")
    a = ap.parse_args()
    if a.stage == 0:
        if not a.skip_mechanism:
            stage0("cpu")
        if a.analytic:
            stage0_analytic(a.device, a.epochs,
                            hidden_dims=[int(v) for v in a.hidden_dims.split(",")],
                            dim_k=a.dim_k, eval_every=a.eval_every)


# ===================================================================== #
#  STAGE 2 -- Neo-Hookean nonlinear elasticity (paper Sec. III, Eq. 16-20)
# ===================================================================== #

J_FLOOR = 1e-3          # clamp: J below this means an inverted/degenerate element
_J_WARNED = {"n": 0}


class NonlinearRegistration(X.PhysicsInformedRegistration):
    """Shipped model, plus the full 9-component displacement gradient.

    REVERTED to the SHIPPED base, not the fixed-strain one. From measurement,
    not preference: with FixedStrainRegistrationV2 the networks did not learn --
    uniform_pressure scored 1.0x better than predicting zero displacement versus
    122x for the shipped model, and final training loss was 1.9e-02 versus
    1.0e-06. Cutting the coordinate path also cuts the physics loss off from the
    PointNet backbone (gradient 3.2e+02 -> 1.7e+00), and that gradient is
    evidently load-bearing for optimisation even though it is not strain.

    CONSEQUENCE, stated plainly: the displacement gradients handed to the
    Neo-Hookean law inherit the shipped model's cross-point contamination, so
    F = I + grad(u) is not the true deformation gradient. Linear and nonlinear
    are affected identically, so the COMPARISON stays fair, but neither is
    evaluating real continuum physics. This reproduces the published code's
    behaviour rather than correcting it.

    The forward below mirrors X.PhysicsInformedRegistration.forward exactly --
    same tensors, same order, `source` (not source_used) into the head, same
    grad_outputs=ones derivatives -- and only additionally returns the x/y/z
    leaf tensors so the nonlinear branch can build the full gradient. The
    shipped file itself is not modified.
    """

    def forward(self, data, pointwise_strain=None, want_full_gradient=False):
        device = data.device
        source = data[:, 0, :, :]
        target = data[:, 1, :, :]

        x = source[:, :, 0].reshape(-1, 1)
        y = source[:, :, 1].reshape(-1, 1)
        z = source[:, :, 2].reshape(-1, 1)
        x.requires_grad_(True)
        y.requires_grad_(True)
        z.requires_grad_(True)

        B, N = data.shape[0], source.shape[1]
        source_used = torch.zeros(B, N, 3, device=device)
        source_used[:, :, 0] = x.reshape(B, -1)
        source_used[:, :, 1] = y.reshape(B, -1)
        source_used[:, :, 2] = z.reshape(B, -1)

        source_four = X._homogeneous(source_used)
        target_four = X._homogeneous(target)

        pf_src = self.ptfeatures(source_four)
        pf_tgt = self.ptfeatures(target_four)
        g = torch.cat((pf_src, pf_tgt), -1)
        g_rep = g.unsqueeze(1).repeat(1, N, 1)
        g_cat = torch.cat((g_rep, source), -1)
        disp_pre = self.list_layers2(g_cat.permute(0, 2, 1))
        disp = self.last_layer(disp_pre).permute(0, 2, 1)
        deformed_source = source + disp

        Ux = disp[:, :, 0].reshape(-1, 1)
        Uy = disp[:, :, 1].reshape(-1, 1)
        Uz = disp[:, :, 2].reshape(-1, 1)

        sb = self._stress_backbone()
        g_s = torch.cat((sb(source_four), sb(target_four)), -1)
        g_s_rep = g_s.unsqueeze(1).repeat(1, N, 1)
        g_s_cat = torch.cat((g_s_rep, source), -1).permute(0, 2, 1)

        S = {}
        for k in ["sxx", "syy", "szz", "sxy", "sxz", "syz"]:
            S[k] = getattr(self, "pinns_" + k + "2")(
                getattr(self, "pinns_" + k + "1")(g_s_cat)).reshape(-1, 1)

        def d(o, w):
            return X.check_derivative(torch.autograd.grad(
                o, w, torch.ones_like(o), create_graph=True, allow_unused=True)[0], w)

        Exx, Eyy, Ezz = d(Ux, x), d(Uy, y), d(Uz, z)
        dUx_dy, dUy_dx = d(Ux, y), d(Uy, x)
        dUx_dz, dUz_dx = d(Ux, z), d(Uz, x)
        dUy_dz, dUz_dy = d(Uy, z), d(Uz, y)
        Exy = 0.5 * (dUx_dy + dUy_dx)
        Exz = 0.5 * (dUx_dz + dUz_dx)
        Eyz = 0.5 * (dUy_dz + dUz_dy)

        mb1 = d(S["sxx"], x) + d(S["sxy"], y) + d(S["sxz"], z)
        mb2 = d(S["sxy"], x) + d(S["syy"], y) + d(S["syz"], z)
        mb3 = d(S["sxz"], x) + d(S["syz"], y) + d(S["szz"], z)

        out = {"deformed_source": deformed_source,
               "Ux": Ux, "Uy": Uy, "Uz": Uz,
               "Sxx": S["sxx"], "Syy": S["syy"], "Szz": S["szz"],
               "Sxy": S["sxy"], "Sxz": S["sxz"], "Syz": S["syz"],
               "Exx": Exx, "Eyy": Eyy, "Ezz": Ezz,
               "Exy": Exy, "Exz": Exz, "Eyz": Eyz,
               "mb1": mb1, "mb2": mb2, "mb3": mb3}

        if want_full_gradient:
            out["grad_u"] = [[Exx, dUx_dy, dUx_dz],
                             [dUy_dx, Eyy, dUy_dz],
                             [dUz_dx, dUz_dy, Ezz]]
        return out


def deformation_gradient(grad_u):
    """F = I + grad(u). Returns a (N, 3, 3) tensor."""
    n = grad_u[0][0].shape[0]
    F = torch.zeros(n, 3, 3, device=grad_u[0][0].device, dtype=grad_u[0][0].dtype)
    for a in range(3):
        for b in range(3):
            F[:, a, b] = grad_u[a][b].reshape(-1)
        F[:, a, a] = F[:, a, a] + 1.0
    return F


def neo_hookean_stress(F, lam, mu, j_floor=J_FLOOR):
    """
    Paper Eq. 18:   sigma = mu * J^-1 * (F F^T - I) + lambda * (J - 1) * I

    J is clamped away from zero; a determinant at or below the floor means the
    element has collapsed or inverted, which the incompressible-ish material
    should never do at 5% strain. Occurrences are counted and warned about.
    """
    J = torch.det(F)
    n_bad = int((J <= j_floor).sum().item())
    if n_bad:
        _J_WARNED["n"] += n_bad
        if _J_WARNED["n"] <= 5 or _J_WARNED["n"] % 1000 == 0:
            print("  WARNING: J <= " + str(j_floor) + " at " + str(n_bad)
                  + " points (cumulative " + str(_J_WARNED["n"])
                  + "); clamping. Element inverted or collapsed.", flush=True)
    J = torch.clamp(J, min=j_floor)

    B = torch.bmm(F, F.transpose(1, 2))                       # left Cauchy-Green
    I = torch.eye(3, device=F.device, dtype=F.dtype).unsqueeze(0)
    Jc = J.reshape(-1, 1, 1)
    return mu.reshape(-1, 1, 1) / Jc * (B - I) + lam.reshape(-1, 1, 1) * (Jc - 1.0) * I, J, B


def neo_hookean_energy(F, J, B, lam, mu):
    """Paper Eq. 20:  mu/2 (tr(F F^T) - 3 - 2 ln J) + lambda/2 (J - 1)^2"""
    trB = B[:, 0, 0] + B[:, 1, 1] + B[:, 2, 2]
    return (mu.reshape(-1) / 2.0 * (trB - 3.0 - 2.0 * torch.log(J))
            + lam.reshape(-1) / 2.0 * (J - 1.0) ** 2)


def nonlinear_loss(out, lam_vec, mu_vec):
    """
    Constitutive residual + static equilibrium + strain energy, the nonlinear
    counterparts of X.linear_elasticity_loss. The predicted stress components
    come from the same six heads; only the target they are matched against and
    the energy expression change.
    """
    F = deformation_gradient(out["grad_u"])
    sigma_nh, J, B = neo_hookean_stress(F, lam_vec, mu_vec)

    pred = {"xx": out["Sxx"], "yy": out["Syy"], "zz": out["Szz"],
            "xy": out["Sxy"], "xz": out["Sxz"], "yz": out["Syz"]}
    idx = {"xx": (0, 0), "yy": (1, 1), "zz": (2, 2),
           "xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}

    mse = torch.nn.MSELoss()
    zero = torch.zeros_like(out["Sxx"])
    constitutive = sum(mse(pred[k].reshape(-1) - sigma_nh[:, a, b], zero.reshape(-1))
                       for k, (a, b) in idx.items())
    momentum = (mse(out["mb1"], zero) + mse(out["mb2"], zero) + mse(out["mb3"], zero))
    energy = neo_hookean_energy(F, J, B, lam_vec, mu_vec).mean()

    total = constitutive + momentum + energy
    return total, constitutive, momentum, energy, J
