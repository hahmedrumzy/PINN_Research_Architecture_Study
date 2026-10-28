"""
Build Stage-2 notebooks: linear vs Neo-Hookean nonlinear elasticity.

Design note -- BATCHED. The previous stage used one notebook per run, and
Kaggle's queue wait (hours) dominated the actual compute (under an hour). Here
each notebook runs all 6 configurations for one condition (2 physics models x 3
seeds) back to back, so 24 runs cost 4 queue waits instead of 24.

Architecture is fixed at the prior sweep's winner: trunk [1024,512,256,128,64],
relu, max pooling, separate backbone, dim_k 1024. That config had the lowest
held-out RMSE of the ten architecture runs (1.644e-03) and is also the cheapest
of the competitive ones at 25.2M parameters.
"""
import json
import os
import re
import uuid

SP = os.path.dirname(os.path.abspath(__file__))

CONDITIONS = [
    ("uniform_pressure", "up"),            # 0.00 non-affine
    ("two_zone_material", "tz"),           # 0.46
    ("two_point_poles", "tp"),             # 0.77
    ("four_points_poles_equator", "fp"),   # 0.885
]
SEEDS = [0, 1, 2]
EPOCHS = 2001


def slugify(t):
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", t.lower())).strip("-")


def _src(s):
    return [l + "\n" for l in s.strip("\n").split("\n")]


def build(condition, tag, outdir):
    cells = []

    def code(s):
        cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
                      "execution_count": None, "outputs": [], "source": _src(s)})

    def md(s):
        cells.append({"cell_type": "markdown", "id": uuid.uuid4().hex[:8],
                      "metadata": {}, "source": _src(s)})

    md("# Stage 2 - " + condition + "\n\nlinear vs Neo-Hookean, 3 seeds each, "
       + str(EPOCHS) + " epochs, trunk 1024 / relu / max.")

    code("CONDITION = " + repr(condition) + "\n"
         + "SEEDS = " + repr(SEEDS) + "\n"
         + "EPOCHS = " + str(EPOCHS))

    code("""
import os, sys, glob, shutil, time, json
import numpy as np
import torch
assert torch.cuda.is_available(), "No GPU - accelerator must be GPU T4 x2"
print("GPU:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
_ = (torch.randn(256, 256, device="cuda") @ torch.randn(256, 256, device="cuda")).sum().item()
print("CUDA OK")

INPUT = "/kaggle/input"
cands = glob.glob(os.path.join(INPUT, "**", "toy_datasets"), recursive=True)
assert cands, "toy_datasets not found; saw " + str([r for r, d, f in os.walk(INPUT)][:30])
ROOT = os.path.dirname(cands[0])
WORK = "/kaggle/working/data"
os.makedirs(WORK, exist_ok=True)
for n in ["toy_datasets", "ground_truth"]:
    if not os.path.exists(os.path.join(WORK, n)):
        shutil.copytree(os.path.join(ROOT, n), os.path.join(WORK, n))
for f in ["experiment_architecture_combined.py", "experiment_full_study.py"]:
    hit = glob.glob(os.path.join(INPUT, "**", f), recursive=True)
    assert hit, f + " not found in dataset"
    shutil.copy(hit[0], "/kaggle/working/" + f)
gen = glob.glob(os.path.join(INPUT, "**", "generators"), recursive=True)
if gen and not os.path.exists("/kaggle/working/generators"):
    shutil.copytree(gen[0], "/kaggle/working/generators")

DATA = os.path.join(WORK, "toy_datasets", CONDITION)
GT = os.path.join(WORK, "ground_truth", CONDITION)
files = sorted(glob.glob(DATA + "/*.npy"))
assert len(files) == 5, "expected 5 samples, got " + str(len(files))
print("condition:", CONDITION, "| samples:", len(files))
""")

    code("""
sys.path.insert(0, "/kaggle/working")
import experiment_architecture_combined as X
import experiment_full_study as F

# EXPLICIT held-out split, asserted rather than assumed
ds = X.load_dataset(DATA)
loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False,
                                     num_workers=0, drop_last=True)
seen_n = sum(b.shape[0] for b in loader)
assert seen_n == 4, "expected 4 training samples, got " + str(seen_n)
print("held-out split asserted: train on files 1-4, score file 5 separately")

LAM, MU = 82.21, 1.68
HIDDEN = [1024, 512, 256, 128, 64]


def score(model):
    errs = []
    for f in files:
        a = np.load(f).astype(np.float32)
        gt = np.load(os.path.join(GT, os.path.basename(f).replace(".npy", "_ground_truth.npy")))
        b = torch.tensor(a).unsqueeze(0).cuda()
        out = model(b)
        pred = (out["deformed_source"] - b[:, 0, :, :]).detach().cpu().numpy()[0]
        errs.append(float(np.sqrt(((pred - gt) ** 2).sum(1).mean())))
    return errs


def run_one(model_kind, seed):
    torch.manual_seed(seed)
    if model_kind == "linear":
        m = X.PhysicsInformedRegistration(hidden_dims=HIDDEN).cuda()
    else:
        m = F.NonlinearRegistration(hidden_dims=HIDDEN).cuda()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    hist = []
    t0 = time.time()
    for ep in range(EPOCHS):
        m.train()
        tot = 0.0
        cnt = 0
        for data in loader:
            data = data.float().cuda()
            opt.zero_grad()
            if model_kind == "linear":
                out = m(data)
                flat = data[:, 0, :, :].reshape(-1, 3)
                l1, l2 = X.get_lame_vectors(flat, LAM, MU, LAM, MU)
                phys = X.linear_elasticity_loss(
                    out["Sxx"], out["Syy"], out["Szz"], out["Sxy"], out["Sxz"], out["Syz"],
                    out["Exx"], out["Eyy"], out["Ezz"], out["Exy"], out["Exz"], out["Eyz"],
                    out["mb1"], out["mb2"], out["mb3"], l1, l2)[0]
            else:
                out = m(data, want_full_gradient=True)
                npt = out["Ux"].shape[0]
                lam_v = torch.full((npt,), LAM, device="cuda")
                mu_v = torch.full((npt,), MU, device="cuda")
                phys = F.nonlinear_loss(out, lam_v, mu_v)[0]
            lc = X.chamfer_loss(out["deformed_source"], data[:, 1, :, :], data.shape[2])
            loss = 10.0 * lc + phys
            if torch.isnan(loss):
                raise RuntimeError("NaN at epoch " + str(ep))
            loss.backward(retain_graph=True)
            opt.step()
            tot += loss.item()
            cnt += 1
        hist.append(tot / cnt)
        if ep % 200 == 0:
            print("    " + model_kind + " seed " + str(seed) + " ep " + str(ep)
                  + "  loss " + format(tot / cnt, ".6f"), flush=True)
    errs = score(m)
    return {"model": model_kind, "seed": seed,
            "rmse_seen": float(np.mean(errs[:4])), "rmse_unseen": float(errs[4]),
            "rmse_per_sample": errs, "final_loss": hist[-1],
            "hours": (time.time() - t0) / 3600.0, "history": hist}
""")

    code("""
zero = [np.sqrt((np.load(os.path.join(GT, os.path.basename(f).replace(".npy",
       "_ground_truth.npy"))) ** 2).sum(1).mean()) for f in files]
ZERO_RMSE = float(np.mean(zero))
print("predict-zero baseline RMSE:", format(ZERO_RMSE, ".6e"))
print()

results = []
for kind in ["linear", "nonlinear"]:
    for sd in SEEDS:
        print("=== " + kind + " seed " + str(sd) + " ===", flush=True)
        r = run_one(kind, sd)
        results.append(r)
        print("    -> unseen RMSE " + format(r["rmse_unseen"], ".6e")
              + "   (" + format(r["hours"], ".2f") + " h)", flush=True)
        json.dump({"condition": CONDITION, "epochs": EPOCHS,
                   "rmse_predict_zero": ZERO_RMSE, "results": results},
                  open("/kaggle/working/stage2_" + CONDITION + ".json", "w"), indent=1)
""")

    code("""
print()
print("=" * 62)
print("CONDITION: " + CONDITION)
print("model".ljust(12) + "seed".rjust(6) + "SEEN".rjust(13) + "UNSEEN".rjust(13))
print("-" * 62)
for r in results:
    print(r["model"].ljust(12) + str(r["seed"]).rjust(6)
          + format(r["rmse_seen"], "13.4e") + format(r["rmse_unseen"], "13.4e"))
print("-" * 62)
for kind in ["linear", "nonlinear"]:
    u = [r["rmse_unseen"] for r in results if r["model"] == kind]
    if u:
        print(kind.ljust(12) + "mean " + format(np.mean(u), ".4e")
              + "   spread " + format(min(u), ".4e") + " - " + format(max(u), ".4e"))
lin = [r["rmse_unseen"] for r in results if r["model"] == "linear"]
non = [r["rmse_unseen"] for r in results if r["model"] == "nonlinear"]
if lin and non:
    sep = (max(lin) < min(non)) or (max(non) < min(lin))
    print()
    print("seed ranges overlap: " + str(not sep)
          + ("  -> difference is NOT resolvable" if not sep else "  -> separated"))
""")

    nb = {"cells": cells,
          "metadata": {"kernelspec": {"language": "python", "display_name": "Python 3",
                                       "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11"}},
          "nbformat": 4, "nbformat_minor": 5}
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "nb.ipynb")
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)

    title = "S2 " + tag
    slug = slugify(title)
    meta = {"id": "haziqahmedrumzy/" + slug, "title": title,
            "code_file": "nb.ipynb", "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": True, "enable_internet": False,
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": ["haziqahmedrumzy/pinn-sphere-conditions"],
            "competition_sources": [], "kernel_sources": []}
    with open(os.path.join(outdir, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=1)

    chk = json.load(open(path))
    for i, c in enumerate(chk["cells"]):
        if c["cell_type"] != "code":
            continue
        assert len(c["source"]) > 1, "cell %d collapsed" % i
        compile("".join(c["source"]), "<%s cell %d>" % (slug, i), "exec")
    return slug


if __name__ == "__main__":
    for cond, tag in CONDITIONS:
        d = os.path.join(SP, "notebooks", "nb_s2-" + tag)
        s = build(cond, tag, d)
        print("built " + s + "  (" + cond + ")")
