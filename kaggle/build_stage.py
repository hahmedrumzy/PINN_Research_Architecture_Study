import json, os, sys, uuid
SP = os.path.dirname(os.path.abspath(__file__))

def build(slug, title, condition, hd, act, pool, epochs, seed, outdir, dim_k=1024, custom_dims=None):
    cells = []
    def _src(s): return [l + "\n" for l in s.strip("\n").split("\n")]
    def code(s): cells.append({"cell_type":"code","id":uuid.uuid4().hex[:8],"metadata":{},
                               "execution_count":None,"outputs":[],"source":_src(s)})
    def md(s):   cells.append({"cell_type":"markdown","id":uuid.uuid4().hex[:8],
                               "metadata":{},"source":_src(s)})

    md("# " + title + "\n\ncondition=" + condition + "  width=" + hd +
       "  act=" + act + "  pool=" + pool + "  epochs=" + str(epochs) + "  seed=" + str(seed))

    code("""
CONDITION = "%s"
HD, ACT, POOL = "%s", "%s", "%s"
EPOCHS, SEED = %d, %d
DIM_K = %d
CUSTOM_DIMS = %s
""" % (condition, hd, act, pool, epochs, seed, dim_k, repr(custom_dims)))

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
if not cands:
    raise RuntimeError("toy_datasets not found; saw " + str([r for r,d,f in os.walk(INPUT)][:30]))
ROOT = os.path.dirname(cands[0])
WORK = "/kaggle/working/data"
os.makedirs(WORK, exist_ok=True)
for n in ["toy_datasets", "ground_truth"]:
    if not os.path.exists(os.path.join(WORK, n)):
        shutil.copytree(os.path.join(ROOT, n), os.path.join(WORK, n))
hits = glob.glob(os.path.join(INPUT, "**", "experiment_architecture_combined.py"), recursive=True)
shutil.copy(hits[0], "/kaggle/working/experiment_architecture_combined.py")
DATA = os.path.join(WORK, "toy_datasets", CONDITION)
GT = os.path.join(WORK, "ground_truth", CONDITION)
OUT = "/kaggle/working/run"
print("data:", DATA, "| samples:", len(glob.glob(DATA + "/*.npy")))
""")

    code("""
# Run in-process so the weight-init seed can be overridden without editing the
# training script (its torch.manual_seed(0) is hardcoded at line 578).
sys.path.insert(0, "/kaggle/working")
import experiment_architecture_combined as X

_orig_seed = torch.manual_seed
torch.manual_seed = lambda _ignored: _orig_seed(SEED)
print("weight-init seed overridden to", SEED)

# register a trunk size the script does not ship with (no file edit needed:
# parse_args reads HIDDEN_DIM_CONFIGS at call time, and we inject before main)
if CUSTOM_DIMS is not None:
    X.HIDDEN_DIM_CONFIGS[HD] = CUSTOM_DIMS
    print("registered", HD, "=", CUSTOM_DIMS)
print("dim_k =", DIM_K)

# EXPLICIT held-out split: assert file 5 is never trained on, rather than
# relying on the batch_size=2 + drop_last accident staying true.
_files = sorted(glob.glob(os.path.join(DATA, "*.npy")))
assert len(_files) == 5, "expected 5 samples, got " + str(len(_files))
_ds = X.load_dataset(DATA)
_loader = torch.utils.data.DataLoader(_ds, batch_size=2, shuffle=False,
                                      num_workers=0, drop_last=True)
_seen = sum(b.shape[0] for b in _loader)
assert _seen == 4, "expected 4 training samples, got " + str(_seen)
print("held-out split asserted: train on files 1-4, score file 5 separately")

sys.argv = ["run", "--traindata-path", DATA, "--testdata-path", DATA,
            "--outdir", OUT, "--device", "cuda", "--epochs", str(EPOCHS),
            "--batch-size", "2", "--mode", "full_grid", "--dim-k", str(DIM_K),
            "--save-every", str(EPOCHS - 1),
            "--hidden-dims-subset", HD, "--activations-subset", ACT,
            "--poolings-subset", POOL]
t0 = time.time()
X.main()
torch.manual_seed = _orig_seed
print("elapsed hours:", round((time.time() - t0) / 3600, 3))
""")

    code("""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

run_dir = glob.glob(os.path.join(OUT, "hd=*"))[0]
hist = np.genfromtxt(os.path.join(run_dir, "history.csv"), delimiter=",",
                     names=True, dtype=None, encoding=None)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for i, k in enumerate(["total", "chamfer", "pinns"]):
    ax[i].plot(hist[k]); ax[i].set_yscale("log")
    ax[i].set_title(k); ax[i].set_xlabel("epoch"); ax[i].grid(alpha=.3)
plt.suptitle(os.path.basename(run_dir) + "  seed=" + str(SEED))
plt.tight_layout(); plt.savefig("/kaggle/working/loss_curves.png", dpi=110)
plt.show()

n = len(hist["total"])
print("epochs run:", n)
for frac in [0.25, 0.5, 0.75, 1.0]:
    i = int(n * frac) - 1
    print("  at %5d: total %.6f  chamfer %.6f  pinns %.6f"
          % (i + 1, hist["total"][i], hist["chamfer"][i], hist["pinns"][i]))
last10 = hist["total"][-max(1, n // 10):]
print("plateaued?  last-10%% relative spread = %.3e" % (last10.std() / abs(last10.mean())))
""")

    code("""
files = sorted(glob.glob(DATA + "/*.npy"))
zero = [np.sqrt((np.load(os.path.join(GT, os.path.basename(f).replace(".npy",
        "_ground_truth.npy"))) ** 2).sum(1).mean()) for f in files]
ZERO_RMSE = float(np.mean(zero))

model = X.PhysicsInformedRegistration(dim_k=DIM_K, hidden_dims=X.HIDDEN_DIM_CONFIGS[HD],
                                      activation=ACT, pool_type=POOL).cuda()
model.load_state_dict(torch.load(os.path.join(run_dir, "model_last.pth"), map_location="cuda"))
model.eval()
nparams = sum(p.numel() for p in model.parameters())

errs = []
for f in files:
    a = np.load(f).astype(np.float32)
    gt = np.load(os.path.join(GT, os.path.basename(f).replace(".npy", "_ground_truth.npy")))
    b = torch.tensor(a).unsqueeze(0).cuda()
    pred = (model(b)["deformed_source"] - b[:, 0, :, :]).detach().cpu().numpy()[0]
    errs.append(float(np.sqrt(((pred - gt) ** 2).sum(1).mean())))

res = {"condition": CONDITION, "dim_k": DIM_K, "hidden_dims": HD, "activation": ACT, "pool": POOL,
       "seed": SEED, "epochs": EPOCHS, "params_M": nparams / 1e6,
       "rmse_mean": float(np.mean(errs)), "rmse_per_sample": errs,
       "rmse_seen": float(np.mean(errs[:4])), "rmse_unseen": float(errs[4]),
       "overfit_gap": float(errs[4] / np.mean(errs[:4]) - 1.0),
       "rmse_predict_zero": ZERO_RMSE,
       "final_total": float(hist["total"][-1]),
       "final_chamfer": float(hist["chamfer"][-1]),
       "final_pinns": float(hist["pinns"][-1])}
with open("/kaggle/working/result.json", "w") as fh:
    json.dump(res, fh, indent=1)

print(json.dumps(res, indent=1))
print()
print("RMSE vs ground truth :", format(res["rmse_mean"], ".6e"))
print("RMSE predicting zero :", format(ZERO_RMSE, ".6e"))
print("BEATS DOING NOTHING? ", res["rmse_mean"] < ZERO_RMSE)
""")

    nb = {"cells": cells,
          "metadata": {"kernelspec": {"language":"python","display_name":"Python 3","name":"python3"},
                       "language_info": {"name":"python","version":"3.11"}},
          "nbformat": 4, "nbformat_minor": 5}
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "nb.ipynb")
    with open(path, "w") as f: json.dump(nb, f, indent=1)
    meta = {"id": "haziqahmedrumzy/" + slug, "title": title,
            "code_file": "nb.ipynb", "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": True, "enable_internet": False,
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": ["haziqahmedrumzy/pinn-sphere-conditions"],
            "competition_sources": [], "kernel_sources": []}
    with open(os.path.join(outdir, "kernel-metadata.json"), "w") as f: json.dump(meta, f, indent=1)
    chk = json.load(open(path))
    for i, c in enumerate(chk["cells"]):
        if c["cell_type"] != "code": continue
        assert len(c["source"]) > 1, "cell %d collapsed" % i
        compile("".join(c["source"]), "<%s cell %d>" % (slug, i), "exec")
    return path

if __name__ == "__main__":
    STAGE1 = [
        ("pinn-s1-w1024-relu-max",  "S1 w1024 relu max",  "baseline_1024", "relu", "max"),
        ("pinn-s1-w2048-relu-max",  "S1 w2048 relu max",  "expanded_2048", "relu", "max"),
        ("pinn-s1-w4096-relu-max",  "S1 w4096 relu max",  "expanded_4096", "relu", "max"),
        ("pinn-s1-w1024-tanh-max",  "S1 w1024 tanh max",  "baseline_1024", "tanh", "max"),
        ("pinn-s1-w1024-sin-max",   "S1 w1024 sin max",   "baseline_1024", "sin",  "max"),
        ("pinn-s1-w1024-relu-attn", "S1 w1024 relu attn", "baseline_1024", "relu", "attention"),
    ]
    for slug, title, hd, act, pool in STAGE1:
        d = os.path.join(SP, "nb_" + slug)
        build(slug, title, "two_point_poles", hd, act, pool, 2000, 0, d)
        print("built", slug)
