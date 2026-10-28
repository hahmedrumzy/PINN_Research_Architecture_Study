"""
Plot training loss vs epoch for every completed run.

    py -3.13 kaggle/plot_losses.py

Reads history.csv from the downloaded Kaggle outputs and writes
kaggle/results/loss_curves_all.png (overlay) and loss_curves_grid.png (per-run).
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SP = (r"C:/Users/haziq/AppData/Local/Temp/claude/"
      r"c--Users-haziq--vscode-Registration-PINNs/"
      r"b79bbacb-d4e8-46ff-a84d-b2491ca106af/scratchpad/results")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "kaggle", "results")

runs = []
for d in sorted(glob.glob(os.path.join(SP, "*"))):
    slug = os.path.basename(d)
    if slug == "pilot":
        continue
    h = glob.glob(os.path.join(d, "**", "history.csv"), recursive=True)
    r = os.path.join(REPO, "kaggle", "results", slug, "result.json")
    if not h or not os.path.exists(r):
        continue
    meta = json.load(open(r))
    hist = np.genfromtxt(h[0], delimiter=",", names=True)
    label = (f"trunk{meta['hidden_dims'].split('_')[1]:>4} "
             f"dimk{meta.get('dim_k', 1024):<4} {meta['activation']:<4} "
             f"{meta['pool'][:4]}")
    runs.append((label, hist, meta))

if not runs:
    raise SystemExit("no history.csv found yet")
print(f"plotting {len(runs)} runs")

# ---- overlay: all runs on shared axes -------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
cmap = plt.cm.tab10(np.linspace(0, 1, max(len(runs), 10)))
for i, (label, hist, meta) in enumerate(sorted(runs, key=lambda r: r[2]["rmse_per_sample"][4])):
    for j, key in enumerate(["total", "chamfer", "pinns"]):
        y = np.clip(hist[key], 1e-9, None)
        ax[j].plot(y, label=label, color=cmap[i], lw=1.2, alpha=.85)
for j, key in enumerate(["total", "chamfer", "pinns"]):
    ax[j].set_yscale("log"); ax[j].set_xlabel("epoch")
    ax[j].set_title(key + " loss"); ax[j].grid(alpha=.3)
ax[0].set_ylabel("loss (log scale)")
ax[2].legend(fontsize=7, loc="upper right", ncol=1, framealpha=.9)
plt.suptitle("Training loss vs epoch - all runs (two_point_poles, 2000 epochs, seed 0)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "loss_curves_all.png"), dpi=130)
print("wrote loss_curves_all.png")

# ---- grid: one panel per run ---------------------------------------------
n = len(runs)
cols = 5
rowsn = int(np.ceil(n / cols))
fig, axes = plt.subplots(rowsn, cols, figsize=(3.4 * cols, 2.9 * rowsn), squeeze=False)
for k, (label, hist, meta) in enumerate(sorted(runs, key=lambda r: r[2]["rmse_per_sample"][4])):
    a = axes[k // cols][k % cols]
    for key, c in [("total", "k"), ("chamfer", "tab:blue"), ("pinns", "tab:red")]:
        a.plot(np.clip(hist[key], 1e-9, None), label=key, color=c, lw=1)
    a.set_yscale("log"); a.grid(alpha=.3)
    a.set_title(f"{label}\nunseen RMSE {meta['rmse_per_sample'][4]:.3e}", fontsize=8)
    if k == 0:
        a.legend(fontsize=7)
for k in range(n, rowsn * cols):
    axes[k // cols][k % cols].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "loss_curves_grid.png"), dpi=130)
print("wrote loss_curves_grid.png")
