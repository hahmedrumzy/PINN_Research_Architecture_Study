"""
Build ONE self-contained Colab notebook covering all four Stage-2 conditions.

Design for session fragility: conditions run in priority order and the full
result JSON is printed after EACH one completes. A dropped session therefore
still leaves usable output in the notebook for everything finished so far,
rather than losing the lot.

Order: two_point_poles first (every other measurement in the study is on that
condition, so it slots straight into existing comparisons), then the most
asymmetric, then the middle, then the trivial one last.
"""
import base64
import glob
import io
import json
import os
import uuid
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "colab")

ORDER = ["two_point_poles", "four_points_poles_equator",
         "two_zone_material", "uniform_pressure"]
SEEDS = [0, 1, 2]
EPOCHS = 2001          # measured: w1024 is within 20% of final at ep 890, 5% at ep 1828


def payload():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for mod in ["experiment_architecture_combined.py", "experiment_full_study.py"]:
            z.write(os.path.join(REPO, mod), mod)
        for cond in ORDER:
            for d in ["toy_datasets", "ground_truth"]:
                for f in glob.glob(os.path.join(REPO, d, cond, "*.npy")):
                    z.write(f, os.path.join(d, cond, os.path.basename(f)))
    return base64.b64encode(buf.getvalue()).decode()


def _src(s):
    return [l + "\n" for l in s.strip("\n").split("\n")]


cells = []


def code(s):
    cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
                  "execution_count": None, "outputs": [], "source": _src(s)})


def md(s):
    cells.append({"cell_type": "markdown", "id": uuid.uuid4().hex[:8],
                  "metadata": {}, "source": _src(s)})


md("# Stage 2 - all four conditions\n\n"
   "Linear vs Neo-Hookean elasticity. 4 conditions x 2 models x 3 seeds = 24 runs,\n"
   + str(EPOCHS) + " epochs each. Roughly 9 hours on a Colab T4.\n\n"
   "**Before running: Runtime -> Change runtime type -> T4 GPU.**\n\n"
   "Then Runtime -> Run all, and leave the tab open.\n\n"
   "Results print after EACH condition finishes, so if the session drops you\n"
   "still have everything completed up to that point - just copy what printed.")

code("import torch\n"
     "assert torch.cuda.is_available(), 'No GPU. Runtime -> Change runtime type -> T4 GPU'\n"
     "print('GPU:', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))\n"
     "_ = (torch.randn(256, 256, device='cuda') @ torch.randn(256, 256, device='cuda')).sum().item()\n"
     "print('CUDA kernel launch OK')")

code("# all data + code embedded; nothing to upload\n"
     "import base64, io, zipfile, os\n"
     "PAYLOAD = '''" + payload() + "'''\n"
     "zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))).extractall('/content/work')\n"
     "os.chdir('/content/work')\n"
     "print('conditions:', sorted(os.listdir('toy_datasets')))")

code("ORDER = " + repr(ORDER) + "\n"
     "SEEDS = " + repr(SEEDS) + "\n"
     "EPOCHS = " + str(EPOCHS))

code("""
import sys, glob, time, json
import numpy as np
sys.path.insert(0, '/content/work')
import experiment_architecture_combined as X
import experiment_full_study as F

LAM, MU = 82.21, 1.68
HIDDEN = [1024, 512, 256, 128, 64]


def setup(condition):
    data = 'toy_datasets/' + condition
    gt = 'ground_truth/' + condition
    files = sorted(glob.glob(data + '/*.npy'))
    assert len(files) == 5, condition + ': expected 5 samples, got ' + str(len(files))
    ds = X.load_dataset(data)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False,
                                         num_workers=0, drop_last=True)
    seen_n = sum(b.shape[0] for b in loader)
    assert seen_n == 4, condition + ': expected 4 training samples, got ' + str(seen_n)
    zero = [np.sqrt((np.load(os.path.join(gt, os.path.basename(f).replace('.npy',
            '_ground_truth.npy'))) ** 2).sum(1).mean()) for f in files]
    return loader, files, gt, float(np.mean(zero))


def score(model, files, gt):
    errs = []
    for f in files:
        a = np.load(f).astype(np.float32)
        g = np.load(os.path.join(gt, os.path.basename(f).replace('.npy', '_ground_truth.npy')))
        b = torch.tensor(a).unsqueeze(0).cuda()
        out = model(b)
        pred = (out['deformed_source'] - b[:, 0, :, :]).detach().cpu().numpy()[0]
        errs.append(float(np.sqrt(((pred - g) ** 2).sum(1).mean())))
    return errs


def run_one(kind, seed, loader, files, gt):
    torch.manual_seed(seed)
    m = (X.PhysicsInformedRegistration(hidden_dims=HIDDEN).cuda() if kind == 'linear'
         else F.NonlinearRegistration(hidden_dims=HIDDEN).cuda())
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    t0 = time.time()
    last = 0.0
    for ep in range(EPOCHS):
        m.train()
        tot = 0.0
        cnt = 0
        for data in loader:
            data = data.float().cuda()
            opt.zero_grad()
            if kind == 'linear':
                out = m(data)
                flat = data[:, 0, :, :].reshape(-1, 3)
                l1, l2 = X.get_lame_vectors(flat, LAM, MU, LAM, MU)
                phys = X.linear_elasticity_loss(
                    out['Sxx'], out['Syy'], out['Szz'], out['Sxy'], out['Sxz'], out['Syz'],
                    out['Exx'], out['Eyy'], out['Ezz'], out['Exy'], out['Exz'], out['Eyz'],
                    out['mb1'], out['mb2'], out['mb3'], l1, l2)[0]
            else:
                out = m(data, want_full_gradient=True)
                n = out['Ux'].shape[0]
                phys = F.nonlinear_loss(out,
                                        torch.full((n,), LAM, device='cuda'),
                                        torch.full((n,), MU, device='cuda'))[0]
            lc = X.chamfer_loss(out['deformed_source'], data[:, 1, :, :], data.shape[2])
            loss = 10.0 * lc + phys
            if torch.isnan(loss):
                raise RuntimeError('NaN at epoch ' + str(ep))
            loss.backward(retain_graph=True)
            opt.step()
            tot += loss.item()
            cnt += 1
        last = tot / cnt
        if ep % 250 == 0:
            print('      ep ' + str(ep) + '  loss ' + format(last, '.6f'), flush=True)
    errs = score(m, files, gt)
    del m, opt
    torch.cuda.empty_cache()
    return {'model': kind, 'seed': seed,
            'rmse_seen': float(np.mean(errs[:4])), 'rmse_unseen': float(errs[4]),
            'rmse_per_sample': errs, 'final_loss': last,
            'hours': (time.time() - t0) / 3600.0}
""")

code("""
ALL = {}
t_start = time.time()

for ci, cond in enumerate(ORDER):
    print()
    print('#' * 66)
    print('# CONDITION ' + str(ci + 1) + '/' + str(len(ORDER)) + ': ' + cond)
    print('#' * 66, flush=True)
    loader, files, gt, zero_rmse = setup(cond)
    print('  predict-zero baseline: ' + format(zero_rmse, '.6e'), flush=True)

    results = []
    for kind in ['linear', 'nonlinear']:
        for sd in SEEDS:
            print('   === ' + kind + ' seed ' + str(sd) + ' ===', flush=True)
            r = run_one(kind, sd, loader, files, gt)
            results.append(r)
            print('      -> unseen RMSE ' + format(r['rmse_unseen'], '.6e')
                  + '   (' + format(r['hours'], '.2f') + ' h)', flush=True)

    ALL[cond] = {'condition': cond, 'epochs': EPOCHS,
                 'rmse_predict_zero': zero_rmse, 'results': results}

    lin = [r['rmse_unseen'] for r in results if r['model'] == 'linear']
    non = [r['rmse_unseen'] for r in results if r['model'] == 'nonlinear']
    overlap = not (max(lin) < min(non) or max(non) < min(lin))
    print()
    print('  ' + cond + ' SUMMARY')
    print('    linear     mean ' + format(np.mean(lin), '.4e')
          + '  range ' + format(min(lin), '.4e') + ' - ' + format(max(lin), '.4e'))
    print('    nonlinear  mean ' + format(np.mean(non), '.4e')
          + '  range ' + format(min(non), '.4e') + ' - ' + format(max(non), '.4e'))
    print('    ranges overlap: ' + str(overlap)
          + ('  -> NOT resolvable' if overlap else '  -> SEPARATED'))
    print('    elapsed so far: ' + format((time.time() - t_start) / 3600.0, '.2f') + ' h')
    print()
    print('  ---- COPY THIS LINE (partial results, safe if session drops) ----')
    print(json.dumps(ALL))
    print('  ----------------------------------------------------------------', flush=True)

print()
print('ALL CONDITIONS DONE in ' + format((time.time() - t_start) / 3600.0, '.2f') + ' h')
""")

code("""
print('=' * 70)
print('FINAL - linear vs nonlinear across all conditions')
print('condition'.ljust(28) + 'linear mean'.rjust(13) + 'nonlin mean'.rjust(13)
      + 'ratio'.rjust(8) + 'overlap'.rjust(9))
print('-' * 70)
for cond, d in ALL.items():
    lin = [r['rmse_unseen'] for r in d['results'] if r['model'] == 'linear']
    non = [r['rmse_unseen'] for r in d['results'] if r['model'] == 'nonlinear']
    ov = not (max(lin) < min(non) or max(non) < min(lin))
    print(cond.ljust(28) + format(np.mean(lin), '13.4e') + format(np.mean(non), '13.4e')
          + format(np.mean(non) / np.mean(lin), '8.3f') + str(ov).rjust(9))
print('-' * 70)
print()
print('=' * 70)
print('COPY EVERYTHING BELOW THIS LINE BACK')
print('=' * 70)
print(json.dumps(ALL))
""")

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

os.makedirs(OUT, exist_ok=True)
path = os.path.join(OUT, "stage2_ALL.ipynb")
with open(path, "w") as f:
    json.dump(nb, f)

chk = json.load(open(path))
for i, c in enumerate(chk["cells"]):
    if c["cell_type"] == "code":
        assert len(c["source"]) > 1, "cell %d collapsed" % i
        compile("".join(c["source"]), "<cell %d>" % i, "exec")
print("built " + os.path.basename(path) + "  ("
      + str(os.path.getsize(path) // 1024) + " KB, " + str(len(cells)) + " cells)")
