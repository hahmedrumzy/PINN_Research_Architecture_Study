"""
Build self-contained Colab notebooks for Stage 2 (linear vs Neo-Hookean).

Each notebook embeds its own data and code as a base64 zip, so there is nothing
to upload, no Kaggle token, no Drive mount. Open in Colab, set the runtime to
GPU, Run all.

One notebook per condition, ~2 h each on a Colab T4, which keeps every run well
inside a free-tier session. Results are printed as JSON at the end so they can
be copied straight back.
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

CONDITIONS = [
    ("uniform_pressure", "up"),
    ("two_zone_material", "tz"),
    ("two_point_poles", "tp"),
    ("four_points_poles_equator", "fp"),
]
SEEDS = [0, 1, 2]
EPOCHS = 2001          # measured: w1024 is within 20% of final at ep 890, 5% at ep 1828


def payload(condition):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for mod in ["experiment_architecture_combined.py", "experiment_full_study.py"]:
            z.write(os.path.join(REPO, mod), mod)
        for d in ["toy_datasets", "ground_truth"]:
            for f in glob.glob(os.path.join(REPO, d, condition, "*.npy")):
                z.write(f, os.path.join(d, condition, os.path.basename(f)))
    return base64.b64encode(buf.getvalue()).decode()


def _src(s):
    return [l + "\n" for l in s.strip("\n").split("\n")]


def build(condition, tag):
    cells = []

    def code(s):
        cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
                      "execution_count": None, "outputs": [], "source": _src(s)})

    def md(s):
        cells.append({"cell_type": "markdown", "id": uuid.uuid4().hex[:8],
                      "metadata": {}, "source": _src(s)})

    md("# Stage 2 - " + condition + "\n\n"
       "Linear vs Neo-Hookean elasticity, 3 seeds each, " + str(EPOCHS) + " epochs.\n\n"
       "**Before running: Runtime -> Change runtime type -> T4 GPU.**\n\n"
       "Then Runtime -> Run all. Takes roughly 2 hours. Results print as JSON at "
       "the end - copy that back.")

    code("import torch\n"
         "assert torch.cuda.is_available(), 'No GPU. Runtime -> Change runtime type -> T4 GPU'\n"
         "print('GPU:', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))\n"
         "print('torch', torch.__version__)\n"
         "_ = (torch.randn(256, 256, device='cuda') @ torch.randn(256, 256, device='cuda')).sum().item()\n"
         "print('CUDA kernel launch OK')")

    code("# data + code are embedded below; nothing to upload\n"
         "import base64, io, zipfile, os\n"
         "PAYLOAD = '''" + payload(condition) + "'''\n"
         "zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))).extractall('/content/work')\n"
         "os.chdir('/content/work')\n"
         "print('unpacked:', sorted(os.listdir('.')))\n"
         "print('samples :', len(os.listdir('toy_datasets/" + condition + "')))")

    code("CONDITION = " + repr(condition) + "\n"
         "SEEDS = " + repr(SEEDS) + "\n"
         "EPOCHS = " + str(EPOCHS))

    code("""
import sys, glob, time, json
import numpy as np
sys.path.insert(0, '/content/work')
import experiment_architecture_combined as X
import experiment_full_study as F

DATA = 'toy_datasets/' + CONDITION
GT = 'ground_truth/' + CONDITION
files = sorted(glob.glob(DATA + '/*.npy'))
assert len(files) == 5, 'expected 5 samples, got ' + str(len(files))

ds = X.load_dataset(DATA)
loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False,
                                     num_workers=0, drop_last=True)
seen_n = sum(b.shape[0] for b in loader)
assert seen_n == 4, 'expected 4 training samples, got ' + str(seen_n)
print('held-out split asserted: train on files 1-4, score file 5 separately')

LAM, MU = 82.21, 1.68
HIDDEN = [1024, 512, 256, 128, 64]

zero = [np.sqrt((np.load(os.path.join(GT, os.path.basename(f).replace('.npy',
        '_ground_truth.npy'))) ** 2).sum(1).mean()) for f in files]
ZERO_RMSE = float(np.mean(zero))
print('predict-zero baseline RMSE:', format(ZERO_RMSE, '.6e'))


def score(model):
    errs = []
    for f in files:
        a = np.load(f).astype(np.float32)
        gt = np.load(os.path.join(GT, os.path.basename(f).replace('.npy', '_ground_truth.npy')))
        b = torch.tensor(a).unsqueeze(0).cuda()
        out = model(b)
        pred = (out['deformed_source'] - b[:, 0, :, :]).detach().cpu().numpy()[0]
        errs.append(float(np.sqrt(((pred - gt) ** 2).sum(1).mean())))
    return errs


def run_one(kind, seed):
    torch.manual_seed(seed)
    m = (X.PhysicsInformedRegistration(hidden_dims=HIDDEN).cuda() if kind == 'linear'
         else F.NonlinearRegistration(hidden_dims=HIDDEN).cuda())
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
        hist.append(tot / cnt)
        if ep % 250 == 0:
            print('    ' + kind + ' seed ' + str(seed) + ' ep ' + str(ep)
                  + '  loss ' + format(tot / cnt, '.6f'), flush=True)
    errs = score(m)
    return {'model': kind, 'seed': seed,
            'rmse_seen': float(np.mean(errs[:4])), 'rmse_unseen': float(errs[4]),
            'rmse_per_sample': errs, 'final_loss': hist[-1],
            'hours': (time.time() - t0) / 3600.0}
""")

    code("""
results = []
for kind in ['linear', 'nonlinear']:
    for sd in SEEDS:
        print('=== ' + kind + ' seed ' + str(sd) + ' ===', flush=True)
        r = run_one(kind, sd)
        results.append(r)
        print('    -> unseen RMSE ' + format(r['rmse_unseen'], '.6e')
              + '   (' + format(r['hours'], '.2f') + ' h)', flush=True)
        json.dump({'condition': CONDITION, 'epochs': EPOCHS,
                   'rmse_predict_zero': ZERO_RMSE, 'results': results},
                  open('/content/stage2_' + CONDITION + '.json', 'w'), indent=1)
print('done')
""")

    code("""
print('=' * 62)
print('CONDITION: ' + CONDITION)
print('model'.ljust(12) + 'seed'.rjust(6) + 'SEEN'.rjust(13) + 'UNSEEN'.rjust(13))
print('-' * 62)
for r in results:
    print(r['model'].ljust(12) + str(r['seed']).rjust(6)
          + format(r['rmse_seen'], '13.4e') + format(r['rmse_unseen'], '13.4e'))
print('-' * 62)
lin = [r['rmse_unseen'] for r in results if r['model'] == 'linear']
non = [r['rmse_unseen'] for r in results if r['model'] == 'nonlinear']
print('linear     mean ' + format(np.mean(lin), '.4e')
      + '  range ' + format(min(lin), '.4e') + ' - ' + format(max(lin), '.4e'))
print('nonlinear  mean ' + format(np.mean(non), '.4e')
      + '  range ' + format(min(non), '.4e') + ' - ' + format(max(non), '.4e'))
overlap = not (max(lin) < min(non) or max(non) < min(lin))
print('ranges overlap: ' + str(overlap)
      + ('  -> difference NOT resolvable' if overlap else '  -> SEPARATED'))
print('predict-zero baseline: ' + format(ZERO_RMSE, '.4e'))
print()
print('=' * 62)
print('COPY EVERYTHING BELOW THIS LINE BACK')
print('=' * 62)
print(json.dumps({'condition': CONDITION, 'epochs': EPOCHS,
                  'rmse_predict_zero': ZERO_RMSE, 'results': results}))
""")

    nb = {"cells": cells,
          "metadata": {"accelerator": "GPU",
                       "colab": {"provenance": [], "gpuType": "T4"},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "stage2_" + tag + "_" + condition + ".ipynb")
    with open(path, "w") as f:
        json.dump(nb, f)

    chk = json.load(open(path))
    for i, c in enumerate(chk["cells"]):
        if c["cell_type"] == "code":
            assert len(c["source"]) > 1, "cell %d collapsed" % i
            compile("".join(c["source"]), "<cell %d>" % i, "exec")
    return path, os.path.getsize(path)


if __name__ == "__main__":
    for cond, tag in CONDITIONS:
        p, sz = build(cond, tag)
        print("built " + os.path.basename(p) + "  (" + str(sz // 1024) + " KB)")
