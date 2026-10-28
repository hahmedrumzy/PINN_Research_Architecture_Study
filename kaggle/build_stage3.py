"""
Stage 3 notebook builder: complexity (n point loads) x trunk width.

20 runs = 4 conditions (n4/n8/n16/n32_random_points) x 5 trunk widths
(256/512/1024/2048/4096), all relu / max / separate backbone / dim_k=1024.

Reuses build() from build_stage.py unchanged, then patches the produced
notebook to add three things the stage-1/2 notebooks did not have:

  1. --save-every EPOCHS-1, so model_last.pth really is the FINAL epoch.
     experiment_architecture_combined.py saves on `epoch % save_every == 0`
     over epoch in range(EPOCHS), and never unconditionally at the end. With
     the default save_every=200 and EPOCHS=2000 the last save is epoch 1800,
     so every stage-1/2 result.json actually scored 200-epoch-stale weights.
     Setting save_every = EPOCHS-1 makes the final epoch index a multiple of
     it, so the last write is the true final state.

  2. An EXPLICIT, ASSERTED train/held-out split (files 1-4 train, file 5
     never seen). The trainer's own DataLoader(batch_size=2, drop_last=True,
     shuffle=False) over 5 sorted files already drops file 5 every epoch;
     this cell proves that rather than assuming it. No edit to the training
     script is needed or made.

  3. Evaluation reported separately for seen (1-4) and unseen (5).

Nothing here pushes to Kaggle.
"""
import json, os, sys, uuid

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
from build_stage import build          # reused unchanged

EPOCHS = 2001          # see kaggle/stage1_estimate.md for the justification
SEED = 0
DIM_K = 1024

WIDTHS = [
    ("w256",  "tiny_256",      [256, 128, 64, 32, 16]),
    ("w512",  "small_512",     [512, 256, 128, 64, 32]),
    ("w1024", "baseline_1024", None),
    ("w2048", "expanded_2048", None),
    ("w4096", "expanded_4096", None),
]
NS = [4, 8, 16, 32]


def slugify(title):
    return "-".join(title.lower().split())


SPLIT_CELL = '''
# ---- EXPLICIT held-out split, asserted (no edit to the training script) ----
# get_sample() does sorted(os.listdir(DATA)); the loader is
# DataLoader(batch_size=2, shuffle=False, drop_last=True). Over 5 samples that
# yields batches [0,1] and [2,3] and silently drops index 4 EVERY epoch. We do
# not rely on that silently -- we rebuild the identical loader and prove it.
sys.path.insert(0, "/kaggle/working")
import experiment_architecture_combined as X

BATCH_SIZE = 2
files = [os.path.join(DATA, f) for f in sorted(os.listdir(DATA))]
assert len(files) == 5, "expected 5 samples, got %d" % len(files)
assert all(f.endswith(".npy") for f in files), "non-.npy entry in " + DATA

TRAIN_FILES, HELDOUT_FILE = files[:4], files[4]

_ds = X.load_dataset(DATA)
_dl = torch.utils.data.DataLoader(_ds, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=0, drop_last=True)
seen = np.concatenate([b.numpy() for b in _dl], axis=0)
assert len(_ds) == 5, "dataset size changed"
assert seen.shape[0] == 4, "expected 4 seen samples, got %d" % seen.shape[0]

ref = np.stack([np.load(f).astype(np.float32) for f in TRAIN_FILES])
assert np.array_equal(seen, ref), "the 4 seen samples are NOT files 1-4"

held = np.load(HELDOUT_FILE).astype(np.float32)
assert not any(np.array_equal(held, s) for s in seen), "HELD-OUT SAMPLE LEAKED INTO TRAINING"

print("split OK")
print("  train  (seen)  :", [os.path.basename(f) for f in TRAIN_FILES])
print("  heldout(unseen):", os.path.basename(HELDOUT_FILE))
del _ds, _dl, seen, ref, held
'''

EVAL_CELL = '''
def _gt(f):
    return os.path.join(GT, os.path.basename(f).replace(".npy", "_ground_truth.npy"))

model = X.PhysicsInformedRegistration(dim_k=DIM_K, hidden_dims=X.HIDDEN_DIM_CONFIGS[HD],
                                      shared_backbone=False, activation=ACT, pool_type=POOL).cuda()
model.load_state_dict(torch.load(os.path.join(run_dir, "model_last.pth"), map_location="cuda"))
model.eval()
nparams = sum(p.numel() for p in model.parameters())

def rmse_of(f):
    a = np.load(f).astype(np.float32)
    gt = np.load(_gt(f))
    b = torch.tensor(a).unsqueeze(0).cuda()
    pred = (model(b)["deformed_source"] - b[:, 0, :, :]).detach().cpu().numpy()[0]
    return float(np.sqrt(((pred - gt) ** 2).sum(1).mean()))

def zero_of(f):
    return float(np.sqrt((np.load(_gt(f)) ** 2).sum(1).mean()))

seen_rmse   = [rmse_of(f) for f in TRAIN_FILES]
unseen_rmse = rmse_of(HELDOUT_FILE)
seen_zero   = [zero_of(f) for f in TRAIN_FILES]
unseen_zero = zero_of(HELDOUT_FILE)

res = {"condition": CONDITION, "dim_k": DIM_K, "hidden_dims": HD, "activation": ACT,
       "pool": POOL, "backbone": "separate", "seed": SEED, "epochs": EPOCHS,
       "params_M": nparams / 1e6,
       "train_files": [os.path.basename(f) for f in TRAIN_FILES],
       "heldout_file": os.path.basename(HELDOUT_FILE),
       "rmse_seen_mean": float(np.mean(seen_rmse)),
       "rmse_seen_per_sample": seen_rmse,
       "rmse_unseen": unseen_rmse,
       "generalisation_gap": unseen_rmse - float(np.mean(seen_rmse)),
       "generalisation_ratio": unseen_rmse / float(np.mean(seen_rmse)),
       "rmse_predict_zero_seen": float(np.mean(seen_zero)),
       "rmse_predict_zero_unseen": unseen_zero,
       "rmse_mean_all5": float(np.mean(seen_rmse + [unseen_rmse])),
       "final_total": float(hist["total"][-1]),
       "final_chamfer": float(hist["chamfer"][-1]),
       "final_pinns": float(hist["pinns"][-1])}
with open("/kaggle/working/result.json", "w") as fh:
    json.dump(res, fh, indent=1)

print(json.dumps(res, indent=1))
print()
print("RMSE seen  (files 1-4) : %.6e   (predict-zero %.6e)" % (res["rmse_seen_mean"], res["rmse_predict_zero_seen"]))
print("RMSE unseen(file 5)    : %.6e   (predict-zero %.6e)" % (unseen_rmse, unseen_zero))
print("generalisation ratio   : %.3f" % res["generalisation_ratio"])
print("BEATS DOING NOTHING (unseen)? ", unseen_rmse < unseen_zero)
'''


def _mkcell(s):
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
            "execution_count": None, "outputs": [],
            "source": [l + "\n" for l in s.strip("\n").split("\n")]}


def patch(path):
    nb = json.load(open(path))
    cells = nb["cells"]

    def setsrc(i, s):
        cells[i]["source"] = [l + "\n" for l in s.strip("\n").split("\n")]

    # locate cells by content, not by index
    i_run = next(i for i, c in enumerate(cells)
                 if c["cell_type"] == "code" and "--poolings-subset" in "".join(c["source"]))
    i_res = next(i for i, c in enumerate(cells)
                 if c["cell_type"] == "code" and "result.json" in "".join(c["source"]))

    # 1. force the final epoch to be checkpointed
    s = "".join(cells[i_run]["source"])
    old = '"--poolings-subset", POOL]'
    assert old in s, "run cell shape changed; refusing to patch blindly"
    setsrc(i_run, s.replace(
        old, '"--poolings-subset", POOL,\n            "--save-every", str(EPOCHS - 1)]'))

    # 3. split-aware evaluation
    setsrc(i_res, EVAL_CELL)

    # 2. asserted split, inserted immediately before the training cell
    cells.insert(i_run, _mkcell(SPLIT_CELL))

    with open(path, "w") as f:
        json.dump(nb, f, indent=1)

    # revalidate: every code cell must still compile
    chk = json.load(open(path))
    for i, c in enumerate(chk["cells"]):
        if c["cell_type"] != "code":
            continue
        assert len(c["source"]) > 1, "cell %d collapsed" % i
        compile("".join(c["source"]), "<%s cell %d>" % (os.path.basename(os.path.dirname(path)), i), "exec")


def main():
    outroot = os.path.join(SP, "notebooks")
    built = []
    for n in NS:
        for wtag, hd, dims in WIDTHS:
            title = "S3 n%d %s" % (n, wtag)
            slug = slugify(title)
            d = os.path.join(outroot, "nb_" + slug)
            path = build(slug, title, "n%d_random_points" % n, hd, "relu", "max",
                         EPOCHS, SEED, d, dim_k=DIM_K, custom_dims=dims)
            patch(path)
            meta = json.load(open(os.path.join(d, "kernel-metadata.json")))
            assert meta["id"] == "haziqahmedrumzy/" + slugify(meta["title"]), \
                "id/title mismatch: %s vs %s" % (meta["id"], meta["title"])
            built.append(slug)
            print("built", slug)
    print("\n%d notebooks built. NOTHING PUSHED." % len(built))
    print(" ".join(built))


if __name__ == "__main__":
    main()
