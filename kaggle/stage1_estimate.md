# Stage 3 estimate — complexity (n point loads) x trunk width

20 runs = 4 conditions (`n4/n8/n16/n32_random_points`) x 5 trunk widths
(256, 512, 1024, 2048, 4096). All runs: `relu` / `max` pooling / separate
backbone / `dim_k=1024` / `lr=1e-3` / `batch_size=2` / seed 0.

Written by Subagent B. **Nothing has been pushed to Kaggle.**

---

## 1. Datasets

Generated locally with the existing `generators/n_random_points.py`:

```
python generators/n_random_points.py --n 4
python generators/n_random_points.py --n 16
python generators/n_random_points.py --n 32
```

`n8_random_points` was left untouched. All four verified: 5 files each, shape
`(2, 1024, 3)`, `float32`, finite, `max|u| = 0.050000` exactly, ground truth
matches `target - source` to < 1e-6, and each `toy_datasets/<cond>/` holds only
`.npy` files (`assert_only_npy` passes — the trainer's flat `os.listdir` +
`np.load` will not crash).

### Non-affine fraction

Best affine `source -> target` by least squares; `max residual / max|u|`.

The reference numbers quoted in the task are **sample-1 values**, not means —
my implementation reproduces them exactly, which validates the metric:

| condition | sample 1 (ref) | reference | mean of 5 |
|---|---|---|---|
| uniform_pressure | 0.000 | 0.00 | 0.000 |
| two_zone_material | 0.469 | 0.47 | 0.464 |
| two_point_poles | 0.748 | 0.75 | 0.769 |
| four_random_points | 0.804 | 0.80 | 0.866 |
| n8_random_points | 0.820 | 0.82 | 0.887 |
| four_points_poles_equator | 0.853 | 0.85 | 0.885 |

New conditions:

| condition | sample 1 | mean of 5 | per-sample | mean\|u\| |
|---|---|---|---|---|
| n4_random_points | 0.804 | **0.866** | 0.804 0.886 0.832 0.883 0.923 | 0.00708 |
| n8_random_points | 0.820 | **0.887** | 0.820 0.877 0.908 0.954 0.876 | 0.00909 |
| n16_random_points | 0.781 | **0.863** | 0.781 0.952 0.744 0.845 0.993 | 0.01228 |
| n32_random_points | 0.646 | **0.803** | 0.646 0.928 0.791 0.880 0.771 | 0.01292 |

**Complexity SATURATES at n=8 and then DECLINES.** It does not rise with n.
Mean non-affine fraction: 0.866 -> 0.887 -> 0.863 -> 0.803 for n = 4, 8, 16, 32.
The n=4 -> n=8 rise (0.021) is smaller than the per-sample spread (~0.05-0.10),
so it is not resolvable with 5 samples; the n=32 drop is larger and looks real.

The mechanism is physical and predictable in hindsight: more randomly-placed
inward point loads on a sphere average towards isotropic inward pressure, and
`uniform_pressure` scores 0.00. `mean|u|` nearly doubles from n=4 to n=32 while
`max|u|` is pinned at 0.05 — the field is getting flatter and more global, i.e.
*more* affine, not less. The inertia-relief unbalanced fraction also falls with
n (n=4 sample 1: 5.3e-2; n=32 sample 1: 6.0e-3), same story.

**Consequence for the experiment: `n` is not a monotone complexity axis.** The
four conditions span 0.80-0.89 mean non-affine fraction — a range of 0.086,
which is comparable to the within-condition spread. Sweeping n gives four
datasets of *essentially the same* non-rigidity. See "Flagged" below.

**Also note:** `n4_random_points` is **bit-identical** to the existing
`four_random_points` (verified with `np.array_equal` on all 5 files). Same
generator, same seeds, same `load_points(seed, n=4)`. This is expected and was
called out in the task, but it means n=4 contributes no new data — only a new
condition name — and the existing `four_random_points` results are directly
reusable for it.

---

## 2. Explicit held-out split

The prior sweep held out sample 5 by accident. In
`experiment_architecture_combined.py`:

- `get_sample()` -> `sorted(os.listdir(data_dir))` — deterministic order.
- `DataLoader(trainset, batch_size=args.batch_size, shuffle=False,
  num_workers=0, drop_last=True)` (line 555-556).

With 5 samples and `batch_size=2` that yields batches `[0,1]` and `[2,3]`, and
index 4 — the 5th file in sorted order — is dropped **every epoch,
deterministically**, because `shuffle=False`. The behaviour is already the one
we want; the problem is that it is implicit.

**Design: assert it, do not assume it.** A cell inserted immediately before
training rebuilds the *identical* loader and proves the split, without touching
the training script:

```python
BATCH_SIZE = 2
files = [os.path.join(DATA, f) for f in sorted(os.listdir(DATA))]
assert len(files) == 5
assert all(f.endswith(".npy") for f in files)
TRAIN_FILES, HELDOUT_FILE = files[:4], files[4]

_ds = X.load_dataset(DATA)
_dl = torch.utils.data.DataLoader(_ds, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=0, drop_last=True)
seen = np.concatenate([b.numpy() for b in _dl], axis=0)
assert len(_ds) == 5
assert seen.shape[0] == 4                      # drop_last really dropped one
ref = np.stack([np.load(f).astype(np.float32) for f in TRAIN_FILES])
assert np.array_equal(seen, ref)               # and it dropped the RIGHT one
held = np.load(HELDOUT_FILE).astype(np.float32)
assert not any(np.array_equal(held, s) for s in seen)   # no leak
```

This fails loudly if anyone later changes `batch_size`, adds a 6th sample,
turns on `shuffle`, or renames files so sorted order changes. Dry-run on CPU
against all four new conditions: **all assertions pass.**

Evaluation then reports both numbers separately —
`rmse_seen_mean` (files 1-4), `rmse_unseen` (file 5), plus
`generalisation_gap`, `generalisation_ratio`, and predict-zero baselines for
each — instead of the single all-5 mean the prior sweep reported.

**The accidental hold-out was real and the gap is measurable.** Recomputing
from the 10 completed stage-1/2 `result.json` files:

| run | mean RMSE files 1-4 | RMSE file 5 | ratio |
|---|---|---|---|
| s1-w1024-relu-attn | 0.001292 | 0.001694 | 1.311 |
| s1-w1024-relu-max | 0.001377 | 0.001644 | 1.193 |
| s1-w2048-relu-max | 0.001278 | 0.001708 | 1.337 |
| s1-w4096-relu-max | 0.001386 | 0.001845 | 1.332 |
| s2-dimk512 | 0.001310 | 0.001671 | 1.275 |
| s2-dimk2048 | 0.001529 | 0.001709 | 1.118 |
| s2-w256 | 0.001795 | 0.001906 | 1.062 |
| s2-w512 | 0.001771 | 0.001808 | 1.021 |
| s1-w1024-sin-max | 0.003953 | 0.003592 | 0.909 |
| s1-w1024-tanh-max | 0.004328 | 0.003942 | 0.911 |

Unseen is worse in 8 of 10 runs, mean ratio 1.147. The two exceptions are the
`sin` and `tanh` runs, whose training error is ~3x higher and still dominates.
The prior sweep's headline "rmse_mean" therefore mixed train and test error;
stage 3 will not.

---

## 3. Epoch count: **2001**

Not 2000-by-assumption. The reasoning, and a correction to the premise:

### The premise "plateaued at epoch 300-500" is only true for wide trunks

From the four surviving `history.csv` files (100-epoch moving average of total
loss, which is needed because history.csv rounds to 1e-6 and the plateau sits
at ~2e-5, i.e. only ~20 quantisation steps):

| run | MA100 @500 | @1000 | @1500 | @2000 | still falling at 2000? |
|---|---|---|---|---|---|
| s1-w1024-relu-attn | 2.14e-5 | 1.94e-5 | 1.86e-5 | 1.80e-5 | essentially flat |
| s2-dimk512 | 2.23e-5 | 1.95e-5 | 1.91e-5 | 1.83e-5 | essentially flat |
| s2-dimk2048 | 3.42e-5 | 2.44e-5 | 2.03e-5 | 2.13e-5 | flat after ~1200 |
| s2-w512 | 7.83e-5 | 4.02e-5 | 3.34e-5 | 2.66e-5 | **NO — still dropping 20% over the last 500** |

- Wide trunk (1024) reaches within 1.2x of its final loss by **epoch ~475** and
  within 1.1x by ~830. For those configs 2000 epochs is largely waste.
- **Trunk 512 has not converged at 2000.** 70% of its total drop from epoch 500
  onward happened *after* epoch 500, and its last-500-epoch moving average
  falls a further 20%. It reaches within 1.2x of its own best only at epoch 1576.
- Trunk 256 has no surviving history, but it finished with the worst loss
  (3.0e-5 vs 1.8e-5) and worst RMSE (0.00182) of the whole sweep, so it is
  almost certainly slower still.

The "flat over the final 500" claim comes from the wide-trunk curves. It does
not generalise to the narrow ones, and the narrow ones are exactly what this
sweep varies.

### Why not cut to 500

Stage 3's independent variable *is* trunk width. Cutting the budget to where
the wide trunks plateau would leave w256/w512 systematically undertrained and
manufacture a "wider is better" result that is really "narrower needed more
epochs" — a width x epochs confound on the one axis being measured.

### Why truncation is safe at all

`experiment_architecture_combined.py` uses **plain `Adam(lr=1e-3)` with no LR
scheduler** (line 584; the only LR adjustment is the 0.1x for `sin`, unused
here). Epoch *k* of a 1000-epoch run is therefore bit-identical to epoch *k* of
a 2000-epoch run at the same seed. The epoch count is a pure stopping choice,
not a change of trajectory — so results stay directly comparable to the 10
completed stage-1/2 runs, which were all 2000.

### Decision

**2001 epochs**, uniform across all 20 runs.

- 2000 is the smallest budget under which the *slowest* configs in the sweep
  (w256, w512) get as close to convergence as the evidence shows is reachable;
  it is not enough for them, but going higher (3000 -> ~26 GPU-h) would consume
  the whole weekly quota with no headroom for reruns.
- Uniform across widths, so no per-config budget confound.
- Matches the 10 existing runs exactly, so stage 1/2 remain a valid reference.
- Honest reporting requirement: **state in the write-up that w256/w512 are still
  descending at the budget** rather than describing 2000 as "converged". The
  saved `history.csv` supports checking this per run.
- Waste is real but accepted: for w1024/w2048/w4096 roughly 60% of the epochs
  buy <10% loss reduction. Buying that back would cost the comparability above.

### Why 2001 and not 2000 — a bug in the prior sweep

`experiment_architecture_combined.py` line 604:

```python
if epoch % args.save_every == 0:
    torch.save(model.state_dict(), os.path.join(run_dir, "model_last.pth"))
```

`epoch` runs `range(args.epochs)` and there is **no unconditional save after
the loop**. With the default `save_every=200` and `epochs=2000`, the last write
happens at `epoch=1800`; epoch 1999 is never saved. So `model_last.pth` in every
stage-1/2 run is the **epoch-1800** state, while the `final_total` reported
alongside it came from `history[-1]` = epoch 2000. **Every RMSE in the prior
sweep was measured on weights 200 epochs stale, and was reported next to a loss
from a different point in training.**

Fix, without editing the training script: the stage-3 notebooks pass
`--save-every (EPOCHS - 1)` = 2000, so the final epoch index 2000 satisfies
`2000 % 2000 == 0` and the true final state is written. `epochs=2001` makes that
land on a round 2000 training epochs. Cost: one extra epoch (~1-2 s).

---

## 4. GPU-hour and wall-clock estimate

Measured T4 per-epoch times from the prior sweep: trunk 256/512/1024
~1.15-1.36 s/epoch, trunk 2048 ~1.36 s/epoch, trunk 4096 ~2.39 s/epoch
(inclusive of startup). Central estimate uses 1.25 s for 256/512/1024;
conservative uses the 1.36 s top of the range.

| width | runs | s/epoch (central) | h/run | h total (central) | h total (conservative) |
|---|---|---|---|---|---|
| w256 | 4 | 1.25 | 0.695 | 2.78 | 3.02 |
| w512 | 4 | 1.25 | 0.695 | 2.78 | 3.02 |
| w1024 | 4 | 1.25 | 0.695 | 2.78 | 3.02 |
| w2048 | 4 | 1.36 | 0.756 | 3.02 | 3.02 |
| w4096 | 4 | 2.39 | 1.328 | 5.31 | 5.31 |
| **total** | **20** | | | **16.68** | **17.41** |

Plus ~1.5 min/session for Kaggle startup, dataset copy and the plotting/eval
cells: +0.50 h.

### **Total: ~17.2-17.9 GPU-hours — call it 18 GPU-hours.**

### **Expected wall-clock: ~9 hours** (8.8-9.2 h)

- Two concurrent GPU sessions -> ideal makespan = total / 2 = 8.6-9.0 h.
- Packing is near-perfect: the longest single run (w4096, 1.33 h) is far
  shorter than half the total, and `queue3.sh` orders longest-first.
- `queue3.sh` polls every 3 min against the session cap, so each of the ~10
  handoffs per lane adds ~1.5 min on average: +0.25 h.

### Quota

18 of ~30 GPU-h/week. Fits in a single week with ~12 h headroom — enough to
rerun the entire w4096 column (5.3 h) plus a few others if something fails.

Assumption to confirm: Kaggle's quota meter is being read as session wall-clock
hours on a GPU machine. If a `T4 x2` session is metered at 2x, the true cost is
~36 h and **will not fit in one week** — it would need splitting across two
weeks or dropping to a single T4.

---

## 5. Flagged for human decision

1. **`n` is not a working complexity axis.** Non-affine fraction saturates at
   n=8 and falls by n=32. The four conditions span 0.80-0.89, barely wider than
   the within-condition sample spread. As designed, this sweep varies trunk
   width across four datasets of near-identical non-rigidity, and will probably
   not show a complexity x capacity interaction because there is little
   complexity variation to interact with. Options: (a) run it anyway as a
   width sweep with a replication check across four related datasets; (b) pick
   the complexity axis from the *measured* fractions instead, e.g.
   `uniform_pressure` (0.00) -> `two_zone_material` (0.46) -> `two_point_poles`
   (0.77) -> `n8_random_points` (0.89), which spans the full range; (c) change
   what n varies — e.g. concentrate loads in a shrinking patch, or scale the
   load *count* while keeping total load fixed, so the field stays local.
   I have not acted on this; the datasets as specified are built and verified.

2. **n4 is a bit-identical duplicate of `four_random_points`.** Expected, but it
   means one of the 20 runs (per width) is a re-run of an existing condition.
   For w1024/w2048/w4096 the stage-1 results already cover it, so up to 3 of the
   20 runs (~2.7 GPU-h) could be skipped if the epoch/eval changes are accepted
   as non-blocking — though they are not strictly comparable, since stage-1
   scored epoch-1800 weights on all 5 samples (see the bug in section 3).

3. **BLOCKER — the Kaggle dataset must be updated before any push.**
   `haziqahmedrumzy/pinn-sphere-conditions` currently contains 7 conditions
   (`four_points_poles_equator`, `four_random_points`, `n8_random_points`,
   `stiffer_material`, `two_point_poles`, `two_zone_material`,
   `uniform_pressure`). It has **no `n4`, `n16` or `n32`**. All 15 n4/n16/n32
   notebooks would fail at the `DATA` path. A new dataset version must be
   uploaded first. That is itself a push, so it needs the same approval.

4. **The stale-checkpoint bug (section 3) invalidates the reported RMSEs of the
   10 completed stage-1/2 runs**, which were measured on epoch-1800 weights.
   Stage 3 fixes it going forward, but the stage-1/2 numbers should be treated
   as approximate, or those runs re-scored if their checkpoints survive.

---

## 6. Files

- `kaggle/build_stage3.py` — builder; reuses `build()` from `build_stage.py`
  unchanged, then patches in `--save-every`, the asserted split cell, and the
  seen/unseen evaluation. Re-compiles every code cell and asserts
  `id == "haziqahmedrumzy/" + slugify(title)` for each notebook.
- `kaggle/notebooks/nb_s3-*` — 20 notebook directories, built and validated.
- `kaggle/queue3.sh` — push queue, longest-first, 3-minute retry against the
  2-session cap. **Dry-runs by default; requires the literal argument `GO`.**

**Nothing has been pushed.**
