# Physics-Informed Registration: Paper, Findings and My Work

Research internship at UCL under Dr Yipeng Hu, a co-author of the paper. The work is a research fork of the paper's code: a synthetic benchmark with exact answers, an architecture and elasticity study, and an audit of the physics loss.

## The problem: aligning MRI and ultrasound of the prostate

The paper tries to map a prostate MRI scan onto an ultrasound scan of the same patient, so that tumours visible on MRI can be targeted during a needle biopsy guided by ultrasound.

**Why it matters.** Prostate cancer is diagnosed by biopsy: a needle takes small tissue samples from suspicious areas.

- **MRI** shows suspicious areas clearly, but it is taken days before the procedure and can't guide a needle in real time.
- **Transrectal ultrasound (TRUS)** is used live during the biopsy to steer the needle, but tumours are hard to see on it.

So the suspicious areas are marked on the MRI and then transferred onto the live ultrasound. This is called MRI–ultrasound fusion biopsy. If the transfer is off by a few millimetres, the needle can miss a lesion that is itself only around 5–10 mm across.

**Why it's hard.** The prostate doesn't stay the same shape between the two scans. The ultrasound probe presses against it, and the patient's position and bladder filling differ. A rigid shift and rotation can't capture that squashing, so the alignment has to be non-rigid: every point gets its own displacement.

The two scans also look completely different, so matching them pixel by pixel doesn't work well. Instead, the prostate is outlined (segmented) in both scans, and each outline is turned into a cloud of points: here 1024 points per scan, 512 on the surface and 512 inside. The task becomes: move the MRI points so they line up with the ultrasound points.

**The catch.** Lining up the surfaces is easy to measure, but the inside of the prostate has few visible features. Many different interior deformations give the same surface match, and most are physically unrealistic (tissue folding, tearing or changing volume). The method therefore needs an extra rule that picks a realistic deformation. That rule is called a regulariser.

**How accuracy is measured.** Target registration error (TRE): the distance between matching landmarks (small cysts, calcifications, the tip and base of the gland) after alignment. These landmarks are not used during alignment, so they give an independent check. Typical values in the paper are 6–7 mm.

## Background: PointNet and physics-informed networks

**PointNet** (Qi et al., 2017) is a neural network designed to take a point cloud as input. A point cloud is just a list of (x, y, z) coordinates, and the order of the list means nothing: the same shape listed in a different order is the same shape. Ordinary networks would treat a reordered list as different input.

PointNet solves this in two steps:

1. **Same network on every point.** A small network is applied to each point separately, turning its 3 coordinates into a longer list of numbers called features. Here each point gets 1024 features.
2. **Max-pooling.** For each of the 1024 features, take the largest value across all points. This gives one 1024-number summary of the whole shape, the global feature. Taking a maximum ignores order, so the summary is the same however the points are listed.

One side effect matters later: for each feature, only the single point holding the maximum influences the summary. Nudge any other point slightly and that feature doesn't change.

**How the paper uses it for alignment.** Both clouds (MRI and ultrasound) go through PointNet, giving two 1024-number summaries. These are joined (2048 numbers) and attached to every MRI point's own coordinates (2048 + 3 = 2051 numbers per point). A second network, the trunk, shrinks this step by step (1024 → 512 → 256 → 128 → 64) to 3 numbers: that point's displacement. A separate branch outputs 6 numbers per point: the stress.

**Physics-informed neural networks (PINNs)** (Raissi et al., 2019) add physics to training. The network's output is a smooth function of its inputs, so its spatial derivatives can be computed exactly with automatic differentiation (autograd). Those derivatives are plugged into the governing equations, and how badly the equations fail (the residual) is added to the loss. Training then pushes the network towards outputs that obey the physics.

Here the physics is elasticity. For linear elasticity:

$$\varepsilon = \tfrac{1}{2}\left(\nabla \mathbf{u} + \nabla \mathbf{u}^{T}\right), \qquad \nabla \cdot \sigma = 0, \qquad \sigma = C : \varepsilon$$

That is: strain from the displacement gradient, forces in balance, and Hooke's law linking stress to strain. Every one of these needs the correct derivative of each point's displacement with respect to that point's own position.

## The paper

Min et al., *Biomechanics-informed Non-rigid Medical Image Registration with Elasticity Theories*, IEEE Transactions on Medical Imaging 45(6), June 2026. UCL Hawkes Institute, Shandong University and Leeds. Code is public on GitHub.

**The idea.** Use PointNet to predict how each MRI point moves, and use a PINN loss so the movement obeys elasticity. Training minimises:

- **Chamfer loss:** how far each moved MRI point is from its nearest ultrasound point, and vice versa (the alignment term).
- **Equilibrium loss:** forces should balance (∇·σ = 0).
- **Constitutive loss:** stress should match Hooke's law applied to strain.
- **Energy loss:** stored elastic energy should be small.

The prostate is modelled as two zones: a soft outer zone (peripheral zone, 5 kPa) and a stiffer inner zone (transition zone, 500 kPa), both nearly incompressible (Poisson's ratio 0.49). They test linear and nonlinear (Neo-Hookean) elasticity, and two layouts: separate backbones for displacement and stress (Cfg1) or one shared backbone (Cfg2).

**What it improves on.**

- **Classical biomechanical registration** (e.g. GMM-FEM) solves an optimisation per patient with a finite-element model. Accurate, but slow at the moment of use.
- **Networks trained on FEM simulations** are fast, but need many simulations built in advance, which is laborious.
- **Smoothness regularisers** (e.g. CPD, bending energy) are simple but treat all tissue the same, so they can't capture a soft zone and a stiff zone moving differently.
- **WarpPINNs**, the closest PINN work, works on images rather than points and predicts displacement only, not stress.

The pitch: physics built into a fast network, with no FEM needed during training or use.

**What it claims.**

| Experiment | Data | Main claimed result |
| --- | --- | --- |
| Patient-specific registration | 77 real MRI–ultrasound pairs | Physics loss lowers TRE significantly (p < 0.001); for the 40% largest deformations, 7.86 → about 6.3 mm |
| Trained on many patients, tested on new ones | 75 train, 33 test pairs | TRE 6.96 → 6.12 mm (p = 0.018); physics loss on new patients falls from 20 to 10⁻¹⁴, taken as "learned the physics" |
| Simulated, known answer | 22 FEM-deformed prostate meshes | Error 1.80 → 1.25 mm with physics loss; linear vs nonlinear elasticity not significantly different |
| Estimating tissue stiffness (inverse problem) | Same simulations, 6 successful cases | Stiffness ratio error 14% with nonlinear vs 76% with linear (p = 0.002) |
| Ablation | Simulations | Using all physics terms beats using one alone (e.g. 1.27 vs 1.42–1.52 mm) |

## What I found: the strain is computed wrongly

The published code does not compute strain. Its "strain" is exactly zero at 71% of points and has no relationship to the real strain (correlation −0.015).

**The bug.** Strain at point *i* needs ∂u_i/∂x_i: how that point's displacement changes with its own position. The code uses `torch.autograd.grad(Ux, x, grad_outputs=ones)`, which instead returns, for each point *j*:

$$\sum_i \frac{\partial u_i}{\partial x_j}$$

That is how much the *total* displacement of the cloud changes when point *j* moves. It equals the wanted value only if each point's output depends on its own coordinates alone.

**Why max-pooling makes it zero.** In this code, the trunk receives each point's coordinates without a gradient path back to the input. So the only way a point affects any output is through the max-pooled global feature. Each of the 1024 features has a single winning point. A point that wins no feature has no effect on any output, so its derivative is exactly zero.

296 of 1024 points win at least one feature, and they are exactly the points with nonzero "strain" (checked as set equality). The other 728 (71%) read zero. The 296 nonzero values measure how a point shifts the shape summary, not its strain. The output is a mask of max-pool winners, not a strain field.

**Measured on an untrained network:**

| Quantity | Value |
| --- | --- |
| Real per-point derivative (mean) | 5.19 × 10⁻⁷ |
| What the code returns (mean) | 2.73 × 10⁻⁴, 525× too large |
| Correlation with real displacement gradient | −0.015 |
| Points with exactly zero strain | 71.1% |

**Consequence: the physics loss constrains nothing.** With strain mostly zero, the constitutive loss just drives predicted stress to zero. Zero stress then satisfies equilibrium and gives zero energy. The whole physics loss can reach about zero without the displacement obeying any physics. In my runs it reached about zero every time, and exactly 0.0 in 3 of 10.

**The fix.** Give the trunk coordinates that carry a gradient, and hold the global shape summary fixed while differentiating. Physically this is the right definition: strain is the derivative of the displacement field for a fixed body, not the effect of changing the body's shape. My findings file calls it an approximation instead, because it drops a real gradient path through the global feature; both views are defensible, so state which one you take. Tested on uniform pressure, where the exact answer is −0.05 normal strain and zero shear everywhere, using the same trained weights for both versions:

| Version | Normal strain | Error | Points exactly zero |
| --- | --- | --- | --- |
| Published code | −0.000329 | 99.3% | 73.9% |
| Corrected | −0.049505 | 1.0% | 0% |
| Exact | −0.05 | — | — |

Correlation with the real gradient rose from −0.015 to +0.891, stable across three seeds.

**But the corrected model doesn't train.** With the fix, it did no better than predicting no movement at all (1.0× vs 122× for the published model). The physics-loss gradient reaching the backbone fell from 320 to 1.7. So the wrong gradient was doing useful work in optimisation even though it isn't strain. Why is not yet explained (see the last section).

## What I built and tested

No change I tested beat the published setup, except that halving the feature dimension matched it while training about a third faster. All runs used the paper's separate-backbone layout (Cfg1) with linear elasticity unless stated.

**1. A test set with known answers.** Seven loading cases on a sphere of 1024 points (512 surface, 512 interior), five variants each, all scaled to 5% peak displacement. Answers come from the analytical (Lamé) solution and from my own finite-element code (scikit-fem). The finite-element result matched the analytical sphere to within 0.7%, improved steadily as the mesh was refined, and satisfied the energy balance to 9 digits. The network trains only on the before and after point clouds (Chamfer loss ×10 plus the physics loss); the exact answers are used only for scoring afterwards.

Two side findings: the number of point loads doesn't control how complex the deformation is (many random loads average out towards uniform pressure), and the "stiffer material" case was byte-identical to uniform pressure, so it was dropped.

**2. The noise floor.** Same setup, only the random seed changed. Results varied by up to 6× on some cases. On `two_point_poles`, used for the architecture tests, the variation was 4–7%, so any smaller difference there is not real.

**3. Architecture sweep** (`two_point_poles`, error on the held-out sample, lower is better; predicting no movement scores 8.33 × 10⁻³):

| Change | Parameters | Error (×10⁻³) | Verdict |
| --- | --- | --- | --- |
| Published (trunk 1024, feature dim 1024, ReLU, max-pool) | 25.2M | 1.644 | Best |
| Feature dim 512 | 17.7M | 1.671 | Same, within noise |
| Attention pooling | 25.2M | 1.694 | Same, within noise |
| Trunk 2048 | 54.5M | 1.708 | Same, within noise |
| Feature dim 2048 | 40.1M | 1.709 | Same, within noise |
| Trunk 512 | 14.2M | 1.808 | Worse |
| Trunk 4096 | 142.5M | 1.845 | Worse (overfits) |
| Trunk 256 | 9.6M | 1.906 | Worse (underfits) |
| Sine activation | 25.2M | 3.592 | Much worse |
| Tanh activation | 25.2M | 3.942 | Much worse |

Only the activation function clearly matters (ReLU beats tanh by 140%). Trunk width shows a U-shape centred on the published value. Feature dimension and pooling make no measurable difference.

**4. Smaller, faster model.** On a Kaggle T4 GPU over 2000 epochs, halving the feature dimension (1024 → 512) cut parameters by 30% and training from about 42 to 28 minutes, with error change of +1.6%, below the noise floor. The 42-minute baseline is from the attention-pooling run (same size), because the original log wasn't saved.

**5. Linear vs nonlinear elasticity.** 24 runs: 4 cases × 2 theories × 3 seeds, 1000 epochs, on a Colab A100 (4.5 hours). No significant difference anywhere, and differences were smaller than seed noise. This was predicted beforehand: at 5% strain the two theories differ by about 5% in stress. It agrees with the paper's own forward-problem result.

| Case | Non-affine | Linear error (×10⁻³) | Neo-Hookean error (×10⁻³) | Difference | p |
| --- | --- | --- | --- | --- | --- |
| uniform_pressure | 0.00 | 0.663 | 0.583 | −12.1% | 1.00 |
| two_zone_material | 0.46 | 1.379 | 1.367 | −0.9% | 1.00 |
| two_point_poles | 0.77 | 1.715 | 1.680 | −2.0% | 0.50 |
| four_points_poles_equator | 0.89 | 2.608 | 2.659 | +1.9% | 0.10 |

With 3 seeds per group the smallest possible p is 0.10, so no difference could reach significance. Every case beat predicting no movement by 3–77×, so this is a genuine null rather than a broken pipeline.

**6. Other code defects.**

- Checkpoints saved only every 200 epochs with no final save, so results came from epoch 1800 of 2000. Fixing this improved one run by about 10%.
- With batch size 2 and 5 files, the 5th file was silently never trained on. I used it as the held-out test, but the original error metric mixed training and test data.
- Loss logged to 6 decimal places when it plateaus near 2 × 10⁻⁵, so it printed as 0.000000.
- The device check was inverted: with no GPU it chose CUDA instead of CPU.
- The stiffness map in the training code splits the sphere along the z-axis, but my two-zone test case is zoned by radius. This doesn't matter at the default settings (both regions soft), but would make the physics loss disagree with the data if the stiff region were switched on.

## What this means for the paper's claims

The paper's improvements may be real, but my findings suggest they don't come from enforcing elasticity.

| Paper's claim | How my findings bear on it |
| --- | --- |
| Physics loss on new patients falls to 10⁻¹⁴, so the network "learned the physics" | Explained by the bug: near-zero strain lets stress go to zero, and every physics term then vanishes for free |
| Adding the physics loss lowers TRE on real patients | Not tested on their data. If the physics is inert, the gain likely comes from the extra gradient acting as a generic regulariser |
| Using all physics terms beats one alone (ablation) | Doesn't show the physics works; all terms depend on the same broken derivative |
| Linear and nonlinear elasticity perform the same | Agrees with my result, and is expected at these small strains |
| Nonlinear elasticity estimates stiffness far better (14% vs 76%) | Not tested. Stiffness is estimated through the constitutive loss, which uses the broken strain, so this is the claim most directly at risk |

The bug is architectural, not a typo: any PointNet-style PINN that differentiates with `grad_outputs=ones` through a shared global feature has the same problem.

## Repository: what's where

The repo is a fork of the paper's code, with a synthetic benchmark and experiment scripts added. Results are in `Findings/FINDINGS.md`; this section explains the files.

**Top-level files**

| File | What it is |
| --- | --- |
| `experiment_architecture_combined.py` | Main training script. A one-file rewrite of the paper's model where trunk width, activation, pooling and feature dimension can each be changed. Trains, logs loss per epoch, saves weights. `onaxis` mode changes one setting at a time (6 runs); `full_grid` tries every combination (18). |
| `experiment_full_study.py` | Follow-up study built on the script above. Stage 0 demonstrates and fixes the strain bug; stage 1 tests width against deformation complexity; stage 2 compares linear and Neo-Hookean elasticity. Holds the corrected models (use V2) and the Neo-Hookean model. |
| `non_rigid_registration_modified_v10.py`, `pinns_modified_v10.py` | Edited copies of two original files, not used by any script. The second has a typo on line 11 that stops it importing. |
| `readme.txt` | The original one-paragraph project note. |
| Paper PDF | The paper being studied. |

**Folders**

| Folder | Contents |
| --- | --- |
| `Findings/` | `FINDINGS.md` (the results, about 250 lines) and `REPORT.md` (the findings plus the full working transcript, about 7,000 lines). |
| `original_research_code/` | The paper's three code files, kept for reference: the no-physics baseline, the physics loss library, and the full PINN model, where the strain bug lives. |
| `generators/` | Scripts that build the synthetic spheres. `_common.py` (sphere sampling, materials, 5% scaling, saving) and `_fem.py` (finite-element solver) are shared; there is one script per loading case. |
| `toy_datasets/` | The before and after point clouds the network trains on. One folder per loading case, 5 files each. |
| `ground_truth/` | The exact displacement of every point, used only for scoring. |
| `kaggle/` | Scripts to build, queue and check Kaggle runs, plot losses and export the transcript, plus every result that came back. |
| `colab/` | Self-contained Colab notebooks for the linear vs Neo-Hookean comparison; data and code are built in. |
| `ToyDataset(FEA Sphere)/`, `archive/` | The first single-sphere generator and the first five spheres. No longer used. |

Not in the repo: trained weights (about 3.4 GB), local test runs and the Python environment.

**How the three data folders fit together.** The generator writes `before` and `after = before + displacement` into `toy_datasets/`, and writes the displacement into `ground_truth/`. The network trains on `toy_datasets/` without ever seeing the answer; scoring compares its prediction with `ground_truth/`. Each training file has shape (2, 1024, 3): before and after, 1024 points (512 surface first, then 512 interior), and x, y, z. Surface points are identical in all 5 files; interior points differ by random seed.

**The loading cases**

| Case | What happens to the sphere | Solved by | Non-affine |
| --- | --- | --- | --- |
| `uniform_pressure` | Pressed evenly on all sides; shrinks uniformly | Exact formula | 0.00 |
| `stiffer_material` | Same, 100× stiffer and pushed 100× harder; byte-identical data, so unused | Exact formula | 0.00 |
| `two_zone_material` | Stiff core inside a soft shell, pressed evenly | Exact formula | 0.46 |
| `two_point_poles` | Pinched at top and bottom | Finite elements | 0.77 |
| `four_points_poles_equator` | Pinched at both poles and two opposite equator points | Finite elements | 0.89 |
| `four_random_points` | Pushed at 4 random surface points | Finite elements | 0.87 |
| `n_random_points` (N = 1 to 32) | Pushed at N random points | Finite elements | 0.80–0.89 |

**Inside `kaggle/`**

- `build_stage.py`, `build_stage2.py`, `build_stage3.py`: build the notebooks for the architecture study, the elasticity comparison and the complexity × width study.
- `queue_push.sh`, `queue2.sh`, `queue3.sh`: push notebooks one at a time, retrying while Kaggle's limit of 2 running jobs is full.
- `check.py`: shows every run's status, pushes anything waiting, downloads finished results.
- `plot_losses.py`, `export_chat.py`, `build_report.py`: loss plots, transcript export, and combining findings with the transcript.
- `notebooks/`: one folder per Kaggle run. `results/`: per run, the scores (`result.json`), loss per epoch (`history.csv`), the log and a loss plot; plus the strain-bug measurements and the elasticity results.

**Needs fixing before reuse:** `plot_losses.py`, `queue_push.sh` and `queue2.sh` read a temporary folder that no longer exists; `build_report.py` expects `FINDINGS.md` in the repo root rather than `Findings/`; comments in `uniform_pressure.py` and `two_point_poles.py` still quote the old deformation size (4 × 10⁻⁴), though the code uses 0.05.

## Not yet established, and next steps

The bug and its effect on the strain are solid; what it means for training and for real patients is still open.

**Not established:**

- **Everything is synthetic.** Nothing has been run on real MRI–ultrasound data.
- **The architecture sweep and the smaller-model result rest on one seed each.**
- **Why the corrected model fails to train is unexplained.** Two simpler causes need ruling out first: the fix may also be cutting the data loss off from the backbone, or the now-correct physics loss may be pulling predictions towards zero movement at the current loss weighting. The failing run also used a new training loop, which is a third suspect.
- **The training data is weak.** Four near-duplicate samples, and the held-out file is the same deformation resampled.
- **The 1000-epoch elasticity comparison wasn't trained to convergence**, though both theories had the same budget.
- **The complexity × width study is not finished.** Its 20 notebooks (4, 8, 16 and 32 push points × trunk width) are built, but results aren't in yet.

**Next steps, in order:**

- [ ] Check whether the data loss still reaches the backbone in the corrected model.
- [ ] Train the corrected model with physics weight 0, then sweep the weight upwards.
- [ ] Re-run feature dim 512 with 2–3 more seeds (about 30 minutes each on a T4).
- [ ] Collect the complexity × width results.
- [ ] Test the inverse problem (stiffness estimation) with the published and corrected strain.
- [ ] Discuss with Dr Hu whether the corrected strain can be tested on the clinical MRI–ultrasound data.
