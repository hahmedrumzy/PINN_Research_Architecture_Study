## Repository: what's where

The repo is a fork of the paper's code, with a synthetic benchmark and experiment scripts added. Results are in `Findings/FINDINGS.md`; this section explains the files.

### Top-level files

| File | What it is |
| --- | --- |
| `experiment_architecture_combined.py` | Main training script. A one-file rewrite of the paper's model where trunk width, activation, pooling and feature dimension can each be changed. Trains, logs loss per epoch, saves weights. `onaxis` mode changes one setting at a time (6 runs); `full_grid` tries every combination (18). |
| `experiment_full_study.py` | Follow-up study built on the script above. Stage 0 demonstrates and fixes the strain bug; stage 1 tests width against deformation complexity; stage 2 compares linear and Neo-Hookean elasticity. Holds the corrected models (use V2) and the Neo-Hookean model. |
| `non_rigid_registration_modified_v10.py`, `pinns_modified_v10.py` | Edited copies of two original files, not used by any script. The second has a typo on line 11 that stops it importing. |
| `readme.txt` | The original one-paragraph project note. |
| Paper PDF | The paper being studied. |

### Folders

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

### How the three data folders fit together

The generator writes `before` and `after = before + displacement` into `toy_datasets/`, and writes the displacement into `ground_truth/`. The network trains on `toy_datasets/` without ever seeing the answer; scoring compares its prediction with `ground_truth/`.

Each training file has shape (2, 1024, 3): before and after, 1024 points (512 surface first, then 512 interior), and x, y, z. Surface points are identical in all 5 files; interior points differ by random seed.

### The loading cases

| Case | What happens to the sphere | Solved by | Non-affine |
| --- | --- | --- | --- |
| `uniform_pressure` | Pressed evenly on all sides; shrinks uniformly | Exact formula | 0.00 |
| `stiffer_material` | Same, 100× stiffer and pushed 100× harder; byte-identical data, so unused | Exact formula | 0.00 |
| `two_zone_material` | Stiff core inside a soft shell, pressed evenly | Exact formula | 0.46 |
| `two_point_poles` | Pinched at top and bottom | Finite elements | 0.77 |
| `four_points_poles_equator` | Pinched at both poles and two opposite equator points | Finite elements | 0.89 |
| `four_random_points` | Pushed at 4 random surface points | Finite elements | 0.87 |
| `n_random_points` (N = 1 to 32) | Pushed at N random points | Finite elements | 0.80–0.89 |

### Inside `kaggle/`

- `build_stage.py`, `build_stage2.py`, `build_stage3.py`: build the notebooks for the architecture study, the elasticity comparison and the complexity × width study.
- `queue_push.sh`, `queue2.sh`, `queue3.sh`: push notebooks one at a time, retrying while Kaggle's limit of 2 running jobs is full.
- `check.py`: shows every run's status, pushes anything waiting, downloads finished results.
- `plot_losses.py`, `export_chat.py`, `build_report.py`: loss plots, transcript export, and combining findings with the transcript.
- `notebooks/`: one folder per Kaggle run.
- `results/`: per run, the scores (`result.json`), loss per epoch (`history.csv`), the log and a loss plot; plus the strain-bug measurements and the elasticity results.

### Needs fixing before reuse

- `plot_losses.py`, `queue_push.sh` and `queue2.sh` read a temporary folder that no longer exists.
- `build_report.py` expects `FINDINGS.md` in the repo root rather than `Findings/`.
- Comments in `uniform_pressure.py` and `two_point_poles.py` still quote the old deformation size (4 × 10⁻⁴), though the code uses 0.05.
