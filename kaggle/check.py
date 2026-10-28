"""
Standalone status / push / collect tool. Run it any time:

    py -3.13 kaggle/check.py

It (1) shows the status of every run, (2) pushes any notebook that has not been
submitted yet if a GPU slot is free, (3) downloads results for finished runs,
and (4) prints the comparison tables. Safe to run repeatedly.
"""
import json
import glob
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(REPO, "kaggle", "notebooks")
RES = os.path.join(REPO, "kaggle", "results")
USER = "haziqahmedrumzy"

# prior architecture sweep -- all on two_point_poles
RUNS_ARCH = ["s1-w1024-relu-max", "s1-w2048-relu-max", "s1-w4096-relu-max",
             "s1-w1024-tanh-max", "s1-w1024-sin-max", "s1-w1024-relu-attn",
             "s2-w256", "s2-w512", "s2-dimk512", "s2-dimk2048"]

# stage 1 -- trunk width x boundary condition
CONDS = {"up": "uniform_pressure", "tz": "two_zone_material",
         "tp": "two_point_poles", "fp": "four_points_poles_equator"}
WIDTHS = [256, 512, 1024, 2048, 4096]
RUNS_S1 = ["s1-" + c + "-w" + str(w) for c in CONDS for w in WIDTHS]

RUNS = RUNS_ARCH + RUNS_S1


def kaggle(*args):
    return subprocess.run([sys.executable, "-m", "kaggle", *args],
                          capture_output=True, text=True).stdout


def status(slug):
    out = kaggle("kernels", "status", USER + "/" + slug)
    for s in ("COMPLETE", "RUNNING", "QUEUED", "ERROR", "CANCEL"):
        if s in out:
            return s
    return "NOT_PUSHED"


def nb_dir(slug):
    for d in glob.glob(os.path.join(NB, "*")):
        meta = os.path.join(d, "kernel-metadata.json")
        if os.path.exists(meta) and json.load(open(meta))["id"].endswith("/" + slug):
            return d
    return None


def load(slugs):
    out = []
    for slug in slugs:
        f = os.path.join(RES, slug, "result.json")
        if os.path.exists(f):
            d = json.load(open(f))
            d["_slug"] = slug
            out.append(d)
    return out


print("=" * 78)
states = {}
for slug in RUNS:
    states[slug] = status(slug)

busy = sum(1 for s in states.values() if s in ("RUNNING", "QUEUED"))
done = sum(1 for s in states.values() if s == "COMPLETE")
print("complete " + str(done) + " | running/queued " + str(busy)
      + " | GPU slots in use " + str(busy) + "/2")

for slug, st in states.items():
    if st == "NOT_PUSHED" and busy < 2:
        d = nb_dir(slug)
        if d:
            out = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", d],
                                 capture_output=True, text=True).stdout
            if "successfully pushed" in out.lower():
                print("  PUSHED " + slug)
                busy += 1

for slug, st in states.items():
    if st == "COMPLETE" and not os.path.exists(os.path.join(RES, slug, "result.json")):
        os.makedirs(os.path.join(RES, slug), exist_ok=True)
        kaggle("kernels", "output", USER + "/" + slug, "-p", os.path.join(RES, slug))
        print("  downloaded " + slug)

# ---------------- stage 1 table ----------------
s1 = load(RUNS_S1)
if s1:
    print()
    print("=" * 78)
    print("STAGE 1 - trunk width x condition, scored on held-out file 5")
    print("condition".ljust(28) + "width".rjust(7) + "par(M)".rjust(9)
          + "SEEN".rjust(11) + "UNSEEN".rjust(11) + "gap".rjust(7))
    print("-" * 78)
    for d in sorted(s1, key=lambda r: (r["condition"], r["params_M"])):
        w = d["hidden_dims"].split("_")[1]
        print(d["condition"].ljust(28) + w.rjust(7)
              + format(d["params_M"], "9.1f")
              + format(d.get("rmse_seen", float("nan")), "11.3e")
              + format(d.get("rmse_unseen", float("nan")), "11.3e")
              + format(100 * d.get("overfit_gap", float("nan")), "6.0f") + "%")
    print("-" * 78)
    for c in sorted({d["condition"] for d in s1}):
        sub = [d for d in s1 if d["condition"] == c]
        best = min(sub, key=lambda r: r["rmse_unseen"])
        print("  " + c.ljust(28) + "best so far: width "
              + best["hidden_dims"].split("_")[1]
              + "   (" + str(len(sub)) + "/5 widths done)")
    print()
    print("Stage 1: " + str(len(s1)) + "/20 collected")

# ---------------- prior architecture sweep ----------------
arch = load(RUNS_ARCH)
if arch:
    z = arch[0]["rmse_predict_zero"]
    print()
    print("=" * 70)
    print("PRIOR ARCHITECTURE SWEEP (two_point_poles)")
    print("trunk".ljust(8) + "dim_k".rjust(6) + "act".rjust(6) + "pool".rjust(11)
          + "par(M)".rjust(8) + "SEEN".rjust(11) + "UNSEEN".rjust(11))
    print("-" * 70)
    for d in sorted(arch, key=lambda r: r["rmse_per_sample"][4]):
        seen = sum(d["rmse_per_sample"][:4]) / 4
        w = (d["hidden_dims"].replace("expanded_", "").replace("baseline_", "")
             .replace("tiny_", "").replace("small_", ""))
        print(w.ljust(8) + str(d.get("dim_k", 1024)).rjust(6)
              + d["activation"].rjust(6) + d["pool"].rjust(11)
              + format(d["params_M"], "8.1f")
              + format(seen, "11.3e") + format(d["rmse_per_sample"][4], "11.3e"))
    print("-" * 70)
    print("predict zero (do nothing)".ljust(39) + format(z, "11.3e") + format(z, "11.3e"))
