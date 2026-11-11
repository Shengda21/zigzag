"""Make a compact evidence figure from the derived, post-hoc E1-E3 analyses."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
arms = json.loads((HERE / "derived_results.json").read_text(encoding="utf-8"))["arms"]
fig = plt.figure(figsize=(8.5, 5.6), constrained_layout=True)
gs = fig.add_gridspec(2, 2)
ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]

# A nontrivial full-complex instance of the artifact-dimension upper bound.
a = next(x for x in arms if x["run"] == "world636_S1")
h = a["E1"]["full_history"]
ax[0].step(a["steps"], h["series_cumulative_rho1_bound"], where="post", lw=2, color="#bb6b12", label="Cumulative bound")
ax[0].step(a["steps"], h["series_alpha1"], where="post", lw=2, color="#345f9c", label="Artifact dimension")
ax[0].axvline(240, ls=":", lw=1.3, color="#555555")
ax[0].set(title="A  Historical-union bound", xlabel="Training step", ylabel="H1 dimension")
ax[0].legend(frameon=False, fontsize=8, loc="upper left")
ax[0].set_ylim(bottom=-.1)

# Simultaneous bootstrap max error versus displayed delta windows.
x = np.arange(len(arms))
qf = [a["E2"]["sup_error"]["all_cells"]["q95"] for a in arms]
qb = [a["E2"]["sup_error"]["B_closed"]["q95"] for a in arms]
ax[1].plot(x, qf, marker="o", color="#bb6b12", label="Full complex")
ax[1].plot(x, qb, marker="s", color="#345f9c", label="B subcomplex")
ax[1].axhline(.10, ls="--", lw=1.3, color="#6953a1")
ax[1].axhline(.05, ls=":", lw=1.3, color="#6953a1")
ax[1].text(4.1, .108, "window 0.10", fontsize=8, color="#6953a1")
ax[1].text(4.1, .058, "window 0.05", fontsize=8, color="#6953a1")
ax[1].set(title="B  Bootstrap max error (95th pct.)", xlabel="Confirmation arm", ylabel="Max score error")
ax[1].set_xticks(x, ["635-1", "635-2", "636-1", "636-2", "637-1", "637-2"], rotation=35, ha="right")
ax[1].legend(frameon=False, fontsize=8, loc="center left")
ax[1].set_ylim(0, .38)

# Paired-root bootstrap, C edges versus the joint face task.
gaps = [a["E3"]["stage_128_384_edge_face_gap"] for a in arms]
mean = np.array([g["mean_edge_minus_face"] for g in gaps])
lo = np.array([g["bootstrap_percentile_95_interval"][0] for g in gaps])
hi = np.array([g["bootstrap_percentile_95_interval"][1] for g in gaps])
ax[2].errorbar(x, mean, yerr=np.vstack([mean - lo, hi - mean]), fmt="o", color="#187e75", capsize=3, lw=1.4)
ax[2].axhline(0, ls=":", lw=1, color="#555555")
ax[2].set(title="C  C-edge minus joint-face score", xlabel="Confirmation arm", ylabel="Mean score gap, steps 128-384")
ax[2].set_xticks(x, ["635-1", "635-2", "636-1", "636-2", "637-1", "637-2"], rotation=35, ha="right")
ax[2].set_ylim(-.055, .16)

for a in ax:
    a.spines[["top", "right"]].set_visible(False)
    a.grid(axis="y", alpha=.18)
fig.savefig(HERE / "e1_e3_evidence.png", dpi=220)
fig.savefig(HERE / "e1_e3_evidence.svg")
fig.savefig(HERE / "e1_e3_evidence.pdf")
