"""Event-level bootstrap for three prespecified curriculum transitions, E2.

This is post-hoc and reports resampling frequencies, not true-event probabilities.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "capability-zigzag-repro"
sys.path.insert(0, str(REPO / "crosslingual" / "code"))
sys.path.insert(0, str(REPO / "paper_sec4"))
import cw_zigzag as CW  # noqa: E402
import zz_analyze as ZA  # noqa: E402
import zz_robust as R  # noqa: E402
from theta_sweep import kappa  # noqa: E402

SPEC = json.loads((REPO / "complex_worlds/data/RCN12_FORGET3/world635/COMPLEX.json").read_text(encoding="utf-8"))
RUNS = REPO / "cloud_return_round4/complex_worlds/runs/RCN12_FORGET3"
PAIRS = ((288, 304), (480, 496), (496, 512))
DELTAS = (0.0, 0.02, 0.05, 0.10)
B = 1000


def calc(cx, lo_a, lo_b, hi_a, hi_b):
    w_lo, w_hi = lo_a | lo_b, hi_a | hi_b
    both = kappa(cx, w_hi, w_lo, [lo_a, lo_b], 1)
    fwd = kappa(cx, w_hi, w_lo, [lo_a], 1)
    bwd = kappa(cx, w_hi, w_lo, [lo_b], 1)
    return fwd - both, bwd - both, both


def arm(run):
    ev = ZA.load_run(run)
    sm = R.ScoreModel(SPEC, ev)
    cx = CW.CWComplex([(n, d, list(b)) for n, d, b in SPEC["cells"]])
    sub = R.Sub(SPEC["cells"], R.K_B_CELLS).bind(cx)
    mk = R.Masker(cx)
    scores = sm.scores(sm.weights(B, 20260925))
    arr = np.stack([scores[n] for n in cx.names], axis=1)
    ix = [(sm.steps.index(a), sm.steps.index(b)) for a, b in PAIRS]
    result = {f"{a}-{b}": {str(d): [] for d in DELTAS} for a, b in PAIRS}
    for r in range(B + 1):
        cache = {}
        for delta in DELTAS:
            for bound in (round(.8 - delta, 4), round(.8 + delta, 4)):
                if bound not in cache:
                    cache[bound] = [sub.remap(m) for m in mk.masks(arr[r], bound)]
            low, high = cache[round(.8 - delta, 4)], cache[round(.8 + delta, 4)]
            for (a, b), (i, j) in zip(PAIRS, ix):
                result[f"{a}-{b}"][str(delta)].append(calc(sub.cx, low[i], low[j], high[i], high[j]))
    compact = {}
    for pair, by_delta in result.items():
        compact[pair] = {}
        for delta, values in by_delta.items():
            v = np.asarray(values)
            compact[pair][delta] = {
                "point_counts_birth_death_transition": v[0].tolist(),
                "bootstrap_positive_fraction_birth_death_transition": np.mean(v[1:] > 0, axis=0).tolist(),
                "bootstrap_mean_counts_birth_death_transition": np.mean(v[1:], axis=0).tolist(),
            }
    return {"run": run.name, "pairs": compact}


def main():
    out = []
    for run in sorted(RUNS.iterdir()):
        if run.is_dir() and any(run.glob("eval_*.jsonl*")):
            a = arm(run)
            out.append(a)
            print(run.name, a["pairs"]["480-496"]["0.05"]["bootstrap_positive_fraction_birth_death_transition"], flush=True)
    (HERE / "event_bootstrap.json").write_text(json.dumps({"B": B, "theta": .8, "arms": out}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
