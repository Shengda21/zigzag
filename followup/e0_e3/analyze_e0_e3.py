"""Post-hoc E0-E3 reanalysis of archived FORGET3 confirmation evaluations.

Reads only source files. Writes derived JSON in this directory. No training or
changes to preregistration, source evaluations, or archived paper results.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "capability-zigzag-repro"
sys.path.insert(0, str(REPO / "crosslingual" / "code"))
sys.path.insert(0, str(REPO / "paper_sec4"))
import cw_zigzag as CW  # noqa: E402
import zz_analyze as ZA  # noqa: E402
import zz_robust as R  # noqa: E402
from mv_counts import spurious_history  # noqa: E402

RUNS = REPO / "cloud_return_round4" / "complex_worlds" / "runs" / "RCN12_FORGET3"
SPEC = json.loads((REPO / "complex_worlds" / "data" / "RCN12_FORGET3" / "world635" / "COMPLEX.json").read_text(encoding="utf-8"))
THETA = 0.8
BOOT = 1000
SEED = 20260925


def finite(x):
    return float(x) if np.isfinite(x) else None


def hist_summary(cx, masks):
    h = spurious_history(cx, masks, 1)
    cumulative = np.cumsum([x["connecting_rank_prev_new"] for x in h]).tolist()
    strict = [i for i, (x, bound) in enumerate(zip(h, cumulative)) if x["spurious"] < bound]
    first_gap = None
    if strict:
        t = strict[0]
        prev = 0
        for m in masks[:t]:
            prev |= m
        cur = prev | masks[t]
        j_cols = [int(v) for v in cx.induced(prev, cur, 1)]
        i_cols = [int(v) for v in cx.induced(masks[t], cur, 1)]
        old_support = [int(v) for m in masks[:t] for v in cx.induced(m, prev, 1)]
        old_support_in_cur = [int(v) for m in masks[:t] for v in cx.induced(m, cur, 1)]
        lost_unsupported = (len(cx.homology(prev, 1)[0]) - CW.rank(old_support)) - (CW.rank(j_cols) - CW.rank(old_support_in_cur))
        overlap_j_i = CW.rank(j_cols) + CW.rank(i_cols) - CW.rank(j_cols + i_cols)
        overlap_s_i = CW.rank(old_support_in_cur) + CW.rank(i_cols) - CW.rank(old_support_in_cur + i_cols)
        first_gap = {"checkpoint_index": t, "loss_of_unsupported_directions": lost_unsupported,
                     "previous_artifact_becomes_supported": overlap_j_i - overlap_s_i,
                     "bound_slack": cumulative[t] - h[t]["spurious"]}
    return {
        "final_beta1": h[-1]["beta"],
        "final_supported_rank": h[-1]["beta"] - h[-1]["spurious"],
        "final_alpha1": h[-1]["spurious"],
        "final_cumulative_rho1_bound": cumulative[-1],
        "strict_bound_checkpoint_indices": strict,
        "first_strict_bound_decomposition": first_gap,
        "series_alpha1": [x["spurious"] for x in h],
        "series_cumulative_rho1_bound": cumulative,
    }


def face_detail(evals, steps, main_templates):
    out = []
    for step in steps:
        rows = [r for r in evals[step] if r.get("task") == "fC123" and r.get("template") in main_templates]
        by = {}
        for override in (False, True):
            part = [r for r in rows if r.get("override") is override]
            fields = [r.get("field_correct") for r in part if isinstance(r.get("field_correct"), list)]
            by[str(override).lower()] = {
                "n_rows": len(part),
                "accuracy": finite(np.mean([r["correct"] for r in part])) if part else None,
                "valid_format": finite(np.mean([r.get("valid_format", False) for r in part])) if part else None,
                "field_accuracy": [finite(np.mean([x[j] for x in fields if len(x) > j])) for j in range(4)] if fields else None,
            }
        out.append({"step": step, "keep": by["false"], "override": by["true"]})
    return out


def analyse_run(run_dir):
    evals = ZA.load_run(run_dir)
    sm = R.ScoreModel(SPEC, evals)
    cx = CW.CWComplex([(n, d, list(b)) for n, d, b in SPEC["cells"]])
    sub = R.Sub(SPEC["cells"], R.K_B_CELLS).bind(cx)
    weights = sm.weights(BOOT, SEED)
    scores = sm.scores(weights)
    arr = np.stack([scores[n] for n in cx.names], axis=1)  # replicate, cell, checkpoint
    steps = sm.steps
    mk = R.Masker(cx)
    full_masks = mk.masks(arr[0], THETA)
    b_masks = [sub.remap(m) for m in full_masks]
    h1_classify, _ = R.make_classifier(cx)
    h1_categories = Counter(h1_classify(a, b)[1] for a, b in zip(full_masks[:-1], full_masks[1:]) if h1_classify(a, b)[0] < 0)

    # A scalar baseline is (cell count, beta0, beta1). Its collisions are
    # descriptive ambiguity, not predictive accuracy or an independent label.
    grouped = defaultdict(set)
    examples = defaultdict(dict)
    for t, m in enumerate(full_masks):
        key = (m.bit_count(), len(cx.homology(m, 0)[0]), len(cx.homology(m, 1)[0]))
        grouped[key].add(m)
        examples[key].setdefault(m, t)
    collisions = sum(len(v) - 1 for v in grouped.values() if len(v) > 1)
    collision_example = None
    for key, states in examples.items():
        if len(states) > 1:
            (m1, t1), (m2, t2) = list(states.items())[:2]
            collision_example = {"steps": [steps[t1], steps[t2]], "scalar_summary_cellcount_beta0_beta1": list(key),
                                 "first_only_cells": [n for i, n in enumerate(cx.names) if m1 >> i & 1 and not m2 >> i & 1],
                                 "second_only_cells": [n for i, n in enumerate(cx.names) if m2 >> i & 1 and not m1 >> i & 1]}
            break
    unchanged_betti = sum(
        a != b and all(len(cx.homology(a, k)[0]) == len(cx.homology(b, k)[0]) for k in (0, 1))
        for a, b in zip(full_masks[:-1], full_masks[1:])
    )

    # Simultaneous bootstrap error is computed on a fixed complete cell/time
    # family. This is an estimate of sampling variability, not a coverage proof.
    dev = np.abs(arr[1:] - arr[0:1])
    b_indices = [cx.index(n) for n in sub.names]
    scope = {}
    b_edge_indices = [cx.index(n) for n in sub.names if SPEC["cells"][cx.index(n)][1] == 1]
    for label, selected in (("all_cells", list(range(len(cx.names)))), ("B_closed", b_indices), ("B_edges_only_diagnostic", b_edge_indices)):
        sup = dev[:, selected, :].reshape(BOOT, -1).max(axis=1)
        scope[label] = {
            "cells": len(selected), "checkpoints": len(steps),
            "median": finite(np.median(sup)), "q95": finite(np.quantile(sup, .95)),
            "fraction_le_005": finite(np.mean(sup <= .05)),
            "fraction_le_010": finite(np.mean(sup <= .10)),
        }
    # The face uses a minimum over two separately scored intervention variants.
    # Their errors must also be tracked before the minimum is applied.
    face_components = {}
    for name in ("fC123", "fB123"):
        if name not in sm.comp:
            continue
        face_components[name] = []
        for task, corr, tot in sm.comp[name]:
            w = weights[task]
            val = (w @ corr.T) / np.maximum((w @ tot)[:, None], 1e-12)
            su = np.abs(val[1:] - val[0]).max(axis=1)
            face_components[name].append({"task": task, "q95_sup_all_checkpoints": finite(np.quantile(su, .95))})

    face = face_detail(evals, steps, set(SPEC["main_templates"]))
    c_edges = [n for n in ("C1C2", "C2C3", "C1C3") if n in scores]
    c = [{"step": step,
          "min_C_edge": finite(min(scores[n][0, i] for n in c_edges)),
          "face_score": finite(scores["fC123"][0, i]),
          "keep_accuracy": face[i]["keep"]["accuracy"],
          "override_accuracy": face[i]["override"]["accuracy"],
          "keep_format": face[i]["keep"]["valid_format"],
          "override_format": face[i]["override"]["valid_format"]}
         for i, step in enumerate(steps)]
    margin = [r for r in c if r["min_C_edge"] >= .8 and r["face_score"] < .8]
    c_stage = [r for r in c if 128 <= r["step"] <= 384]
    ci = np.array([steps.index(s) for s in steps if 128 <= s <= 384])
    # For the edge-versus-face contrast, the four tasks use the SAME held-out
    # roots. Resample a root once and share its weight across those tasks.
    shared_tasks = c_edges + ["fC123"]
    assert all(sm.items[t] == sm.items[shared_tasks[0]] for t in shared_tasks)
    nroot = len(sm.items[shared_tasks[0]])
    rng = np.random.default_rng(SEED + 113)
    paired_w = np.vstack([np.ones(nroot), rng.multinomial(nroot, np.full(nroot, 1 / nroot), size=BOOT)])

    def shared_task_score(cell):
        vals = []
        for _, corr, tot in sm.comp[cell]:
            vals.append((paired_w @ corr.T) / np.maximum((paired_w @ tot)[:, None], 1e-12))
        return np.minimum.reduce(vals) if len(vals) > 1 else vals[0]

    c_edge_boot = np.stack([shared_task_score(n) for n in c_edges], axis=0).min(axis=0)
    c_face_boot = shared_task_score("fC123")
    gap_boot = (c_edge_boot - c_face_boot)[:, ci].mean(axis=1)
    face_gap = {"n_shared_roots": nroot, "mean_edge_minus_face": finite(gap_boot[0]),
                "bootstrap_percentile_95_interval": [finite(x) for x in np.quantile(gap_boot[1:], [.025, .975])],
                "bootstrap_fraction_gap_positive": finite(np.mean(gap_boot[1:] > 0))}
    resolution = {}
    for stride in (1, 2, 4):
        ix = list(range(0, len(steps), stride))
        if ix[-1] != len(steps) - 1:
            ix.append(len(steps) - 1)
        bm = [b_masks[i] for i in ix]
        bs = [steps[i] for i in ix]
        bars = R.fast_barcode(sub.cx, bm, 1, "union")
        resolution[str(stride * 16)] = {
            "checkpoints": len(ix),
            "bridge_only_H1_steps": [[bs[j // 2], bs[j // 2 + 1]] for a, b in bars if a == b and a % 2 == 1 for j in (a,)],
            "checkpoint_supported_H1_bars": sum(any(x % 2 == 0 for x in range(a, b + 1)) for a, b in bars),
        }
    return {
        "run": run_dir.name,
        "n_evaluations": len(steps), "steps": steps,
        "E1": {
            "full_history": hist_summary(cx, full_masks),
            "B_history": hist_summary(sub.cx, b_masks),
            "scalar_summary_distinct_state_collisions": collisions,
            "scalar_collision_example": collision_example,
            "adjacent_mask_changes_with_same_betti01": unchanged_betti,
            "h1_decrease_categories": dict(h1_categories),
            "B_union_H1_bars": [list(b) for b in R.fast_barcode(sub.cx, b_masks, 1, "union")],
            "B_intersection_H1_bars": [list(b) for b in R.fast_barcode(sub.cx, b_masks, 1, "intersection")],
        },
        "E2": {"bootstrap_B": BOOT, "seed": SEED, "sup_error": scope,
               "face_raw_component_error": face_components,
               "checkpoint_grid_sensitivity": resolution},
        "E3": {"C_edges": c_edges,
               "checkpoints_edges_ge_08_face_lt_08": [r["step"] for r in margin],
               "stage_128_384_mean_min_edge": finite(np.mean([r["min_C_edge"] for r in c_stage])),
               "stage_128_384_mean_face": finite(np.mean([r["face_score"] for r in c_stage])),
               "stage_128_384_mean_keep_accuracy": finite(np.mean([r["keep_accuracy"] for r in c_stage])),
               "stage_128_384_mean_override_accuracy": finite(np.mean([r["override_accuracy"] for r in c_stage])),
               "stage_128_384_edge_face_gap": face_gap,
               "series": c, "field_series": face},
    }


def main():
    arms = []
    for d in sorted(RUNS.iterdir()):
        if d.is_dir() and any(d.glob("eval_*.jsonl*")):
            arm = analyse_run(d)
            arms.append(arm)
            print(d.name, "B alpha", arm["E1"]["B_history"]["final_alpha1"],
                  "B q95", arm["E2"]["sup_error"]["B_closed"]["q95"], flush=True)
    out = {"scope": "post-hoc CPU reanalysis of six confirmation arms",
           "source": str(RUNS), "theta": THETA, "arms": arms}
    (HERE / "derived_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
