"""Design-matched confidence-complex diagnostic; no training or source mutation.

The observed design is finite-population SRS, not iid roots. Single-root
template coordinates use exact hypergeometric inversion; the cyclic paired
override tasks use a permutation bounded-difference interval. An explicitly
labelled iid CP analysis is a counterfactual calibration, never the real-data
coverage claim. The entire six-arm B-cell/checkpoint/component/template family
is protected once; threshold/window inspection does not change that family.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.stats import beta, hypergeom, binom

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] / "capability-zigzag-repro"
sys.path.insert(0, str(REPO / "crosslingual/code"))
sys.path.insert(0, str(REPO / "paper_sec4"))
import cw_zigzag as CW
import zz_analyze as ZA
import zz_robust as R
from theta_sweep import kappa

SPEC = json.loads((REPO / "complex_worlds/data/RCN12_FORGET3/world635/COMPLEX.json").read_text(encoding="utf-8"))
RUNS = REPO / "cloud_return_round4/complex_worlds/runs/RCN12_FORGET3"
ALPHA = .05
THETAS = (.4, .5, .6, .65, .7, .75, .8, .85, .9)
DELTAS = (0., .05, .1)
SEED = 20261108


def dump(name, obj):
    (HERE / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def cp_lut(n, tail):
    x = np.arange(n + 1)
    lo = np.zeros(n + 1)
    hi = np.ones(n + 1)
    lo[1:] = beta.ppf(tail, x[1:], n - x[1:] + 1)
    hi[:-1] = beta.ppf(1 - tail, x[:-1] + 1, n - x[:-1])
    return lo, hi


def hg_lut(n, N, tail):
    """Equal-tailed inversion for unknown integer successes K in finite N.

    Retain K when P_K[X>=x] >= tail and P_K[X<=x] >= tail.
    A true K is excluded by either tail with probability <= 2*tail.
    """
    pop = np.arange(N + 1)
    lo, hi = [], []
    for x in range(n + 1):
        allowed = (hypergeom.sf(x - 1, N, pop, n) >= tail) & (hypergeom.cdf(x, N, pop, n) >= tail)
        valid = pop[allowed]
        assert valid.size
        lo.append(valid.min() / N)
        hi.append(valid.max() / N)
    return np.array(lo), np.array(hi)


def bcomplex():
    full = CW.CWComplex([(n, d, list(b)) for n, d, b in SPEC["cells"]])
    sub = R.Sub(SPEC["cells"], R.K_B_CELLS).bind(full)
    return full, sub


def extract(run):
    ev = ZA.load_run(run)
    steps = sorted(ev)
    _, sub = bcomplex()
    cell_dims = {n: d for n, d, _ in sub.cells}
    channels = []
    samples = []
    templates = tuple(SPEC["main_templates"])
    for cell in sub.names:
        task = "v" + cell if cell_dims[cell] == 0 else cell
        for override in ((False, True) if cell_dims[cell] == 2 else (None,)):
            for template in templates:
                by_step = []
                keys0 = None
                for step in steps:
                    rows = [r for r in ev[step] if r.get("vocabulary") == "rcn" and r.get("task") == task
                            and r.get("template") == template
                            and (override is None or r.get("override") is override)]
                    keyed = {R.item_key(r): r for r in rows}
                    assert len(keyed) == len(rows), (run.name, cell, template)
                    keys = sorted(keyed)
                    if keys0 is None:
                        keys0 = keys
                    assert keys == keys0
                    by_step.append([int(keyed[k]["correct"]) for k in keys])
                    if override is True:
                        assert all(r.get("other") != r.get("root") for r in rows)
                vals = np.asarray(by_step, dtype=np.int8)
                if vals.shape[1] == 0:
                    continue  # faces have only template0 in this actual design
                channels.append(dict(cell=cell, override=override, template=template,
                                     n=vals.shape[1], paired=override is True))
                samples.append(vals)
    # Cross-check equality with the archived score definition, not a redefined
    # per-prompt/independent-paraphrase score.
    sm = R.ScoreModel(SPEC, ev)
    archived = sm.scores(sm.weights(0, SEED))
    means = [x.mean(axis=1) for x in samples]
    merged = merge(channels, np.asarray(means), sub.names)
    for i, cell in enumerate(sub.names):
        assert np.allclose(merged[i], archived[cell][0])
    return dict(run=run.name, steps=steps, channels=channels, samples=samples,
                nroot_summary=sorted(set(c["n"] for c in channels)))


def merge(channels, values, names):
    """Average fixed paraphrases within variant, then min variants per cell."""
    out = []
    for cell in names:
        variants = defaultdict(list)
        for i, c in enumerate(channels):
            if c["cell"] == cell:
                variants[c["override"]].append(values[i])
        out.append(np.min([np.mean(v, axis=0) for v in variants.values()], axis=0))
    return np.asarray(out)


def confidence_tables(arm, m, method):
    _, sub = bcomplex()
    tail = ALPHA / (2 * m)
    means, lowers, uppers = [], [], []
    for c, sample in zip(arm["channels"], arm["samples"]):
        n = c["n"]
        x = sample.sum(axis=1)
        mean = x / n
        if method == "design_matched":
            if c["paired"]:
                # Expose the ordered n-root SRS sequentially. A transposition
                # changes at most four directed cycle terms, hence range <=4/n.
                eps = 4 * np.sqrt(np.log(1 / tail) / (2 * n))
                lower, upper = np.maximum(0, mean - eps), np.minimum(1, mean + eps)
            else:
                lut = hg_lut(n, 64, tail)
                lower, upper = lut[0][x], lut[1][x]
        elif method == "iid_cp_counterfactual":
            lut = cp_lut(n, tail)
            lower, upper = lut[0][x], lut[1][x]
        else:
            raise ValueError(method)
        means.append(mean)
        lowers.append(lower)
        uppers.append(upper)
    means, lowers, uppers = map(np.asarray, (means, lowers, uppers))
    # Scalar compression of the SAME simultaneous band. No local-family
    # reallocation, no extra assumption, no difference in confidence level.
    point = merge(arm["channels"], means, sub.names)
    coord = (merge(arm["channels"], lowers, sub.names), merge(arm["channels"], uppers, sub.names))
    # Compress AFTER the actual score map (paraphrase averaging and variant
    # minimum). A raw-component maximum can be needlessly inflated by a
    # component that is irrelevant to the final minimum.
    eps = float(np.max(np.maximum(point - coord[0], coord[1] - point)))
    uniform = (np.maximum(0, point - eps), np.minimum(1, point + eps))
    assert np.all(uniform[0] <= coord[0] + 1e-12)
    assert np.all(coord[1] <= uniform[1] + 1e-12)
    return point, coord, uniform, eps, means, lowers, uppers


def certificate(cx, lower, upper, steps):
    masker = R.Masker(cx)
    out = {}
    for theta in THETAS:
        bydelta = {}
        for delta in DELTAS:
            lhi = masker.masks(lower, theta + delta)
            ulo = masker.masks(upper, theta - delta)
            assert all(a & b == a for a, b in zip(lhi, ulo))
            checkpoints = [CW.rank([int(x) for x in cx.induced(a, b, 1)]) for a, b in zip(lhi, ulo)]
            records = []
            for t in range(len(steps) - 1):
                lw, uw = lhi[t] | lhi[t + 1], ulo[t] | ulo[t + 1]
                # These are independent valid lower bounds on three cokernel
                # ranks; their differences are NOT lower bounds on b or d.
                c = [kappa(cx, lw, uw, src, 1) for src in
                     ([ulo[t]], [ulo[t + 1]], [ulo[t], ulo[t + 1]])]
                if any(c):
                    records.append(dict(steps=steps[t:t + 2],
                                        lower_kappa_forward_backward_transition=c))
            bydelta[str(delta)] = dict(checkpoint_image_rank_lower=checkpoints,
                                       n_positive_checkpoints=int(np.count_nonzero(checkpoints)),
                                       sum_checkpoint_image_rank_lower=int(sum(checkpoints)),
                                       positive_transition_records=records,
                                       n_positive_kappa=sum(any(r["lower_kappa_forward_backward_transition"]) for r in records),
                                       n_positive_transition=sum(r["lower_kappa_forward_backward_transition"][2] > 0 for r in records))
        out[str(theta)] = bydelta
    return out


def real_data():
    arms = [extract(run) for run in sorted(RUNS.iterdir()) if run.is_dir() and any(run.glob("eval_*.jsonl*"))]
    assert len(arms) == 6
    m = sum(len(a["channels"]) * len(a["steps"]) for a in arms)
    _, sub = bcomplex()
    results = []
    for arm in arms:
        res = dict(run=arm["run"], steps=arm["steps"], nroot_summary=arm["nroot_summary"], methods={})
        for method in ("design_matched", "iid_cp_counterfactual"):
            point, coord, uniform, eps, means, low, upp = confidence_tables(arm, m, method)
            np.savez_compressed(HERE / f'{arm["run"]}_{method}.npz', point=point, lower=coord[0], upper=coord[1],
                                channel_mean=means, channel_lower=low, channel_upper=upp)
            channels = arm["channels"]
            extreme = {}
            for n in arm["nroot_summary"]:
                tail = ALPHA / (2 * m)
                hg, cp = hg_lut(n, 64, tail), cp_lut(n, tail)
                extreme[str(n)] = dict(finitepop_all_correct_lower=float(hg[0][-1]),
                                       iid_all_correct_lower=float(cp[0][-1]),
                                       finitepop_all_wrong_upper=float(hg[1][0]))
            res["methods"][method] = dict(global_epsilon=eps,
                raw_component_epsilon_diagnostic=float(np.max(np.maximum(means-low,upp-means))),
                extreme_score_calibration=extreme,
                coordinate_aware=certificate(sub.cx, *coord, arm["steps"]),
                scalar_compression=certificate(sub.cx, *uniform, arm["steps"]))
            print(arm["run"], method, "eps", eps, "positive checkpoints theta .5/.6/.8",
                  [res["methods"][method]["coordinate_aware"][str(t)]["0.0"]["n_positive_checkpoints"] for t in (.5, .6, .8)], flush=True)
        results.append(res)
        dump("real_results.json", dict(alpha=ALPHA, family="all six arms, every B-cell component/template and all 49 checkpoints",
             m_template_coordinates=m, cells=sub.names, methods_scope={
                 "design_matched": "Conditional finite-population remaining64 mean; SRS roots and cyclic-pair permutation bound. Assumes fitted models and full saved grid fixed independently of structure partition/order.",
                 "iid_cp_counterfactual": "Diagnostic only; requires independently drawn root/pair units, absent in archive."}, arms=results))
    return m


def validate_algebra():
    # All closed subcomplexes of the minimal parallel-edge complex, all nested
    # triples L <= X <= U at both endpoints, all three source patterns.
    cx = CW.CWComplex([("v", 0, []), ("w", 0, []), ("e", 1, ["v", "w"]), ("f", 1, ["v", "w"])])
    masks = [m for m in range(16) if all(not (m >> i & 1) or m & cl == cl for i, cl in enumerate(cx.closure))]
    triples = [(l, x, u) for l in masks for x in masks for u in masks if l & x == l and x & u == x]
    count = 0
    for la, a, ua in triples:
        for lb, b, ub in triples:
            for pattern in ((0,), (1,), (0, 1)):
                lbnd = kappa(cx, la | lb, ua | ub, [[ua, ub][i] for i in pattern], 1)
                truth = kappa(cx, a | b, a | b, [[a, b][i] for i in pattern], 1)
                assert lbnd <= truth
                count += 1
    # A boundary circle included in its filled disk demonstrates that Betti
    # numbers do not inherit the set-inclusion sandwich.
    cells = [("a",0,[]),("b",0,[]),("c",0,[]),("ab",1,["a","b"]),
             ("bc",1,["b","c"]),("ac",1,["a","c"]),("D",2,["ab","bc","ac"])]
    disk = CW.CWComplex(cells)
    assert len(disk.homology(63, 1)[0]) == 1 and len(disk.homology(127, 1)[0]) == 0
    # Invert exact tails and check frequentist coverage by enumeration at every
    # possible finite population K, rather than simulation alone.
    tail = ALPHA / (2 * 10000)
    hg_checks = []
    for n in (16, 32):
        lo, hi = hg_lut(n, 64, tail)
        worst = 0.
        for K in range(65):
            x = np.arange(n + 1)
            fail = (lo > K/64 + 1e-14) | (hi < K/64 - 1e-14)
            worst = max(worst, float(hypergeom.pmf(x, 64, K, n)[fail].sum()))
        assert worst <= 2 * tail + 1e-12
        hg_checks.append(dict(n=n, max_noncoverage=worst, allocated_noncoverage=2 * tail))
    out = dict(exhaustive_cokernel_lower_inequalities=count, n_nested_triples=len(triples),
               hypergeometric_exact_coverage_enumeration=hg_checks,
               betti_sandwich_counterexample=dict(circle_beta1=1, disk_beta1=0))
    dump("validation.json", out)
    return out


def simulations(reps=1000):
    # IID independent ROOT units, two perfectly correlated paraphrases; reuse
    # root uniforms across all coordinates so no coordinate/time independence
    # can be inadvertently responsible for coverage.
    cells = [("a",0,[]),("b",0,[]),("c",0,[]),("ab",1,["a","b"]),
             ("bc",1,["b","c"]),("ac",1,["a","c"]),("D",2,["ab","bc","ac"])]
    cx = CW.CWComplex(cells)
    mk = R.Masker(cx)
    p = np.array([[.99]*3]*3 + [[.95,.10,.95],[.95,.95,.95],[.10,.95,.95],[.05]*3])
    scenarios = {"clear_transition_and_birth":p,
                 "near_threshold_null":np.array([[.99]*3]*3 + [[.79]*3]*3 + [[.05]*3])}
    # Same fixed 42 template coordinates for both procedures in each experiment.
    m = p.size * 2
    rng = np.random.default_rng(SEED)
    out = []
    for scenario, truth in scenarios.items():
        for n in (32, 128, 512, 2048):
            lut = cp_lut(n, ALPHA / (2*m))
            # Cache the rank functions: 7-cell masks yield a tiny finite state.
            stats = {method:dict(band_fail=0, rank_false_certificates=0, counts=np.zeros((3,3), dtype=int),
                                      checkpoint_ranks=0) for method in ("coordinate_cp", "scalar_same_cp")}
            true_counts = []
            for d in (0., .05, .1):
                hi, lo = mk.masks(truth, .8+d), mk.masks(truth, .8-d)
                true_counts.append([kappa(cx, hi[0]|hi[1],lo[0]|lo[1],[lo[0],lo[1]],1),
                                    kappa(cx, hi[1]|hi[2],lo[1]|lo[2],[lo[1]],1),
                                    CW.rank(list(cx.induced(hi[2],lo[2],1)))])
            for rep in range(reps):
                uniforms = rng.random(n)
                counts = (uniforms[None,None,:] < truth[:,:,None]).sum(axis=2)
                point = counts / n
                lower, upper = lut[0][counts], lut[1][counts]
                eps = np.max(np.maximum(point-lower,upper-point))
                bands = {"coordinate_cp":(lower, upper), "scalar_same_cp":(np.maximum(0,point-eps),np.minimum(1,point+eps))}
                for method, (L,U) in bands.items():
                    st=stats[method]
                    st["band_fail"] += int(np.any((truth < L-1e-12)|(truth > U+1e-12)))
                    for di,d in enumerate((0.,.05,.1)):
                        hi,lo=mk.masks(L,.8+d),mk.masks(U,.8-d)
                        c=[kappa(cx,hi[0]|hi[1],lo[0]|lo[1],[lo[0],lo[1]],1),
                           kappa(cx,hi[1]|hi[2],lo[1]|lo[2],[lo[1]],1),
                           CW.rank(list(cx.induced(hi[2],lo[2],1)))]
                        st["counts"][di] += np.array(c)>0
                        st["rank_false_certificates"] += int(np.any(np.array(c)>true_counts[di]))
            for method, st in stats.items():
                out.append(dict(scenario=scenario,n_roots=n,reps=reps,method=method,
                                simultaneous_band_noncoverage=st["band_fail"]/reps,
                                false_rank_certificates=st["rank_false_certificates"],
                                true_transition_forward_checkpoint_ranks_by_delta=true_counts,
                                positive_certificate_fraction_by_delta=(st["counts"]/reps).tolist()))
            print("simulation", scenario, n, "power delta0", {m:s["counts"][0].tolist() for m,s in stats.items()}, flush=True)
    # Worst-case all-success necessary calibration for iid, same family as real.
    dump("simulation_results.json",dict(alpha=ALPHA,seed=SEED,m_template_coordinates=m,
         correlation="Paraphrases identical; coordinates and times share per-root U; roots iid.",
         columns=["bridge_transition_0to1", "forward_cokernel_1to2", "checkpoint_H1_image_rank_t2"],deltas=[0,.05,.1],results=out))


def calibration(real_m):
    rows=[]
    for m in (real_m, 42):
        tail=ALPHA/(2*m)
        for theta in (.6,.7,.8,.85,.9):
            # All n successes: exact CP lower=tail**(1/n); n is a necessary
            # best-case root count, not a realistic power calculation.
            cp_n=int(np.ceil(np.log(tail)/np.log(theta)))
            h_n=int(np.ceil(np.log(1/tail)/(2*(1-theta)**2)))
            rows.append(dict(family_m=m,theta=theta,best_case_CP_all_correct_n=cp_n,
                             Hoeffding_all_correct_n=h_n))
    power=[]
    for n in (64,128,256,512,1024):
        lo,_=cp_lut(n,ALPHA/(2*real_m))
        for theta in (.8,.85,.9):
            eligible=np.flatnonzero(lo >= theta)
            critical=int(eligible[0]) if eligible.size else n+1
            power.append(dict(n=n,theta=theta,true_coordinate_p=.95,
                              minimum_successes=critical,
                              marginal_probability_CP_lower_ge_theta=float(binom.sf(critical-1,n,.95))))
    dump("sample_calibration.json",dict(alpha=ALPHA,scope="iid fresh-root/pair protocol; all-correct rows are best case, power rows are single-coordinate power only",rows=rows,power=power))


if __name__ == "__main__":
    validate_algebra()
    m=real_data()
    calibration(m)
    simulations()
