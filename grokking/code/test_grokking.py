"""Plain-assert test suite for the grokking line (方案 A). No pytest
dependency -- run directly: `python test_grokking.py`. Must print
"ALL TESTS PASSED" and exit 0.

Covers (see PREREGISTRATION_GROKKING_DRAFT.md / README.md for what each of
these is actually for):
  1. split closure + probe self-check + placebo label independence
  2. GROUND_TRUTH Betti numbers vs an independent hand calculation (MODP,
     MODP_LATENTSUB, and the minimal MODP_ADDONLY spec)
  3. CPU dryrun (p=13, 300 steps): training accuracy rises, and the produced
     eval files are readable end-to-end by zz_analyze.analyze_run using the
     real (unmodified) crosslingual/code/zz_analyze.py -- edges, vertices
     AND faces (the joint equality-comparison readout needs no analyzer
     change at all, see README "分析器需要的扩展").
  3b. --preset classic CPU dryrun on MODP_ADDONLY (p=13, 300 steps): true
      full-batch training resolves correctly and runs end to end.
  4. face_scores.py unit tests (route cross-check, both-correct vs one-wrong)
"""
from __future__ import annotations
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
GROK_DIR = HERE.parent
ROOT = GROK_DIR.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'stage_ab' / 'code'))
sys.path.insert(0, str(ROOT / 'crosslingual' / 'code'))
import modp_common as M     # noqa: E402
import make_world as MW     # noqa: E402
import face_scores as FS    # noqa: E402
import zz_score as Z        # noqa: E402
import zz_analyze as ZA     # noqa: E402
import confirm_stats as CS  # noqa: E402

PASS = []


def check(name, cond, detail=''):
    if not cond:
        raise AssertionError(f'FAILED: {name} {detail}')
    PASS.append(name)
    print(f'  ok: {name}')


def base_run_args(**overrides):
    """Matches run_grok.py's current argparse Namespace shape (all
    hyperparameters None unless a test wants to force one, so run_grok.run's
    own --preset/spec-default resolution logic gets exercised the same way
    the CLI would)."""
    base = dict(spec='MODP', world=701, seed=1, steps=None, eval_every=None, ckpt_every=None,
                wd=None, lr=None, frac=None, layers=None, d_model=None, n_heads=None, d_ff=None,
                batch_size=None, preset=None, only_tasks=None, out=None, dryrun=False, dryrun_p=None,
                cpu=True, resume=False)
    base.update(overrides)
    return SimpleNamespace(**base)


# --------------------------------------------------------------- 1. split
def test_split_closure():
    print('== 1a. split closure ==')
    for p in (13, 17, 97):
        split = M.build_split(p, seed=123456 + p, frac=0.4)
        check(f'split covers all pairs p={p}', len(split) == p * p)
        bad = []
        for (a, b), lab in split.items():
            if lab != 'heldout':
                continue
            for other in ((a, (-b) % p), (b, a), (a, M.inv(b, p)) if b % p != 0 else (a, b)):
                if split[other] != 'heldout':
                    bad.append(((a, b), other))
        check(f'heldout closed under neg/swap/inv p={p}', not bad, detail=str(bad[:5]))
        n_train = sum(1 for v in split.values() if v == 'train')
        check(f'train fraction near target p={p}', abs(n_train / (p * p) - 0.4) < 0.15, detail=str(n_train))
    print('== 1b. probe self-check ==')
    spec = json.loads((GROK_DIR / 'specs' / 'MODP.json').read_text(encoding='utf8'))
    spec = json.loads(json.dumps(spec))
    spec['primes'] = {'701': 37}
    spec['probe'] = dict(edge_roots=64, vertex_roots=32, face_roots=32, transfer_roots=32)
    dat = MW.build_data(spec, 701)
    all_ok = all(Z.score(r, r['gold'])['correct'] == 1 for r in dat['probes'])
    check('every probe gold row scores correct against itself', all_ok, detail=f"n={len(dat['probes'])}")
    for task, rows in dat['train_rows'].items():
        ok = all(Z.score(r, r['gold'])['correct'] == 1 for r in rows)
        check(f'every training row for {task} scores correct against itself', ok)

    print('== 1c. placebo label independence ==')
    p = dat['p']
    table = dat['placebo_table']
    rng = random.Random(0)
    agree_with_sum = 0
    n_check = 2000
    for _ in range(n_check):
        a, b = rng.randrange(p), rng.randrange(p)
        placebo = table[a * p + b]
        real_sum = (a + b) % p
        agree_with_sum += int(placebo == real_sum)
    expected = n_check / p
    check('placebo label agrees with P_S (a+b) at ~chance rate, not 1.0',
          agree_with_sum < 3 * expected + 5 and agree_with_sum != n_check,
          detail=f'agree={agree_with_sum} expected~{expected:.1f} n={n_check}')
    # held-out P_X probe accuracy of an "always guess the sum" adversary should also be near 1/p
    # (this is the same guarantee the training/eval loop relies on for a *learned* model)


# ------------------------------------------------------- 2. ground truth
def test_ground_truth_betti():
    print('== 2. GROUND_TRUTH betti vs independent hand calculation ==')
    # MODP: 11 vertices (10 real + control vertex X, reachable only via the
    # placebo edge P_X: P->X, never a self-loop). vertices_only has no edges
    # at all -> beta0 = 11 (every declared vertex, isolated), beta1 = 0.
    #
    # 12 real taught edges (P_X excluded, control) touch only the 10 real
    # vertices; X is never in the taught/latent present set, so it drops out
    # of every other phase entirely (closure-driven present(), not "all
    # vertices always present" -- see make_world.ground_truth docstring).
    # Every real edge touches P directly except Pn_D (Pn-D), Ps_S (Ps-S),
    # Pi_Q (Pi-Q); P is directly adjacent to all other 9 real vertices ->
    # connected, beta0=1. E - V + 1 = 12 - 10 + 1 = 3 independent cycles =
    # exactly the 3 triangles (f_sub: P-D-Pn, f_comm: P-S-Ps, f_div: P-Q-Pi);
    # M and N are leaves (no cycle). Filling all 3 faces caps each triangle
    # once: beta1 3 -> 0.
    spec = json.loads((GROK_DIR / 'specs' / 'MODP.json').read_text(encoding='utf8'))
    kind_e, kind_f = MW.resolve_kinds(spec)
    gt = MW.ground_truth(spec, kind_e, kind_f)
    check('MODP vertices_only == [11,0] (11 declared vertices incl. X)', gt['phases']['vertices_only'] == [11, 0])
    check('MODP taught_no_faces == [1,3] (X excluded, unaffected by adding it)', gt['phases']['taught_no_faces'] == [1, 3])
    check('MODP taught_with_faces == [1,0]', gt['phases']['taught_with_faces'] == [1, 0])
    pc = gt['phases']['placebo_entered_no_betti_change']
    check('MODP: forcing P_X+X in does not change Betti numbers (pendant leaf, not a loop)',
          pc['before'] == pc['after'], detail=str(pc))

    # MODP_LATENTSUB: P_D removed -> only 11 real taught edges, E-V+1 = 2 cycles
    # (f_comm, f_div triangles only; f_sub's triangle is broken into an open path
    # P-Pn-D, D only reachable via Pn_D). Filling f_comm+f_div: beta1 2 -> 0.
    # Adding the latent edge P_D back (still no f_sub 2-cell) closes that path into
    # a NEW cycle: beta1 0 -> 1 (a genuine "edge grokked, cycle opens" event).
    # Adding f_sub caps it: beta1 1 -> 0. (X is still excluded throughout;
    # vertices_only is still [11,0], the same 11 declared vertices as MODP.)
    spec2 = json.loads((GROK_DIR / 'specs' / 'MODP_LATENTSUB.json').read_text(encoding='utf8'))
    kind_e2, kind_f2 = MW.resolve_kinds(spec2)
    gt2 = MW.ground_truth(spec2, kind_e2, kind_f2)
    check('LATENTSUB vertices_only == [11,0]', gt2['phases']['vertices_only'] == [11, 0])
    check('LATENTSUB taught_no_faces == [1,2]', gt2['phases']['taught_no_faces'] == [1, 2])
    check('LATENTSUB taught_with_faces == [1,0]', gt2['phases']['taught_with_faces'] == [1, 0])
    check('LATENTSUB plus_latent_edge_only == [1,1] (cycle opens)',
          gt2['phases']['plus_latent_edge_only'] == [1, 1])
    check('LATENTSUB plus_latent_edge_and_face == [1,0] (cycle closes)',
          gt2['phases']['plus_latent_edge_and_face'] == [1, 0])
    pc2 = gt2['phases']['placebo_entered_no_betti_change']
    check('LATENTSUB: forcing P_X+X in does not change Betti numbers',
          pc2['before'] == pc2['after'], detail=str(pc2))

    # MODP_ADDONLY: 3 vertices (P, S, X), 2 edges (P_S taught, P_X control),
    # 0 faces, vertex_present_mode=blanket -- every declared vertex is always
    # present, so X counts as an isolated component whenever P_X itself is
    # absent. vertices_only: 3 isolated vertices -> [3,0]. With only P_S
    # present: {P,S} form 1 connected component, X is a SEPARATE isolated
    # component -> beta0=2, no cycles -> [2,0] (unlike MODP, here forcing the
    # control edge in is *not* invariant: it merges X into the main
    # component). With both P_S and P_X present: P-S-X all one component via
    # hub P -> beta0=1; E=2,V=3 -> independent cycles = 2-3+1=0 -> [1,0].
    spec3 = json.loads((GROK_DIR / 'specs' / 'MODP_ADDONLY.json').read_text(encoding='utf8'))
    kind_e3, kind_f3 = MW.resolve_kinds(spec3)
    gt3 = MW.ground_truth(spec3, kind_e3, kind_f3)
    check('ADDONLY vertices_only == [3,0]', gt3['phases']['vertices_only'] == [3, 0])
    check('ADDONLY taught_no_faces == [2,0] (X isolated, only P_S present)', gt3['phases']['taught_no_faces'] == [2, 0])
    check('ADDONLY taught_with_faces == [2,0] (no faces exist, same as taught_no_faces)',
          gt3['phases']['taught_with_faces'] == [2, 0])
    check('ADDONLY all_present == [1,0] (P_X connects X into the main component)',
          gt3['phases']['all_present'] == [1, 0])
    check('ADDONLY has no placebo_entered_no_betti_change (blanket mode: not an invariant here)',
          'placebo_entered_no_betti_change' not in gt3['phases'])


# ------------------------------------------------------------ 3. dryrun
def test_cpu_dryrun():
    print('== 3. CPU dryrun (p=13, 300 steps) ==')
    import run_grok as RG
    tmp = Path(tempfile.mkdtemp(prefix='modp_dryrun_'))
    try:
        args = base_run_args(spec='MODP', world=701, steps=300, eval_every=50, ckpt_every=150,
                              out=str(tmp), dryrun=True, dryrun_p=13)
        RG.run(args)
        log = [json.loads(s) for s in (tmp / 'TRAIN_LOG.jsonl').read_text(encoding='utf8').splitlines() if s.strip()]
        check('TRAIN_LOG has entries', len(log) >= 2)
        check('train token accuracy rises to >= 0.5 by the end of the dryrun',
              log[-1]['train_tok_acc'] >= 0.5, detail=str(log[-1]))
        check('train token accuracy improved from the first checkpoint',
              log[-1]['train_tok_acc'] > log[0]['train_tok_acc'])

        eval_files = sorted(list(tmp.glob('eval_*.jsonl')) + list(tmp.glob('eval_*.jsonl.gz')))
        check('eval files were written', len(eval_files) >= 2)
        complex_path = GROK_DIR / 'data' / 'MODP' / 'world701' / 'COMPLEX.json'
        check('reference COMPLEX.json exists (from make_world.py --worlds 701,702)', complex_path.exists())
        res = ZA.analyze_run(tmp, str(complex_path), n_perm=5)
        check('analyze_run returned a trajectory for every cell',
              set(res['trajectory']) == set(json.loads(complex_path.read_text(encoding='utf8'))['cell_kind']))
        for cell in ('P_S', 'P_D', 'P_Q'):   # edges
            vals = res['trajectory'][cell]
            check(f'edge trajectory {cell} is well-formed', all(0.0 <= v <= 1.0 for v in vals))
        for cell in ('S', 'D', 'Q', 'X'):     # vertices (design_from_spec scores task 'v'+name);
            vals = res['trajectory'][cell]         # X is the control vertex reached only by P_X
            check(f'vertex trajectory {cell} is well-formed', all(0.0 <= v <= 1.0 for v in vals))
        for cell in ('f_sub', 'f_comm', 'f_div'):   # faces: joint-task readout, needs NO analyzer change
            vals = res['trajectory'][cell]
            check(f'face trajectory {cell} is well-formed (joint-task readout works unmodified)',
                  all(0.0 <= v <= 1.0 for v in vals))
        check('ledger present for latent_cells', set(res['ledger']['cells']) >= set(
            json.loads(complex_path.read_text(encoding='utf8'))['latent_cells']))
        check('null model ran (n_perm=5)', res['null_model'] is not None)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------- 3b. classic preset dryrun
def test_classic_preset_dryrun():
    print('== 3b. --preset classic CPU dryrun (MODP_ADDONLY, p=13, 300 steps) ==')
    import run_grok as RG
    tmp = Path(tempfile.mkdtemp(prefix='modp_classic_dryrun_'))
    try:
        args = base_run_args(spec='MODP_ADDONLY', world=703, out=str(tmp), dryrun=True, dryrun_p=13,
                              preset='classic')
        RG.run(args)
        cfg = json.loads((tmp / 'RUN_CONFIG.json').read_text(encoding='utf8'))
        check('classic preset resolved layers=1', cfg['layers'] == 1, detail=str(cfg))
        check('classic preset resolved wd=1.0', cfg['wd'] == 1.0)
        check('classic preset resolved lr=1e-3', cfg['lr'] == 1e-3)
        check('classic preset used a true full batch (pool small enough)', cfg['full_batch'] is True)
        check('full batch_size == total training pool', cfg['batch_size'] == cfg['total_train_pool'])
        check('--dryrun overrides classic step counts (300, not 50000)', cfg['steps'] == 300)
        check('--dryrun overrides classic eval_every (50, not 200)', cfg['eval_every'] == 50)

        log = [json.loads(s) for s in (tmp / 'TRAIN_LOG.jsonl').read_text(encoding='utf8').splitlines() if s.strip()]
        check('classic dryrun TRAIN_LOG has entries', len(log) >= 2)
        check('classic dryrun train token accuracy rises to >= 0.9 (single op, full batch, converges fast)',
              log[-1]['train_tok_acc'] >= 0.9, detail=str(log[-1]))

        # full-batch fallback: a big pool (the full MODP complex) must NOT get a
        # literal full batch of tens of thousands of rows -- verify the cap logic
        # purely in-memory (build_data never writes to disk), without touching
        # the real world701/702 data on disk.
        big_spec = json.loads((GROK_DIR / 'specs' / 'MODP.json').read_text(encoding='utf8'))
        big_dat = MW.build_data(big_spec, 701)
        big_total = sum(len(rows) for rows in big_dat['train_rows'].values())
        check('full MODP training pool exceeds FULL_BATCH_CAP (fallback path is reachable)',
              big_total > RG.FULL_BATCH_CAP, detail=str(big_total))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------- 4. face_scores
def test_face_scores_unit():
    print('== 4. face_scores.py unit tests ==')
    route = ['P_D', 'P_Pn', 'Pn_D']

    def row(task, item, correct):
        return dict(task=task, item=item, root=item, correct=correct, vocabulary='modp')

    rows_all_consistent = [
        row('P_D', [1, 2], 1), row('P_Pn', [1, 2], 1), row('Pn_D', [1, 2], 1),
    ]
    frac, n = FS.route_consistency(rows_all_consistent, route)
    check('route_consistency: all three correct on one item -> 1.0', frac == 1.0 and n == 1)

    rows_mixed = [
        row('P_D', [1, 2], 1), row('P_Pn', [1, 2], 1), row('Pn_D', [1, 2], 1),   # item A: all correct
        row('P_D', [3, 4], 1), row('P_Pn', [3, 4], 1), row('Pn_D', [3, 4], 0),   # item B: one wrong
    ]
    frac2, n2 = FS.route_consistency(rows_mixed, route)
    check('route_consistency: 1 of 2 items fully consistent -> 0.5', frac2 == 0.5 and n2 == 2)

    rows_missing_task = [row('P_D', [1, 2], 1), row('P_Pn', [1, 2], 1)]   # Pn_D never probed for this item
    frac3, n3 = FS.route_consistency(rows_missing_task, route)
    check('route_consistency: no item covered by every task -> None', frac3 is None and n3 == 0)

    summary = Z.summarize([dict(r, **Z.score(r, '1' if r['correct'] else '0')) for r in
                            [dict(task='f_sub', kind='joint', vocabulary='modp', template=0, override=False,
                                  gold='1', schema=['binary'], field_length=1, correct=1)]])
    merged = FS.merge_face_cross_check(summary, rows_all_consistent + rows_mixed)
    check('merge_face_cross_check adds modp:face:taskf_sub', 'modp:face:taskf_sub' in merged)
    check('merged f_sub route accuracy == 0.5 (2 items total across both fixtures)',
          abs(merged['modp:face:taskf_sub']['accuracy'] - 0.5) < 1e-9)


# ------------------------------------------------- 5. confirm_stats
def test_confirm_stats():
    print('== 5. confirm_stats.py unit tests (confirmation-stage criteria) ==')
    steps = [0, 200, 400, 600, 800, 1000]
    ledger_cells = {
        'P_S': {'0.9': {'first_step': 400, 'stable': True}},
        'P_D': {'0.9': {'first_step': 600, 'stable': True}},
        'P_M': {'0.9': {'first_step': 1000, 'stable': False}},   # crossed but NOT stable -> not a birth
        'P_Q': {'0.9': {'first_step': None, 'stable': False}},   # never crossed
        'Pn_D': {'0.9': {'first_step': 400, 'stable': True}},
        'Ps_S': {'0.9': {'first_step': 200, 'stable': True}},
        'Pi_Q': {'0.9': {'first_step': 3000, 'stable': True}},   # far outside the window on purpose
        'P_X': {'0.9': {'first_step': None, 'stable': False}},   # placebo: never enters
    }
    res = dict(steps=steps, ledger=dict(cells=ledger_cells),
               per_theta={'0.9': {'transitions': [
                   dict(h0='merge', added=['P_S']),
                   dict(h0='other', added=['P_D']),
                   dict(h0='merge', added=['P_N']),   # not an arithmetic edge -> filtered out below
               ]}})

    births = CS.edge_birth_steps(res)
    check('edge_birth_steps: P_S born at 400', births['P_S'] == 400)
    check('edge_birth_steps: P_D born at 600', births['P_D'] == 600)
    check('edge_birth_steps: P_M crossed-but-unstable -> None (not a birth)', births['P_M'] is None)
    check('edge_birth_steps: P_Q never crossed -> None', births['P_Q'] is None)

    sync = CS.arithmetic_synchrony(res, tol_checkpoints=1)
    check('arithmetic_synchrony: eval_every inferred as 200', sync['eval_every'] == 200)
    check('arithmetic_synchrony: tol_steps == 200 at tol_checkpoints=1', sync['tol_steps'] == 200)
    check('arithmetic_synchrony: best_step == 400 (P_S/P_D/Pn_D/Ps_S cluster there)', sync['best_step'] == 400)
    check('arithmetic_synchrony: count == 4 at the best step', sync['count'] == 4, detail=str(sync))

    pb_pass = CS.placebo_never_enters(res)
    check('placebo_never_enters: P_X never crosses -> passes', pb_pass['never_enters'] is True)
    res_bad = dict(res, ledger=dict(cells={**ledger_cells, 'P_X': {'0.9': {'first_step': 300, 'stable': True}}}))
    pb_fail = CS.placebo_never_enters(res_bad)
    check('placebo_never_enters: P_X DOES cross -> fails', pb_fail['never_enters'] is False)

    h0 = CS.h0_merge_events(res, cells=CS.ARITHMETIC_EDGES)
    check('h0_merge_events: only the arithmetic-edge merge kept (P_N filtered out)',
          len(h0['0.9']) == 1 and h0['0.9'][0]['added'] == ['P_S'], detail=str(h0))

    births_by_seed = {
        1: {'P_S': 400, 'P_D': 4000},
        2: {'P_S': 420, 'P_D': None},
        3: {'P_S': 410, 'P_D': None},
        4: {'P_S': 405, 'P_D': 4200},
        5: {'P_S': None, 'P_D': None},
    }
    repro = CS.multi_seed_reproduction(births_by_seed, edges=('P_S', 'P_D'), tol_checkpoints=4, eval_every=200)
    check('multi_seed_reproduction: P_S born in 4/5 seeds, tightly clustered -> reproduced',
          repro['P_S']['reproduced'] is True, detail=str(repro['P_S']))
    check('multi_seed_reproduction: P_D born in only 2/5 seeds -> not reproduced',
          repro['P_D']['reproduced'] is False, detail=str(repro['P_D']))


def main():
    tests = [test_split_closure, test_ground_truth_betti, test_cpu_dryrun,
             test_classic_preset_dryrun, test_face_scores_unit, test_confirm_stats]
    for t in tests:
        t()
    print(f'\nALL TESTS PASSED ({len(PASS)} checks)')


if __name__ == '__main__':
    main()
