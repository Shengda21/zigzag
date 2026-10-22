"""Synthetic end-to-end test of zz_analyze on fabricated eval files built from
the real world-501 PROBES rows. Scenario A: every zh cell passes at the same
checkpoint (synchronous); scenario B: zh edges pass one at a time (gradual)."""
from __future__ import annotations
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / 'stage_ab' / 'code'))
import zz_analyze as A     # noqa: E402
import zz_score as Z       # noqa: E402

PROBES = HERE.parent / 'data' / 'world501' / 'PROBES.jsonl'
EDGE_ORDER = list(Z.ATOMIC_EDGES)


def correct_at(r, t, scenario):
    lang, kind, task = r['lang'], r['kind'], r.get('task')
    if r['vocabulary'] == 'general':
        return t >= 1
    if lang == 'en':
        return t >= 2 if kind != 'joint' else t >= 3
    if lang == 'zh':
        if kind == 'base':
            if scenario == 'A':
                return t >= 5
            if task in EDGE_ORDER:
                return t >= 3 + EDGE_ORDER.index(task)
            return t >= 3
        if kind == 'joint':
            return t >= 7 if scenario == 'A' else t >= 9
        return t >= 8
    return t >= 6        # ru / ja transfer


def fabricate(root, scenario, T=10):
    rows = [json.loads(s) for s in PROBES.read_text(encoding='utf8').splitlines() if s.strip()]
    run = root / f'world501_{scenario}'
    run.mkdir(parents=True, exist_ok=True)
    for t in range(T):
        step = 16 * t
        with (run / f'eval_{step:04d}.jsonl').open('w', encoding='utf8') as f:
            for r in rows:
                pred = r['gold'] if correct_at(r, t, scenario) else 'X'
                f.write(json.dumps(dict(r, prediction=pred, step=step, **Z.score(r, pred)), ensure_ascii=False) + '\n')
    (run / 'COMPLETE.json').write_text('{}')
    return run


def main():
    root = Path(os.environ.get('ZZ_DRYRUN_DIR', Path(os.environ.get('TEMP', '/tmp')) / 'zz_dryrun')) / 'analyze_test'
    if root.exists():
        shutil.rmtree(root)
    ra = fabricate(root, 'A')
    rb = fabricate(root, 'B')
    res_a = A.analyze_run(ra, 'crosslingual', n_perm=60, seed=1)
    res_b = A.analyze_run(rb, 'crosslingual', n_perm=60, seed=1)
    A.write_outputs(res_a, root / 'out_A')
    A.write_outputs(res_b, root / 'out_B')
    sa, sb = res_a['synchrony'], res_b['synchrony']
    assert sa['verdict'] == 'synchronous' and sa['t_star_steps'] == [64, 80], sa
    assert sb['verdict'] != 'synchronous', sb
    # topology, primary product complex: en triangles open at step 32 (edges before faces) and are filled at 48;
    # the zh copy enters at step 80 together with its squares, i.e. contractibly -> no Betti change there.
    pt = res_a['per_theta']['0.8']
    b1 = [b[1] for b in pt['betti']]
    assert b1[2] == 2 and b1[3] == 0 and b1[5] == 0, b1
    fills = [t for t in pt['transitions'] if t['h1'] == 'clean_fill']
    assert fills and fills[0]['to_step'] == 48, fills
    # bridge-free subcomplex G_en (+) G_zh: zh vertices+edges appear at 80 as a second component (beta0 1->2) with two open
    # triangles (beta1 2), filled by the zh faces at 112 (clean_fill)
    bf = pt['bridge_free']
    b0bf = [b[0] for b in bf['betti']]
    b1bf = [b[1] for b in bf['betti']]
    assert b0bf[4] == 1 and b0bf[5] == 2 and b1bf[5] == 2 and b1bf[7] == 0, (b0bf, b1bf)
    bfills = [t for t in bf['transitions'] if t['h1'] == 'clean_fill']
    assert any(b['to_step'] == 112 and b['added'] == ['fA@zh', 'fB@zh'] for b in bfills), bfills
    assert res_a['transfer_windows'] is not None and res_a['transfer_windows']['event_delta']['ru_edges'] > 0
    pa = res_a['null_model']
    print(json.dumps(dict(status='ZZ_ANALYZE_TEST_PASSED', A=dict(verdict=sa['verdict'], t_star=sa['t_star_steps'], support=sa['thresholds_supporting'],
                                                                  beta1_theta08=b1, beta0_bridge_free=b0bf, beta1_bridge_free=b1bf, null_keys=sorted(pa)[:6]),
                          B=dict(verdict=sb['verdict'], support=sb['thresholds_supporting'], spread=sb['spread_per_theta']['0.8']))))


if __name__ == '__main__':
    main()
