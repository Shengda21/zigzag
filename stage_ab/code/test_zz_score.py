"""Hand-constructed predictions exercise every branch of zz_score; no model."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import zz_score as Z


def row(gold, schema, fl=2, **kw):
    return dict(id='x', vocabulary='target', kind='joint', gold=gold, schema=schema, field_length=fl, **kw)


def main():
    r = row('D122,D119|P102,P110|P102,P110|1', ['D', 'P', 'P', 'binary'], template=0, condition='DEPENDENT', domain='A')
    assert Z.score(r, r['gold'])['correct'] == 1
    s = Z.score(r, 'D122, D119 | P102,P110|P102,P110|1')
    assert s['correct'] == 1 and s['valid_format'] == 1
    s = Z.score(r, 'D122,D119|P102,P110|P102,P110')
    assert s['correct'] == 0 and s['missing_fields'] == 1 and s['field_correct'] == [1, 1, 1, 0]
    s = Z.score(r, 'D122,D119|P102,P110|P102,P110|1|extra')
    assert s['valid_format'] == 0 and s['extra_fields'] == 1
    s = Z.score(r, 'D122,D119|P102,P110|P102,P110|2')
    assert s['valid_format'] == 0 and s['field_correct'][3] == 0
    s = Z.score(r, 'D122,D119|P102|P102,P110|1')
    assert s['valid_format'] == 0 and s['field_correct'] == [1, 0, 1, 1]
    s = Z.score(r, 'D122,D119|P102,P110|P102,P111|1')
    assert s['valid_format'] == 1 and s['correct'] == 0
    rep = row('D122,D119|P102,P110|0', ['D', 'P', 'step'], template=1, domain='B')
    assert Z.score(rep, 'D122,D119|P102,P110|0')['correct'] == 1
    assert Z.score(rep, 'D122,D119|P102,P110|3')['valid_format'] == 0
    base = dict(id='b', vocabulary='target', kind='base', task='eFD', template=3, gold='D001,D002', schema=['D'], field_length=2)
    assert Z.score(base, 'D001,D002')['correct'] == 1 and Z.score(base, 'D001,D002,D003')['valid_format'] == 0
    mc = dict(id='m', vocabulary='general', kind='belebele', lang='zh', template=0, gold='C', schema=['letter'], field_length=1)
    assert Z.score(mc, 'C')['correct'] == 1 and Z.score(mc, 'E')['valid_format'] == 0 and Z.score(mc, 'c')['correct'] == 0
    tx = dict(id='t', vocabulary='general', kind='xquad', lang='en', template=0, gold='New York City', schema=['text'], field_length=1)
    assert Z.score(tx, 'new york city.')['correct'] == 1 and Z.score(tx, '')['valid_format'] == 0
    ex = dict(id='e', vocabulary='public', kind='edge', task='upper', template=0, gold='HELLO', schema=['exact'], field_length=1)
    assert Z.score(ex, 'HELLO')['correct'] == 1 and Z.score(ex, ' HELLO ')['correct'] == 1 and Z.score(ex, 'hello')['correct'] == 0
    keys = Z.group_keys(r)
    assert 'target:joint:all' in keys and 'target:joint:template0:conditionDEPENDENT' in keys and len(keys) == 1 + 3 + 3
    # Gate arithmetic on constructed summaries.
    recs = []
    for t in (0, 1, 3):
        for e in Z.ATOMIC_EDGES:
            for i in range(10):
                b = dict(base, task=e, template=t)
                recs.append(dict(b, **Z.score(b, b['gold'] if i < 9 else 'D999,D999')))
    summ = Z.summarize(recs)
    g = Z.atomic_gate(summ)
    assert g['passed'] and abs(g['min'] - 0.9) < 1e-9
    recs2 = [dict(b, **Z.score(b, 'D999,D999')) for b in [dict(base, task='eRP', template=3)] * 2]
    g2 = Z.atomic_gate(Z.summarize(recs + recs2))
    assert not g2['passed']
    fmt = dict(id='f', vocabulary='aux', kind='format', template=0, gold='D001,D002|P001,P002|P002,P001|0', schema=['D', 'P', 'P', 'binary'], field_length=2)
    frecs = []
    for t in (0, 1, 2):
        for i in range(20):
            f = dict(fmt, template=t)
            frecs.append(dict(f, **Z.score(f, f['gold'] if i < 19 else 'bad')))
    fs = Z.summarize(frecs)
    a = Z.stage_a_gate(fs, summ)
    assert a['parse']['template0'] == 0.95 and a['checks']['parse_each_template'] and a['checks']['strict_copy'] and a['checks']['missing_or_extra']
    assert a['passed']
    b = Z.stage_b_gate(Z.summarize(recs))
    assert not b['passed'] and b['checks']['atomic'] and not b['checks']['joint']
    print(json.dumps(dict(status='ZZ_SCORE_TESTS_PASSED', atomic_min=g['min'], stage_a=a['checks'], stage_b=b['checks'])))


if __name__ == '__main__':
    main()
