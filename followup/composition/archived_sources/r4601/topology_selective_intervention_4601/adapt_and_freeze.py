"""Prepare the R4601 runner and seal its code/data before any model query."""
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'reversible_intervention_4103'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def replace(source, target, edits):
    text = (PRIOR / source).read_text()
    for old, new in edits:
        assert text.count(old) == 1, (source, old, text.count(old))
        text = text.replace(old, new)
    (HERE / target).write_text(text)


def main():
    for name in ('topology_bank.jsonl', 'hard_bank.jsonl'):
        shutil.copyfile(PRIOR / name, HERE / name)
    replace('train.py', 'train.py', [
        ("choices=('corrupt', 'recover', 'clean1', 'clean2')", "choices=('cycle', 'acyclic')"),
        ("    gate = json.loads((HERE / 'BASELINE_GATE.json').read_text())\n    assert gate['status'] == 'PASS_FRESH_R4103_BASELINE_BEFORE_INTERVENTION'\n    assert gate['bank_manifest_sha256'] == sha(HERE / 'manifest.json')\n", ""),
        ("    source = (\n        ROOT / 'prospective_support_3803' / 'results_control' / 'adapter_0128'\n        if args.arm in ('corrupt', 'clean1') else\n        HERE / 'results_corrupt' / 'adapter_0128'\n        if args.arm == 'recover' else\n        HERE / 'results_clean1' / 'adapter_0128')",
         "    source = ROOT / 'prospective_support_3803' / 'results_control' / 'adapter_0128'"),
        ("    if args.arm in ('corrupt', 'clean1'):\n        baseline_sha = json.loads((HERE / 'probe_baseline' / 'runtime.json').read_text())['adapter_sha256']\n        assert source_sha == baseline_sha\n", ""),
        ("    bank_name = 'broken_train.jsonl' if args.arm == 'corrupt' else 'clean_train.jsonl'",
         "    bank_name = f'train_{args.arm}.jsonl'"),
        ("seed = 410320261114", "seed = 460120261114"),
        ("    source_sha = sha(source / 'adapter_model.safetensors')",
         "    source_sha = sha(source / 'adapter_model.safetensors')\n    assert source_sha == manifest['source_adapter_sha256']"),
        ("'status': 'COMPLETE_R3803_TRAINING'", "'status': 'COMPLETE_R4601_TRAINING'"),
    ])
    replace('evaluate.py', 'evaluate.py', [
        ("choices=('baseline', 'corrupt', 'recover',\n                                        'clean1', 'clean2')",
         "choices=('baseline', 'cycle', 'acyclic')"),
        ("'status': 'COMPLETE_R3803_PROBE'", "'status': 'COMPLETE_R4601_PROBE'"),
    ])
    replace('analyze.py', 'analyze.py', [
        ("choices=('baseline', 'corrupt', 'recover',\n                                        'clean1', 'clean2')",
         "choices=('baseline', 'cycle', 'acyclic')"),
        ("'COMPLETE_R3803_PROBE'", "'COMPLETE_R4601_PROBE'"),
        ("'status': 'COMPLETE_R3803_INDEPENDENT_ANALYSIS'",
         "'status': 'COMPLETE_R4601_INDEPENDENT_ANALYSIS'"),
        ("tok = AutoTokenizer.from_pretrained(BASE / 'implicit_operator_trajectory_3101',",
         "tok = AutoTokenizer.from_pretrained('/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct',"),
    ])
    analysis_path = HERE / 'analyze.py'
    analysis = analysis_path.read_text()
    old = "    assert runtime['status'] == 'COMPLETE_R4601_PROBE'"
    new = "    assert runtime['status'] == ('COMPLETE_R3803_PROBE' if args.arm == 'baseline' else 'COMPLETE_R4601_PROBE')\n    if args.arm == 'baseline':\n        assert runtime['adapter_sha256'] == frozen['source_adapter_sha256']"
    assert analysis.count(old) == 1
    analysis_path.write_text(analysis.replace(old, new))
    gate = json.loads((HERE / 'cpu_gate.json').read_text())
    assert gate['status'] == 'PASS_MODEL_BLIND_EQUAL_DOSE_BANKS'
    assert {name: sha(HERE / name) for name in ('topology_bank.jsonl', 'hard_bank.jsonl')} == gate['eval_banks']
    names = ('PROTOCOL.md', 'build.py', 'adapt_and_freeze.py', 'cpu_gate.json',
             'train_clean.jsonl', 'train_cycle.jsonl', 'train_acyclic.jsonl',
             'topology_bank.jsonl', 'hard_bank.jsonl',
             'train.py', 'evaluate.py', 'analyze.py')
    manifest = {'status': 'FROZEN_BEFORE_R4601_MODEL_QUERY',
                'files': {name: sha(HERE / name) for name in names},
                'source_adapter_sha256': json.loads((PRIOR / 'probe_baseline' /
                                                     'runtime.json').read_text())
                                                     ['adapter_sha256'],
                'reference_banks': gate['eval_banks']}
    (HERE / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'status': manifest['status'],
                      'manifest_sha256': sha(HERE / 'manifest.json'),
                      'files': len(names)}))


if __name__ == '__main__':
    main()
