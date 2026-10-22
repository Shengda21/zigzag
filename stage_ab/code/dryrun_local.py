"""Local CPU dry-run of the complete Stage A/B chain on a toy Qwen2 model.

Builds a 2-layer, 64-hidden Qwen2 with the real Qwen2.5 tokenizer/chat
template (from zigzag_v3/model_metadata), then runs queue_stage_ab.py with
--dryrun 3 --ignore-gates in a scratch directory. Checks that every stage
writes run_manifest / eval files / GATE / COMPLETE, that DEP/IND pairing
hashes agree per seed, and that resume works after a simulated interruption.
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parent
META = R.parents[1] / 'zigzag_v3' / 'model_metadata' / 'models' / 'qwen15'


def build_toy(dst: Path):
    import torch
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
    tok = AutoTokenizer.from_pretrained(META)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=2048, tie_word_embeddings=True,
                      bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, rope_theta=1e6)
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg)
    dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dst, safe_serialization=True)
    tok.save_pretrained(dst)
    return dst


def main():
    scratch = Path(os.environ.get('ZZ_DRYRUN_DIR', Path(os.environ.get('TEMP', '/tmp')) / 'zz_dryrun'))
    if scratch.exists():
        shutil.rmtree(scratch)
    (scratch / 'runs').mkdir(parents=True)
    toy = build_toy(scratch / 'toy_model')
    env = dict(os.environ, MODEL_PATH=str(toy), ZZ_RUNS=str(scratch / 'runs'))
    common = [sys.executable, '-u', str(HERE / 'run_stage_ab.py'), '--device', 'cpu', '--dryrun', '3', '--ignore-gates',
              '--runs', str(scratch / 'runs'), '--eval-batch', '8']
    order = [(401, 'COMMON'), (401, 'FORMAT'), (401, 'DEP1'), (401, 'IND1')]
    for w, stage in order:
        r = subprocess.run(common + ['--world', str(w), '--stage', stage], env=env, capture_output=True, text=True, encoding='utf8', errors='replace')
        print(f'--- {w} {stage} exit={r.returncode}')
        print('\n'.join(r.stdout.splitlines()[-4:]))
        if r.returncode:
            print(r.stderr[-3000:])
            raise SystemExit(f'{stage} failed')
        out = scratch / 'runs' / f'world{w}_{stage}'
        for f in ['run_manifest.json', 'eval_0000.jsonl', 'eval_0000.summary.json', 'COMPLETE.json', 'final/adapter_model.safetensors']:
            assert (out / f).exists(), f
    # pairing identical across DEP1/IND1
    d = json.loads((scratch / 'runs/world401_DEP1/PAIRING.json').read_text(encoding='utf8'))
    i = json.loads((scratch / 'runs/world401_IND1/PAIRING.json').read_text(encoding='utf8'))
    assert d['permutation_sha256'] == i['permutation_sha256'] and d['replay_sha256'] == i['replay_sha256'] and d['slots_sha256'] == i['slots_sha256']
    # resume: delete COMPLETE + final and the last eval, keep resume.pt, rerun -> must continue, not restart
    out = scratch / 'runs/world401_IND1'
    (out / 'COMPLETE.json').unlink()
    shutil.rmtree(out / 'final')
    for f in out.glob('eval_0003.*'):
        f.unlink()
    r = subprocess.run(common + ['--world', '401', '--stage', 'IND1'], env=env, capture_output=True, text=True, encoding='utf8', errors='replace')
    assert r.returncode == 0 and 'RESUMED' in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]
    assert (out / 'COMPLETE.json').exists()
    # independent CPU re-score of every eval file equals the runner's summary
    sys.path.insert(0, str(HERE))
    import zz_score as Z
    checked = 0
    for f in (scratch / 'runs').glob('*/eval_*.jsonl'):
        recs, summ = Z.score_file(f)
        ref = json.loads(f.with_suffix('.summary.json').read_text(encoding='utf8'))['scores']
        assert summ == ref, f
        checked += 1
    print(json.dumps(dict(status='DRYRUN_PASSED', stages=[f'{w}_{s}' for w, s in order], eval_files_rescored=checked, scratch=str(scratch))))


if __name__ == '__main__':
    main()
