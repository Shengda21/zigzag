"""Train a TinyTransformer from scratch on one world of a modular-arithmetic
spec, mixing all taught edges/faces uniformly by task, and periodically
evaluate every probe (writing eval_{step:05d}.jsonl(.gz) + .summary.json that
zz_analyze.analyze_run(..., '<COMPLEX.json>') can read directly).

python run_grok.py --spec MODP --world 701 --seed 1
python run_grok.py --spec MODP_ADDONLY --world 703 --seed 1 --preset classic   (classic full-batch setup)
python run_grok.py --spec MODP --world 701 --seed 1 --dryrun --dryrun-p 13     (CPU smoke test)
python run_grok.py --spec MODP --world 701 --seed 1 --resume                   (continue from resume.pt)
"""
from __future__ import annotations
import argparse
import collections
import gzip
import json
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
GROK_DIR = HERE.parent
ROOT = GROK_DIR.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'stage_ab' / 'code'))
sys.path.insert(0, str(ROOT / 'crosslingual' / 'code'))
import modp_common as M            # noqa: E402
import make_world as MW            # noqa: E402
import zz_score as Z               # noqa: E402
import face_scores as FS           # noqa: E402
from tiny_transformer import TinyTransformer, TinyConfig, count_params  # noqa: E402


def load_manifest_spec(spec_name, world):
    spec = json.loads((GROK_DIR / 'specs' / f'{spec_name}.json').read_text(encoding='utf8'))
    return spec


def build_batches(train_rows, vocab, batch_size, device, rng):
    """One infinite generator: each call to next() draws `batch_size`
    examples by first choosing a taught task name uniformly, then a row
    from that task's pool uniformly (with replacement across steps)."""
    tasks = list(train_rows.keys())
    pools = train_rows
    max_len = M.MAX_LEN
    while True:
        ids_batch = torch.zeros(batch_size, max_len, dtype=torch.long)
        prompt_lens = torch.zeros(batch_size, dtype=torch.long)
        answer_lens = torch.zeros(batch_size, dtype=torch.long)
        for i in range(batch_size):
            task = rng.choice(tasks)
            row = rng.choice(pools[task])
            ids, pl, al = M.encode_row(row, vocab, max_len)
            ids_batch[i] = torch.tensor(ids, dtype=torch.long)
            prompt_lens[i] = pl
            answer_lens[i] = al
        yield ids_batch.to(device), prompt_lens.to(device), answer_lens.to(device)


FULL_BATCH_CAP = 8192   # --preset classic uses a true full batch up to this many rows total
                         # (see README "最小规格与经典设置"); above it, falls back to batch_size=2048.


def make_full_batch(train_rows, vocab, device):
    """Encode every training row (across every taught task) once into a
    single fixed batch -- true full-batch gradient descent, the setting the
    grokking literature (Power et al. 2022; Nanda et al. 2023) actually
    uses. Only practical for a small pool (see FULL_BATCH_CAP); the exact
    same tensors are reused unchanged at every step, so this is a one-time
    cost, not a per-step one."""
    all_rows = [r for rows in train_rows.values() for r in rows]
    max_len = M.MAX_LEN
    ids = torch.zeros(len(all_rows), max_len, dtype=torch.long)
    prompt_lens = torch.zeros(len(all_rows), dtype=torch.long)
    answer_lens = torch.zeros(len(all_rows), dtype=torch.long)
    for i, r in enumerate(all_rows):
        e_ids, pl, al = M.encode_row(r, vocab, max_len)
        ids[i] = torch.tensor(e_ids, dtype=torch.long)
        prompt_lens[i] = pl
        answer_lens[i] = al
    return ids.to(device), prompt_lens.to(device), answer_lens.to(device)


def loss_for_batch(model, ids_batch, prompt_lens, answer_lens):
    logits = model(ids_batch)                      # (B,T,V)
    B, T, V = logits.shape
    targets = ids_batch[:, 1:]                      # predict next token
    logp = logits[:, :-1, :]                        # (B,T-1,V)
    pos = torch.arange(T - 1, device=ids_batch.device)[None, :].expand(B, -1)
    mask = (pos >= (prompt_lens - 1)[:, None]) & (pos < (prompt_lens + answer_lens - 1)[:, None])
    loss = torch.nn.functional.cross_entropy(logp.reshape(-1, V), targets.reshape(-1), reduction='none')
    loss = (loss.reshape(B, T - 1) * mask).sum() / mask.sum().clamp(min=1)
    with torch.no_grad():
        pred = logp.argmax(-1)
        correct_tok = ((pred == targets) & mask).sum().item()
        n_tok = mask.sum().item()
    return loss, correct_tok, n_tok


@torch.no_grad()
def evaluate(model, probes, vocab, id_to_tok, device, batch_size=1024):
    """Group probes by (prompt_len, answer_len) so every batch needs no
    padding at all, greedily decode the answer, score with zz_score, and
    return the scored rows (each carries 'prediction' + score fields)."""
    groups = collections.defaultdict(list)
    for r in probes:
        ids, pl, al = M.encode_row(r, vocab, M.MAX_LEN)
        groups[(pl, al)].append((r, ids[:pl]))
    out = []
    for (pl, al), items in groups.items():
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            prompt = torch.tensor([ids for _, ids in chunk], dtype=torch.long, device=device)
            ans = model.generate(prompt, al)   # (N, al)
            ans_cpu = ans.cpu().tolist()
            for (row, _), a_ids in zip(chunk, ans_cpu):
                pred = M.render_prediction(a_ids, id_to_tok)
                sc = Z.score(row, pred)
                out.append(dict(row, prediction=pred, **sc))
    return out


def make_config(vocab_size, layers, d_model=128, n_heads=4, d_ff=512, max_len=M.MAX_LEN):
    return TinyConfig(vocab_size=vocab_size, max_len=max_len, d_model=d_model, n_layers=layers,
                       n_heads=n_heads, d_ff=d_ff, dropout=0.0)


# --preset classic: the setting the grokking literature (Power et al. 2022;
# Nanda et al. 2023) actually uses -- 1 layer, full-batch AdamW, wd=1.0, long
# training. Meant to be paired with specs/MODP_ADDONLY.json (a single
# operation, so "full batch" is a genuinely small, GPU-cheap tensor) but
# works with any spec; see FULL_BATCH_CAP for the fallback on a larger one.
CLASSIC_PRESET = dict(layers=1, d_model=128, n_heads=4, d_ff=512, wd=1.0, lr=1e-3,
                       steps=50000, eval_every=200, ckpt_every=5000)


def run(args):
    spec = load_manifest_spec(args.spec, args.world)
    if args.dryrun_p:
        spec = json.loads(json.dumps(spec))
        spec['primes'] = {str(args.world): args.dryrun_p}
        spec['probe'] = dict(edge_roots=32, vertex_roots=16, face_roots=16, transfer_roots=16)
    if args.frac is not None:
        spec['split_frac'] = args.frac
    classic = args.preset == 'classic'

    def resolved(name, dryrun_val, classic_val, spec_key):
        cli = getattr(args, name)
        if cli is not None:
            return cli
        if args.dryrun:
            return dryrun_val
        if classic:
            return classic_val
        return spec[spec_key]
    steps = resolved('steps', 300, CLASSIC_PRESET['steps'], 'steps')
    eval_every = resolved('eval_every', 50, CLASSIC_PRESET['eval_every'], 'eval_every')
    ckpt_every = resolved('ckpt_every', 150, CLASSIC_PRESET['ckpt_every'], 'ckpt_every')
    layers = args.layers if args.layers is not None else (CLASSIC_PRESET['layers'] if classic else 2)
    wd = args.wd if args.wd is not None else (CLASSIC_PRESET['wd'] if classic else 1.0)
    lr = args.lr if args.lr is not None else (CLASSIC_PRESET['lr'] if classic else 1e-3)
    d_model = args.d_model if args.d_model is not None else CLASSIC_PRESET['d_model']
    n_heads = args.n_heads if args.n_heads is not None else CLASSIC_PRESET['n_heads']
    d_ff = args.d_ff if args.d_ff is not None else CLASSIC_PRESET['d_ff']

    only_tasks = set(args.only_tasks.split(',')) if args.only_tasks else None
    dat = MW.build_data(spec, args.world, only_tasks=only_tasks)
    p = dat['p']
    vocab = M.build_vocab(p)
    id_to_tok = {i: t for t, i in vocab.items()}
    probes = dat['probes']
    if only_tasks:
        probes = [r for r in probes if r['task'] in only_tasks or r['kind'] == 'transfer']

    # batch size: explicit --batch-size always wins; --preset classic asks for a true
    # full batch (every training row, every step) but falls back to batch_size=2048
    # if the pool is too large for that to be GPU-practical (see FULL_BATCH_CAP).
    full_batch = False
    total_pool = sum(len(rows) for rows in dat['train_rows'].values())
    if args.batch_size is not None:
        batch_size = args.batch_size
    elif classic:
        if total_pool <= FULL_BATCH_CAP:
            batch_size = total_pool
            full_batch = True
        else:
            batch_size = 2048
    else:
        batch_size = 512

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    torch.manual_seed(args.seed)
    cfg = make_config(len(vocab), layers, d_model=d_model, n_heads=n_heads, d_ff=d_ff)
    model = TinyTransformer(cfg).to(device)

    out_dir = Path(args.out) if args.out else GROK_DIR / 'runs' / f'{args.spec}_world{args.world}_seed{args.seed}'
    out_dir.mkdir(parents=True, exist_ok=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=wd)
    start_step = 0
    resume_path = out_dir / 'resume.pt'
    if args.resume and resume_path.exists():
        ck = torch.load(resume_path, map_location=device)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        start_step = ck['step']
        print(f'resumed from step {start_step}')

    if full_batch:
        full_ids, full_pl, full_al = make_full_batch(dat['train_rows'], vocab, device)

        def next_batch():
            return full_ids, full_pl, full_al
    else:
        rng = random.Random(args.seed)
        batches = build_batches(dat['train_rows'], vocab, batch_size, device, rng)

        def next_batch():
            return next(batches)

    full_batch_note = None
    if full_batch:
        full_batch_note = f'true full batch: {total_pool} rows/step'
    elif classic:
        full_batch_note = f'pool {total_pool} > FULL_BATCH_CAP {FULL_BATCH_CAP}; fell back to batch_size={batch_size}'

    (out_dir / 'RUN_CONFIG.json').write_text(json.dumps(dict(
        spec=args.spec, world=args.world, p=p, seed=args.seed, steps=steps, eval_every=eval_every,
        ckpt_every=ckpt_every, wd=wd, lr=lr, layers=layers, batch_size=batch_size, full_batch=full_batch,
        total_train_pool=total_pool, preset=args.preset, full_batch_note=full_batch_note,
        d_model=d_model, n_heads=n_heads, d_ff=d_ff, frac=spec['split_frac'],
        only_tasks=sorted(only_tasks) if only_tasks else None, n_params=count_params(model),
        vocab_size=len(vocab), device=str(device)), ensure_ascii=False, indent=1), encoding='utf8')

    train_log_path = out_dir / 'TRAIN_LOG.jsonl'
    train_log = open(train_log_path, 'a', encoding='utf8')

    t0 = time.time()
    for step in range(start_step, steps):
        model.train()
        ids_batch, prompt_lens, answer_lens = next_batch()
        loss, correct_tok, n_tok = loss_for_batch(model, ids_batch, prompt_lens, answer_lens)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % eval_every == 0 or step == steps - 1:
            model.eval()
            scored = evaluate(model, probes, vocab, id_to_tok, device)
            eval_path = out_dir / f'eval_{step:05d}.jsonl.gz'      # slim (no prompt) + gzip: ~19.7k rows x 300 checkpoints x 20 runs must fit the disk
            with gzip.open(eval_path, 'wt', encoding='utf8') as fh:
                for r in scored:
                    fh.write(json.dumps({k: v for k, v in r.items() if k != 'prompt'}, ensure_ascii=False, separators=(',', ':')) + '\n')
            summary = Z.summarize(scored)
            summary = FS.merge_face_cross_check(summary, scored)
            (out_dir / f'eval_{step:05d}.summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf8')
            train_acc = correct_tok / max(1, n_tok)
            elapsed = time.time() - t0
            print(f'step={step} loss={loss.item():.4f} train_tok_acc={train_acc:.3f} '
                  f'n_probe={len(scored)} elapsed={elapsed:.1f}s')
            train_log.write(json.dumps(dict(step=step, loss=loss.item(), train_tok_acc=train_acc,
                                             n_tok=n_tok, elapsed_s=elapsed)) + "\n")
            train_log.flush()

        if step % ckpt_every == 0 or step == steps - 1:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step + 1),
                       out_dir / f'ckpt_{step:05d}.pt')
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step + 1), resume_path)

    train_log.close()
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', default='MODP')
    ap.add_argument('--world', type=int, required=True)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--eval-every', type=int, default=None)
    ap.add_argument('--ckpt-every', type=int, default=None)
    ap.add_argument('--wd', type=float, default=None)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--frac', type=float, default=None, help='override spec split_frac (in-memory only)')
    ap.add_argument('--layers', type=int, default=None)
    ap.add_argument('--d-model', type=int, default=None)
    ap.add_argument('--n-heads', type=int, default=None)
    ap.add_argument('--d-ff', type=int, default=None)
    ap.add_argument('--batch-size', type=int, default=None)
    ap.add_argument('--preset', choices=['classic'], default=None,
                     help='classic: 1 layer, full-batch AdamW, wd=1.0, 50000 steps (literature setting; '
                          'see README "最小规格与经典设置"). Any explicit flag above overrides its value.')
    ap.add_argument('--only-tasks', default=None, help='comma-separated task names to restrict training+eval to')
    ap.add_argument('--out', default=None)
    ap.add_argument('--dryrun', action='store_true')
    ap.add_argument('--dryrun-p', type=int, default=None, help='small prime for CPU smoke tests')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args()
    if a.dryrun and a.dryrun_p is None:
        a.dryrun_p = 13
    run(a)


if __name__ == '__main__':
    main()
