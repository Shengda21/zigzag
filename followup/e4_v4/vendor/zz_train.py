"""Training / evaluation primitives shared by the Stage A/B and cross-lingual
runners. Numerics follow reproduction/runner_d1.py (per-example mean token
loss averaged over the effective batch, LoRA r16/alpha32 on all projections,
AdamW 2e-4, 20-step warm-up then cosine to 0.1, gradient clip 1.0).

Device-agnostic: the same code path runs on a CPU tiny model for the local
dry-run and on the H800 for the real stages.
"""
from __future__ import annotations
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

import zz_score as Z

LORA = dict(r=16, lora_alpha=32, lora_dropout=.05, bias='none', task_type='CAUSAL_LM',
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])


# ----------------------------------------------------------------------------- io
def dump(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + '.tmp')
    q.write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding='utf8')
    q.replace(p)


def rows(p):
    return [json.loads(s) for s in Path(p).read_text(encoding='utf8').splitlines() if s.strip()]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def seed(n):
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(n)


def rngget():
    s = dict(py=random.getstate(), np=np.random.get_state(), torch=torch.get_rng_state())
    if torch.cuda.is_available():
        s['cuda'] = torch.cuda.get_rng_state_all()
    return s


def rngset(x):
    random.setstate(x['py'])
    np.random.set_state(x['np'])
    torch.set_rng_state(x['torch'])
    if torch.cuda.is_available() and 'cuda' in x:
        torch.cuda.set_rng_state_all(x['cuda'])


# -------------------------------------------------------------------------- model
def load(modelpath, device):
    tok = AutoTokenizer.from_pretrained(modelpath)
    tok.pad_token = tok.eos_token
    tok.padding_side = 'right'
    cuda = device.startswith('cuda')
    dt = torch.bfloat16 if cuda else torch.float32
    attn = 'sdpa' if cuda else 'eager'
    try:                                   # transformers >= 5 uses dtype=, 4.x uses torch_dtype=
        model = AutoModelForCausalLM.from_pretrained(modelpath, torch_dtype=dt, attn_implementation=attn)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(modelpath, dtype=dt, attn_implementation=attn)
    model = model.to(device)
    assert next(model.parameters()).dtype == dt, ('model dtype', next(model.parameters()).dtype)
    model = get_peft_model(model, LoraConfig(**LORA))
    model.config.use_cache = False
    return tok, model


def load_adapter(model, adapter_dir):
    cp = Path(adapter_dir) / 'adapter_model.safetensors'
    set_peft_model_state_dict(model, load_file(str(cp)))
    return sha(cp)


def raw_doc(r):
    """Continued-pretraining document for a record: prompt, newline, answer, newline."""
    return f"{r['prompt']}\n{r['gold']}\n"


def encoded(tok, rs, max_len=512, max_answer=96, fmt='chat'):
    if fmt == 'raw':                     # loss on every token of the document (no chat template, no prompt masking)
        out = []
        for r in rs:
            ids = tok(raw_doc(r), add_special_tokens=False)['input_ids'] + [tok.eos_token_id]
            assert len(ids) <= max_len, (r['id'], len(ids))
            out.append((ids, list(ids)))
        return out
    pp = [tok.apply_chat_template([dict(role='user', content=r['prompt'])], tokenize=False, add_generation_prompt=True) for r in rs]
    ff = [tok.apply_chat_template([dict(role='user', content=r['prompt']), dict(role='assistant', content=r['gold'])], tokenize=False) for r in rs]
    pids = tok(pp, add_special_tokens=False)['input_ids']
    fids = tok(ff, add_special_tokens=False)['input_ids']
    out = []
    for r, p, f in zip(rs, pids, fids):
        assert f[:len(p)] == p, r['id']
        end = f.index(tok.eos_token_id, len(p)) + 1
        f = f[:end]
        assert len(f) <= max_len and end - len(p) <= max_answer, (r['id'], len(f), end - len(p))
        out.append((f, [-100] * len(p) + f[len(p):]))
    return out


def batch_tensors(xs, tok, device):
    n = max(len(x[0]) for x in xs)
    a, b, c = [], [], []
    for ids, lab in xs:
        z = n - len(ids)
        a.append(ids + [tok.pad_token_id] * z)
        b.append(lab + [-100] * z)
        c.append([1] * len(ids) + [0] * z)
    return {k: torch.tensor(v, device=device) for k, v in [('input_ids', a), ('labels', b), ('attention_mask', c)]}


def generate(model, tok, prompts, batchsize, device, max_new_tokens=72, fmt='chat'):
    answers = []
    off = 0
    size = batchsize
    with torch.inference_mode():
        while off < len(prompts):
            chunk = prompts[off:off + size]
            if fmt == 'raw':
                ss = chunk                       # already "<demos>{prompt}\n"; the continuation up to the next newline is the answer
            else:
                ss = [tok.apply_chat_template([dict(role='user', content=p)], tokenize=False, add_generation_prompt=True) for p in chunk]
            try:
                inputs = tok(ss, padding=True, return_tensors='pt', add_special_tokens=False).to(device)
                z = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None,
                                   pad_token_id=tok.pad_token_id)
                texts = tok.batch_decode(z[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                if fmt == 'raw':
                    texts = [t.split('\n')[0] for t in texts]
                answers.extend(texts)
                off += len(chunk)
                del inputs, z
            except torch.cuda.OutOfMemoryError:
                if size <= 8:
                    raise
                size //= 2
                gc.collect()
                torch.cuda.empty_cache()
                print('EVAL_BATCH_REDUCED', size, flush=True)
    return answers


def evaluate(model, tok, rs, out, step, batchsize, device, max_new_tokens=72, fmt='chat', demos_for=None):
    """Greedy decode every row, score with zz_score, write eval_{step}.jsonl and
    its summary. Idempotent: an existing summary is returned unchanged.
    fmt='raw': prompts are few-shot documents built by demos_for(row) -> prefix text."""
    out = Path(out)
    p = out / f'eval_{step:04d}.jsonl'
    summary = p.with_suffix('.summary.json')
    if summary.exists():
        return json.loads(summary.read_text(encoding='utf8'))
    state = rngget()
    model.eval()
    model.config.use_cache = True
    tok.padding_side = 'left'
    t0 = time.time()
    rs = sorted(rs, key=lambda r: len(r['prompt']))
    if fmt == 'raw':
        texts = [(demos_for(r) if demos_for else '') + f"{r['prompt']}\n" for r in rs]
    else:
        texts = [r['prompt'] for r in rs]
    answers = generate(model, tok, texts, batchsize, device, max_new_tokens, fmt)
    recs = []
    with p.with_suffix('.partial').open('w', encoding='utf8') as f:
        for r, s in zip(rs, answers):
            rec = dict(r, prediction=s, step=step, **Z.score(r, s))
            recs.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    p.with_suffix('.partial').replace(p)
    result = dict(step=step, n=len(rs), seconds=time.time() - t0, scores=Z.summarize(recs))
    dump(summary, result)
    print('EVAL', json.dumps(dict(step=step, seconds=round(result['seconds'], 1),
                                  key_scores={k: round(v['accuracy'], 4) for k, v in result['scores'].items() if k.endswith(':all')})), flush=True)
    tok.padding_side = 'right'
    model.config.use_cache = False
    model.train()
    rngset(state)
    return result


def lr_at(s, steps, base=2e-4, warm=20):
    return base * min(s / warm, 1) * (.1 + .9 * (1 + math.cos(math.pi * max(0, s - warm) / max(1, steps - warm))) / 2)


def train_loop(model, tok, out, enc, steps, batch, microbatch, device, *, eval_rows, eval_every, save_every,
               stop_flag, seed_value, extra_eval_steps=(), eval_batch=192, max_new_tokens=72, log_every=16,
               start_step=0, opt=None, resume_target=None, keep_resume=False, fmt='chat', demos_for=None):
    """Fixed-budget LoRA fine-tuning with resumable state.

    enc: list of (ids, labels) already ordered as the training stream; step s
    uses enc[s*batch:(s+1)*batch]. Returns (final_summary, stopped_early).
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cuda = device.startswith('cuda')
    assert len(enc) >= steps * batch, (len(enc), steps, batch)
    if opt is None:
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, betas=(.9, .999),
                                weight_decay=.01, fused=cuda)
    start = start_step
    if (out / 'resume.pt').exists():
        q = torch.load(out / 'resume.pt', map_location='cpu', weights_only=False)
        assert q['target'] == steps, (q['target'], steps)
        set_peft_model_state_dict(model, q['adapter'])
        opt.load_state_dict(q['optimizer'])
        rngset(q['rng'])
        start = q['step']
        del q
        print('RESUMED', start, flush=True)

    def resume(s):
        torch.save(dict(step=s, target=steps, adapter=get_peft_model_state_dict(model), optimizer=opt.state_dict(), rng=rngget()),
                   out / 'resume.tmp')
        (out / 'resume.tmp').replace(out / 'resume.pt')

    def save(s):
        q = out / f'step_{s:04d}'
        q.mkdir(exist_ok=True)
        model.save_pretrained(q, safe_serialization=True)
        dump(q / 'state.json', dict(step=s, adapter_sha256=sha(q / 'adapter_model.safetensors')))

    if start == 0:
        save(0)
        evaluate(model, tok, eval_rows, out, 0, eval_batch, device, max_new_tokens, fmt, demos_for)
    with (out / 'train.jsonl').open('a', encoding='utf8', buffering=1) as log:
        for step in range(start, steps):
            if Path(stop_flag).exists():
                resume(step)
                print('BUDGET_STOP', step, flush=True)
                return None, True
            tick = time.time()
            model.train()
            opt.zero_grad(set_to_none=True)
            lossval = 0.
            xs = enc[step * batch:(step + 1) * batch]
            assert len(xs) == batch
            for off in range(0, batch, microbatch):
                bb = batch_tensors(xs[off:off + microbatch], tok, device)
                ctx = torch.autocast('cuda', dtype=torch.bfloat16) if cuda else torch.autocast('cpu', enabled=False)
                with ctx:
                    output = model(input_ids=bb['input_ids'], attention_mask=bb['attention_mask'])
                    ids = bb['labels'][:, 1:]
                    mask = ids != -100
                    logits = output.logits[:, :-1, :][mask].float()
                    losses = torch.nn.functional.cross_entropy(logits, ids[mask], reduction='none')
                    counts = mask.sum(1)
                    idx = torch.arange(len(counts), device=device).repeat_interleave(counts)
                    per = torch.zeros(len(counts), device=device, dtype=losses.dtype).scatter_add_(0, idx, losses) / counts
                    loss = per.sum() / batch
                assert torch.isfinite(loss)
                loss.backward()
                lossval += float(loss.detach())
                del bb, output, logits, losses, per, loss
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            s = step + 1
            lr = lr_at(s, steps)
            for g in opt.param_groups:
                g['lr'] = lr
            opt.step()
            rec = dict(step=s, loss=lossval, lr=lr, norm=float(norm), seconds=time.time() - tick)
            if cuda:
                rec['peak_memory_gb'] = torch.cuda.max_memory_allocated() / 1e9
            log.write(json.dumps(rec) + '\n')
            if s % log_every == 0:
                print('TRAIN', json.dumps(rec), flush=True)
            if s % eval_every == 0 or s in extra_eval_steps or s == steps:
                evaluate(model, tok, eval_rows, out, s, eval_batch, device, max_new_tokens, fmt, demos_for)
                resume(s)
            if s % save_every == 0 or s == steps:
                save(s)
    if not (out / f'eval_{steps:04d}.summary.json').exists():     # resumed at the last step
        evaluate(model, tok, eval_rows, out, steps, eval_batch, device, max_new_tokens, fmt, demos_for)
    final = json.loads((out / f'eval_{steps:04d}.summary.json').read_text(encoding='utf8'))
    (out / 'final').mkdir(exist_ok=True)
    model.save_pretrained(out / 'final', safe_serialization=True)
    if not keep_resume:
        (out / 'resume.pt').unlink(missing_ok=True)      # optimizer state no longer needed; frees ~110MB per arm
    return final, False
