"""Calibrate trace stopping while retaining learned long composition."""
import argparse
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
MODEL = Path('/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('stop',), required=True)
    args = parser.parse_args()
    manifest = json.loads((HERE / 'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert sha(HERE / name) == digest, name
    for name, digest in manifest['source_sha256'].items():
        assert sha(BASE / name) == digest, name
    source = HERE.parent / 'extension_4006' / 'results_extension' / 'adapter_0128'
    source_sha = json.loads((HERE.parent / 'extension_4006' /
                             'probe_extension_0128' / 'runtime.json').read_text())['adapter_sha256']
    assert sha(source / 'adapter_model.safetensors') == source_sha
    bank = [json.loads(line) for line in (HERE / 'train.jsonl').read_text().splitlines()]
    assert len(bank) == 4096
    import numpy as np
    import torch
    from torch.nn import functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    torch.set_num_threads(8)
    seed = 400720261116
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tok.pad_token_id = 151645
    encoded = []
    for row in bank:
        user = {'role': 'user', 'content': row['prompt']}
        assistant = {'role': 'assistant', 'content': row['answer']}
        prefix = tok.encode(tok.apply_chat_template([user], tokenize=False,
            add_generation_prompt=True), add_special_tokens=False)
        full = tok.encode(tok.apply_chat_template([user, assistant], tokenize=False),
                          add_special_tokens=False)
        assert full[:len(prefix)] == prefix
        encoded.append((full, [-100] * len(prefix) + full[len(prefix):]))
    order = list(range(4096))
    random.Random(seed).shuffle(order)
    encoded = [encoded[i] for i in order]
    out = HERE / f'results_{args.arm}'
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
        attn_implementation='sdpa', device_map={'': 'cuda'}, local_files_only=True)
    model = PeftModel.from_pretrained(base, source, is_trainable=True)
    model.train()
    model.config.use_cache = False
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-5, betas=(.9, .999),
                                  weight_decay=.01, fused=True)

    def backward(block, micro):
        total = 0.
        for start in range(0, len(block), micro):
            part = block[start:start + micro]
            width = max(len(ids) for ids, _ in part)
            ids = torch.full((len(part), width), 151645, device='cuda', dtype=torch.long)
            mask = torch.zeros_like(ids)
            labels = torch.full_like(ids, -100)
            for j, (x, y) in enumerate(part):
                ids[j, :len(x)] = torch.tensor(x, device='cuda')
                labels[j, :len(y)] = torch.tensor(y, device='cuda')
                mask[j, :len(x)] = 1
            targets = labels[:, 1:]
            valid = targets != -100
            counts = valid.sum(1)
            owner = torch.arange(len(part), device='cuda')[:, None].expand_as(targets)[valid]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, :-1]
                losses = F.cross_entropy(logits[valid].float(), targets[valid], reduction='none')
                loss = (losses / counts[owner]).sum() / len(block)
            assert torch.isfinite(loss)
            loss.backward()
            total += float(loss.item())
            del ids, mask, labels, targets, valid, counts, owner, logits, losses, loss
        return total

    batches = [encoded[i:i + 32] for i in range(0, len(encoded), 32)]
    probe = max(batches, key=lambda block: max(len(ids) for ids, _ in block))
    micro = None
    for candidate in (32, 16, 8):
        try:
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            backward(probe, candidate)
            peak = torch.cuda.max_memory_allocated()
            if peak <= .85 * torch.cuda.get_device_properties(0).total_memory:
                micro = candidate
                break
        except torch.cuda.OutOfMemoryError:
            pass
    assert micro is not None
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    (out / 'accommodation.json').write_text(json.dumps({'microbatch': micro,
                                                         'peak_bytes': peak}))

    model.save_pretrained(out / 'adapter_0000')
    with (out / 'training.jsonl').open('w', buffering=1) as log:
        for step in range(1, 129):
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            loss = backward(encoded[(step - 1) * 32:step * 32], micro)
            norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            lr = 3e-6 + 7e-6 * (1 + math.cos(math.pi * (step - 1) / 127)) / 2
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.step()
            if step in (32, 64, 96, 128):
                model.save_pretrained(out / f'adapter_{step:04d}')
                torch.cuda.synchronize()
                print(json.dumps({'arm': args.arm, 'step': step, 'loss': loss,
                    'seconds': time.monotonic() - started}), flush=True)
            log.write(json.dumps({'step': step, 'loss': loss,
                'grad_norm': float(norm), 'lr': lr,
                'seconds': time.monotonic() - tick}) + '\n')
    (out / 'runtime.json').write_text(json.dumps({'status': 'COMPLETE_R3803_TRAINING',
        'arm': args.arm, 'source_adapter_sha256': source_sha,
        'updates': 128, 'microbatch': micro,
        'seconds': time.monotonic() - started}, indent=2))


if __name__ == '__main__':
    main()
