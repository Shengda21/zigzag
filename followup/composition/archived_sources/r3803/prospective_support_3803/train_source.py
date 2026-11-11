"""Train the precommitted hidden r1603 source, without querying probes."""
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    manifest = json.loads((HERE / 'source_manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert sha(HERE / name) == digest, name
    import numpy as np
    import torch
    from torch.nn import functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    torch.set_num_threads(8)
    random.seed(3803)
    np.random.seed(3803)
    torch.manual_seed(3803)
    torch.cuda.manual_seed_all(3803)
    model_path = Path('/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct')
    output = HERE / 'source_results'
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    tok.pad_token_id = 151645
    train = [json.loads(line) for line in (HERE / 'source_train.jsonl').read_text().splitlines()]
    assert len(train) == 16384
    encoded = []
    for row in train:
        user = {'role': 'user', 'content': row['prompt']}
        assistant = {'role': 'assistant', 'content': row['answer']}
        prefix = tok.encode(tok.apply_chat_template([user], tokenize=False,
                            add_generation_prompt=True), add_special_tokens=False)
        full = tok.encode(tok.apply_chat_template([user, assistant], tokenize=False),
                          add_special_tokens=False)
        assert full[:len(prefix)] == prefix
        encoded.append((full, [-100] * len(prefix) + full[len(prefix):]))
    base = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
            attn_implementation='sdpa', device_map={'': 'cuda'}, local_files_only=True)
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.,
        bias='none', task_type='CAUSAL_LM', target_modules=[
            'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
    params = [p for p in model.parameters() if p.requires_grad]

    def backward(records, micro):
        total = 0.
        for start in range(0, len(records), micro):
            block = records[start:start + micro]
            width = max(len(ids) for ids, _ in block)
            ids = torch.full((len(block), width), 151645, device='cuda', dtype=torch.long)
            mask = torch.zeros_like(ids)
            labels = torch.full_like(ids, -100)
            for j, (x, y) in enumerate(block):
                ids[j, :len(x)] = torch.tensor(x, device='cuda')
                labels[j, :len(y)] = torch.tensor(y, device='cuda')
                mask[j, :len(x)] = 1
            targets = labels[:, 1:]
            valid = targets != -100
            counts = valid.sum(1)
            owner = torch.arange(len(block), device='cuda')[:, None].expand_as(targets)[valid]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, :-1]
                losses = F.cross_entropy(logits[valid].float(), targets[valid], reduction='none')
                loss = (losses / counts[owner]).sum() / len(records)
            assert torch.isfinite(loss)
            loss.backward()
            total += loss.item()
            del ids, mask, labels, targets, valid, counts, owner, logits, losses, loss
        return total

    model.train()
    model.config.use_cache = False
    batches = [encoded[i:i + 32] for i in range(0, len(encoded), 32)]
    probe = max(batches, key=lambda b: max(len(x) for x, _ in b))
    micro = None
    for size in (32, 16, 8):
        try:
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            backward(probe, size)
            peak = torch.cuda.max_memory_allocated()
            if peak <= .8 * torch.cuda.get_device_properties(0).total_memory:
                micro = size
                break
        except torch.cuda.OutOfMemoryError:
            pass
    assert micro is not None
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    (output / 'accommodation.json').write_text(json.dumps({'microbatch': micro, 'peak_bytes': peak}))
    optimizer = torch.optim.AdamW(params, lr=1e-4, betas=(.9, .999),
                                  weight_decay=.01, fused=True)
    with (output / 'training.jsonl').open('w', buffering=1) as log:
        for step, batch in enumerate(batches, 1):
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            loss = backward(batch, micro)
            norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            lr = (1e-5 + 9e-5 * (1 + math.cos(math.pi * max(0, step - 16) / 496)) / 2) * min(step / 16, 1)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.step()
            if step % 32 == 0:
                torch.cuda.synchronize()
                print(json.dumps({'stage': 'source', 'step': step, 'loss': loss,
                                  'grad_norm': float(norm), 'seconds': time.monotonic() - started}), flush=True)
            log.write(json.dumps({'step': step, 'loss': loss, 'lr': lr,
                                  'grad_norm': float(norm), 'seconds': time.monotonic() - tick}) + '\n')
    model.save_pretrained(output / 'adapter_0512')
    (output / 'runtime.json').write_text(json.dumps({'status': 'COMPLETE_HIDDEN_R1603_SOURCE',
        'updates': 512, 'microbatch': micro, 'seconds': time.monotonic() - started}, indent=2))
    print(json.dumps({'stage': 'complete', 'seconds': time.monotonic() - started}), flush=True)


if __name__ == '__main__':
    main()
