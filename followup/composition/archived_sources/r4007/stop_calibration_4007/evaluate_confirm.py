"""Query the untouched r4006 confirmation bank at fixed adapters."""
import argparse
import hashlib
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BANK = HERE.parent / 'extension_4006' / 'confirm_4006'
MODEL = Path('/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('lower', 'source', 'selected'), required=True)
    args = parser.parse_args()
    frozen = json.loads((BANK / 'banks_manifest.json').read_text())
    for name, digest in frozen['files'].items():
        assert sha(BANK / name) == digest, name
    calibration = json.loads((HERE / 'manifest.json').read_text())
    for name, digest in calibration['files'].items():
        assert sha(HERE / name) == digest, name
    adapter = {'lower': HERE.parent / 'results_repair' / 'adapter_0032',
               'source': HERE.parent / 'extension_4006' / 'results_extension' / 'adapter_0128',
               'selected': HERE / 'results_stop' / 'adapter_0032'}[args.arm]
    step = {'lower': 32, 'source': 128, 'selected': 32}[args.arm]
    adapter_sha = sha(adapter / 'adapter_model.safetensors')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tok.pad_token_id = 151645
    tok.padding_side = 'left'
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
        attn_implementation='sdpa', device_map={'': 'cuda'}, local_files_only=True)
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False)
    model.eval()
    model.config.use_cache = True
    out = HERE / f'probe_confirm_{args.arm}'
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    for bank_name, expected in (('topology', 5760), ('hard', 1024)):
        bank = list(rows(BANK / f'{bank_name}_bank.jsonl'))
        assert len(bank) == expected
        prefixes = [tok.encode(tok.apply_chat_template(
            [{'role': 'user', 'content': r['prompt']}], tokenize=False,
            add_generation_prompt=True), add_special_tokens=False) for r in bank]
        if bank_name == 'topology':
            groups = [(dim, [i for i, r in enumerate(bank) if r['dimension'] == dim], cap)
                      for dim, cap in ((0, 40), (1, 55), (2, 180))]
        else:
            groups = [(length, [i for i, r in enumerate(bank) if r['length'] == length], 180)
                      for length in (7, 8)]
        with torch.inference_mode(), (out / f'predictions_{bank_name}.jsonl').open('w', buffering=1) as log:
            for group, indices, cap in groups:
                indices.sort(key=lambda i: len(prefixes[i]))
                for start in range(0, len(indices), 512):
                    chosen = indices[start:start + 512]
                    batch = tok.pad({'input_ids': [prefixes[i] for i in chosen]},
                                    padding=True, return_tensors='pt').to('cuda')
                    width = batch['input_ids'].shape[1]
                    generated = model.generate(**batch, max_new_tokens=cap,
                        do_sample=False, num_beams=1, repetition_penalty=1.,
                        eos_token_id=[151645, 151643], pad_token_id=151645)
                    for j, index in enumerate(chosen):
                        raw = generated[j, width:].tolist()
                        stop = next((k for k, token in enumerate(raw)
                                     if token in (151645, 151643)), None)
                        content = raw if stop is None else raw[:stop]
                        log.write(json.dumps({'id': bank[index]['id'],
                            'source_index': index, 'prediction': tok.decode(content,
                                skip_special_tokens=False,
                                clean_up_tokenization_spaces=False),
                            'raw_generated_ids': raw, 'first_eos_index': stop,
                            'terminator_id': None if stop is None else raw[stop],
                            'decode_cap_reached': stop is None and len(raw) >= cap}) + '\n')
                    print(json.dumps({'arm': args.arm, 'step': step,
                        'bank': bank_name, 'group': group,
                        'completed': start + len(chosen),
                        'seconds': time.monotonic() - started}), flush=True)
    (out / 'runtime.json').write_text(json.dumps({'status': 'COMPLETE_R3803_PROBE',
        'arm': args.arm, 'step': step, 'adapter_sha256': adapter_sha,
        'topology_generations': 5760, 'hard_generations': 1024,
        'seconds': time.monotonic() - started}, indent=2))


if __name__ == '__main__':
    main()
