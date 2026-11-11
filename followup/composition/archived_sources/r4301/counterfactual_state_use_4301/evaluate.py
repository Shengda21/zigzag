"""Generate paired visible-state continuations from two fixed adapters."""
import hashlib
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL = Path('/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct')
ADAPTERS = {
    'r1603': ROOT / 'prospective_support_3803' / 'results_control' / 'adapter_0128',
    'r4004': ROOT / 'prospective_world_4004' / 'stop_calibration_4007' /
             'results_stop' / 'adapter_0032',
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    manifest = json.loads((HERE / 'manifest.json').read_text())
    assert manifest['status'] == 'FROZEN_BEFORE_R4301_MODEL_QUERY'
    for name, digest in manifest['files'].items():
        assert sha(HERE / name) == digest, name

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True,
                                         trust_remote_code=False)
    tok.pad_token_id = 151645
    tok.padding_side = 'left'
    for world in ('r1603', 'r4004'):
        bank = rows(HERE / f'bank_{world}.jsonl')
        assert len(bank) == 256
        adapter = ADAPTERS[world]
        adapter_sha = sha(adapter / 'adapter_model.safetensors')
        base = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16, attn_implementation='sdpa',
            device_map={'': 'cuda'}, local_files_only=True)
        model = PeftModel.from_pretrained(base, adapter, is_trainable=False)
        model.eval()
        model.config.use_cache = True
        cases = []
        for index, row in enumerate(bank):
            prompt_ids = tok.encode(tok.apply_chat_template(
                [{'role': 'user', 'content': row['prompt']}], tokenize=False,
                add_generation_prompt=True), add_special_tokens=False)
            for condition in ('true', 'edit'):
                partial = row[f'prefix_{condition}']
                ids = prompt_ids + tok.encode(partial, add_special_tokens=False)
                assert tok.decode(ids, skip_special_tokens=False,
                                  clean_up_tokenization_spaces=False).endswith(partial)
                cases.append((index, condition, ids))
        out = HERE / f'probe_{world}'
        out.mkdir(exist_ok=False)
        start_time = time.monotonic()
        with torch.inference_mode(), (out / 'predictions.jsonl').open('w', buffering=1) as log:
            for start in range(0, len(cases), 128):
                block = cases[start:start + 128]
                batch = tok.pad({'input_ids': [ids for _, _, ids in block]},
                                padding=True, return_tensors='pt').to('cuda')
                width = batch['input_ids'].shape[1]
                generated = model.generate(**batch, max_new_tokens=120,
                    do_sample=False, num_beams=1, repetition_penalty=1.,
                    eos_token_id=[151645, 151643], pad_token_id=151645)
                for j, (index, condition, _) in enumerate(block):
                    raw = generated[j, width:].tolist()
                    stop = next((k for k, token in enumerate(raw)
                                 if token in (151645, 151643)), None)
                    content = raw if stop is None else raw[:stop]
                    prediction = tok.decode(content, skip_special_tokens=False,
                                            clean_up_tokenization_spaces=False)
                    log.write(json.dumps({'id': bank[index]['id'],
                        'source_index': index, 'condition': condition,
                        'prediction': prediction, 'raw_generated_ids': raw,
                        'first_eos_index': stop,
                        'terminator_id': None if stop is None else raw[stop],
                        'decode_cap_reached': stop is None and len(raw) >= 120}) + '\n')
                print(json.dumps({'world': world,
                                  'completed': start + len(block),
                                  'seconds': time.monotonic() - start_time}), flush=True)
        (out / 'runtime.json').write_text(json.dumps({
            'status': 'COMPLETE_R4301_COUNTERFACTUAL_GENERATION',
            'world': world, 'rows': len(cases), 'adapter_sha256': adapter_sha,
            'manifest_sha256': sha(HERE / 'manifest.json'),
            'seconds': time.monotonic() - start_time}, indent=2))
        del model, base
        torch.cuda.empty_cache()
    (HERE / 'complete.txt').write_text('COMPLETE_R4301_COUNTERFACTUAL_GENERATION\n')


if __name__ == '__main__':
    main()
