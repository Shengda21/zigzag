"""Evaluate fixed lower-order and high-capability routes on one bank."""
import hashlib
import json
import re
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL = Path('/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct')
BANK = ROOT / 'counterfactual_state_use_4301'
ADAPTERS = {
    'r1603': {
        'lower': ROOT / 'prospective_support_3803' / 'source_results' / 'adapter_0512',
        'trained': ROOT / 'prospective_support_3803' / 'results_control' / 'adapter_0128'},
    'r4004': {
        'lower': ROOT / 'prospective_world_4004' / 'results_repair' / 'adapter_0032',
        'trained': ROOT / 'prospective_world_4004' / 'stop_calibration_4007' /
                   'results_stop' / 'adapter_0032'},
}
SYMBOLS = dict(zip('ABCDE', 'WYVXU'))
STATE = re.compile(r'[0-7](?: [0-7]){7}\Z')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def prompt(root, op):
    return ('Input: ' + ' '.join(map(str, root)) + '\nProgram: ' +
            SYMBOLS[op] +
            '\nReturn the state after each instruction, separated by " | ". '
            'Return only digits, spaces, and separators.')


def decode(model, tok, encoded, cap, batch_size):
    import torch
    outputs = [None] * len(encoded)
    prompt_tokens = generated_tokens = 0
    for start in range(0, len(encoded), batch_size):
        chosen = encoded[start:start + batch_size]
        batch = tok.pad({'input_ids': chosen}, padding=True,
                        return_tensors='pt').to('cuda')
        width = batch['input_ids'].shape[1]
        with torch.inference_mode():
            generated = model.generate(**batch, max_new_tokens=cap,
                do_sample=False, num_beams=1, repetition_penalty=1.,
                eos_token_id=[151645, 151643], pad_token_id=151645)
        prompt_tokens += sum(len(x) for x in chosen)
        for j, ids in enumerate(chosen):
            raw = generated[j, width:].tolist()
            stop = next((i for i, token in enumerate(raw)
                         if token in (151645, 151643)), None)
            content = raw if stop is None else raw[:stop]
            generated_tokens += len(content) + int(stop is not None)
            outputs[start + j] = {
                'prediction': tok.decode(content, skip_special_tokens=False,
                                         clean_up_tokenization_spaces=False),
                'eos': None if stop is None else raw[stop],
                'raw_generated_ids': raw,
                'prompt_tokens': len(ids),
                'generated_tokens': len(content) + int(stop is not None)}
    return outputs, prompt_tokens, generated_tokens


def single_call(model, tok, bank, world, route, out):
    encoded = [tok.encode(tok.apply_chat_template(
        [{'role': 'user', 'content': row['prompt']}], tokenize=False,
        add_generation_prompt=True), add_special_tokens=False) for row in bank]
    tick = time.monotonic()
    outputs, prompt_tokens, generated_tokens = decode(model, tok, encoded, 180, 256)
    with (out / f'single_{route}.jsonl').open('w', buffering=1) as log:
        for row, pred in zip(bank, outputs):
            log.write(json.dumps({'id': row['id'], 'world': world,
                                  'route': route, **pred}) + '\n')
    return {'route': route, 'calls': len(bank), 'prompt_tokens': prompt_tokens,
            'generated_tokens': generated_tokens,
            'seconds': time.monotonic() - tick}


def controller(model, tok, bank, world, out):
    current = [tuple(row['root']) for row in bank]
    active = [True] * len(bank)
    traces = [[] for _ in bank]
    strict_steps = [[] for _ in bank]
    costs = {'route': 'lower_controller', 'calls': 0, 'prompt_tokens': 0,
             'generated_tokens': 0}
    tick = time.monotonic()
    for step in range(8):
        indices = [i for i, row in enumerate(bank)
                   if active[i] and step < row['length']]
        prompts = [tok.encode(tok.apply_chat_template(
            [{'role': 'user', 'content': prompt(current[i], bank[i]['word'][step])}],
            tokenize=False, add_generation_prompt=True),
            add_special_tokens=False) for i in indices]
        outputs, prompt_tokens, generated_tokens = decode(model, tok, prompts, 40, 256)
        costs['calls'] += len(indices)
        costs['prompt_tokens'] += prompt_tokens
        costs['generated_tokens'] += generated_tokens
        for i, pred in zip(indices, outputs):
            first = pred['prediction'].split(' | ')[0]
            if not STATE.fullmatch(first):
                active[i] = False
                traces[i].append(None)
                strict_steps[i].append(False)
                continue
            state = tuple(int(x) for x in first.split(' '))
            current[i] = state
            traces[i].append(state)
            strict_steps[i].append(pred['eos'] == 151645 and
                                   pred['prediction'] == first)
        print(json.dumps({'world': world, 'route': 'lower_controller',
                          'step': step + 1, 'active': sum(active),
                          'calls': costs['calls']}), flush=True)
    with (out / 'controller.jsonl').open('w', buffering=1) as log:
        for row, states, strict in zip(bank, traces, strict_steps):
            log.write(json.dumps({'id': row['id'], 'world': world,
                                  'states': states, 'strict_steps': strict}) + '\n')
    costs['seconds'] = time.monotonic() - tick
    return costs


def main():
    manifest = json.loads((HERE / 'manifest.json').read_text())
    assert manifest['status'] == 'FROZEN_BEFORE_R4401_QUERIES'
    for name, digest in manifest['files'].items():
        assert sha(HERE / name) == digest, name
    for world, digest in manifest['banks'].items():
        assert sha(BANK / f'bank_{world}.jsonl') == digest
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True,
                                         trust_remote_code=False)
    tok.pad_token_id = 151645
    tok.padding_side = 'left'
    for world in ('r1603', 'r4004'):
        bank = rows(BANK / f'bank_{world}.jsonl')
        assert len(bank) == 256
        out = HERE / f'probe_{world}'
        out.mkdir(exist_ok=False)
        costs = []
        for route in ('lower', 'trained'):
            adapter = ADAPTERS[world][route]
            adapter_sha = sha(adapter / 'adapter_model.safetensors')
            base = AutoModelForCausalLM.from_pretrained(
                MODEL, torch_dtype=torch.bfloat16, attn_implementation='sdpa',
                device_map={'': 'cuda'}, local_files_only=True)
            model = PeftModel.from_pretrained(base, adapter, is_trainable=False)
            model.eval()
            model.config.use_cache = True
            costs.append(single_call(model, tok, bank, world, route, out))
            if route == 'lower':
                costs.append(controller(model, tok, bank, world, out))
            (out / f'{route}_adapter_sha256.txt').write_text(adapter_sha + '\n')
            del model, base
            torch.cuda.empty_cache()
        (out / 'runtime.json').write_text(json.dumps({
            'status': 'COMPLETE_R4401_BASELINES', 'world': world,
            'bank_sha256': manifest['banks'][world], 'costs': costs}, indent=2))
    (HERE / 'complete.txt').write_text('COMPLETE_R4401_BASELINES\n')


if __name__ == '__main__':
    main()
