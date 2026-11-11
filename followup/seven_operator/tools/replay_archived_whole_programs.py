"""Replay frozen whole-program scorers from full archives on a CPU."""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import tempfile

from verify_release_archives import digest


ENDPOINTS = ('source', 'two', 'three', 'exchange1', 'clean1', 'exchange2', 'clean2')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def restore(archive, destination, directory):
    def checked_member(member, root):
        name = PurePosixPath(member.name)
        if name.parts[0] != directory or '..' in name.parts:
            raise ValueError(f'Unexpected archive path: {member.name}')
        if not (member.isfile() or member.isdir()):
            raise ValueError(f'Unexpected archive member type: {member.name}')
        return tarfile.data_filter(member, root)

    with tarfile.open(archive, mode='r|gz') as source:
        source.extractall(destination, filter=checked_member)
    return destination / directory


def replay_word_statistics(root, environment):
    result_path = root / 'RESULT.json'
    expected = read_json(result_path)
    subprocess.run([sys.executable, str(root / 'word_statistics.py')], cwd=root,
                   env=environment, stdout=subprocess.DEVNULL, check=True)
    computed = read_json(result_path)
    max_difference = 0.0
    for tag, cells in expected['endpoints'].items():
        for cell, values in cells.items():
            old_bounds = values.pop('bounds')
            new_bounds = computed['endpoints'][tag][cell].pop('bounds')
            if len(old_bounds) != 2 or len(new_bounds) != 2:
                raise ValueError(f'Expected two confidence bounds at {tag}/{cell}')
            for old, new in zip(old_bounds, new_bounds):
                if not (0 <= old <= 1 and 0 <= new <= 1):
                    raise ValueError(f'Invalid confidence bound at {tag}/{cell}')
                difference = abs(old - new)
                if difference > 8 * max(math.ulp(old), math.ulp(new)):
                    raise ValueError(f'Confidence bound differs at {tag}/{cell}: {old}, {new}')
                max_difference = max(max_difference, difference)
    if computed != expected:
        raise ValueError('Archived word-pattern scores, flags, contrasts or decisions differ')
    return max_difference


def replay(record, archive_dir, environment):
    filename = record['filename']
    directory = record['directory']
    if Path(filename).name != filename or Path(directory).name != directory:
        raise ValueError(f'Nonportable archive identity: {record}')
    archive = archive_dir / filename
    if archive.stat().st_size != record['bytes'] or digest(archive) != record['sha256']:
        raise ValueError(f'Archive integrity check failed: {archive}')
    with tempfile.TemporaryDirectory(prefix='simods-archive-replay-') as work:
        root = restore(archive, Path(work), directory)
        recorded_case_audit = root / 'CASE_RAW_AUDIT.json'
        if digest(recorded_case_audit) != record['case_raw_audit_sha256']:
            raise ValueError(f'Unexpected archived case audit: {filename}')
        case = read_json(recorded_case_audit)
        if case['status'] != 'PASS_FROZEN_CASE_RAW_RECONSTRUCTION':
            raise ValueError(f'Archived case audit did not pass: {filename}')
        if not (root / 'RAW_AUDIT.json').exists():
            return {'filename': filename, 'case': case['case'],
                    'status': 'NO_COMPLETED_WHOLE_PROGRAM_OUTPUTS', 'archive_sha256': record['sha256']}

        original = read_json(root / 'RAW_AUDIT.json')
        command = [sys.executable, str(root / 'audit_outputs.py'), '--study', str(root),
                   '--policy', str(root / 'weights.npy')]
        final = root / 'FINAL_RESULT.json'
        if final.exists():
            looks = read_json(final)['final_look']
            tags = [f'{tag}_L{look}' for look in range(1, looks + 1) for tag in ENDPOINTS]
            command += ['--tags', *tags, '--aggregate', 'NO_AGGREGATE_FILE.json']
        else:
            # Compare the aggregate below so library rounding can only affect the bounds.
            command += ['--tags', *ENDPOINTS, '--aggregate', 'NO_AGGREGATE_FILE.json']
        log = subprocess.run(command, cwd=root, env=environment, capture_output=True)
        if log.returncode:
            sys.stderr.write(log.stderr.decode('utf-8', 'replace')[-6000:])
            raise RuntimeError(f'Frozen whole-program replay failed: {filename}')
        reconstructed = read_json(root / 'RAW_AUDIT.json')
        if reconstructed != original:
            differences = [k for k in original.keys() | reconstructed.keys()
                           if original.get(k) != reconstructed.get(k)]
            raise ValueError(f'Raw replay differs from archive in {filename}: {differences}')
        row = {'filename': filename, 'case': case['case'], 'status': reconstructed['status'],
               'archive_sha256': record['sha256'],
               'total_programs_reconstructed': reconstructed['total_programs_reconstructed'],
               'same_whole_program_reconstruction_record': True}
        if not final.exists():
            row['maximum_cpu_binomial_bound_rounding_difference'] = replay_word_statistics(root, environment)
            row['same_scores_flags_contrasts_and_decisions'] = True
        return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--archive-dir', type=Path, required=True)
    parser.add_argument('--archive', action='append', required=True)
    parser.add_argument('--base-snapshot', type=Path, required=True)
    parser.add_argument('--tokenizer-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    index = {r['filename']: r for r in read_json(args.index)}
    if len(set(args.archive)) != len(args.archive):
        parser.error('Specify each archive at most once')
    snapshot = read_json(args.base_snapshot)['files']
    tokenizer = args.tokenizer_dir.resolve()
    verified = {}
    for name in ('tokenizer.json', 'tokenizer_config.json'):
        path = tokenizer / name
        expected = snapshot[name]
        if path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
            raise ValueError(f'Tokenizer integrity check failed: {path}')
        verified[name] = expected['sha256']
    environment = dict(os.environ, SIMODS_BASE_MODEL=str(tokenizer), HF_HUB_OFFLINE='1',
                       TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='',
                       TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    results = []
    for filename in args.archive:
        if filename not in index or not index[filename].get('case_raw_audit_sha256'):
            parser.error(f'No indexed case raw audit for {filename}')
        row = replay(index[filename], args.archive_dir, environment)
        results.append(row)
        print(json.dumps(row), flush=True)
    output = {
        'scope': ('Frozen whole-program score and token reconstruction from original archives '
                  'using only the matching tokenizer and recorded policy weights. '
                  'Separate from cloud training-lineage and component audits.'),
        'python': sys.version,
        'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'transformers')},
        'tokenizer_sha256': verified,
        'archives': results,
        'new_model_queries': 0,
        'total_programs_reconstructed': sum(r.get('total_programs_reconstructed', 0) for r in results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
