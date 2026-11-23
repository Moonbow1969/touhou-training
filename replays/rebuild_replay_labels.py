import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np

from .collect_stage import AlignmentError, CONTROL_BITS, ROOT, align, save_stage
from .inspect_rpy import DECODER_VERSION, action_streams


def decode(entry):
    path = ROOT / entry['name']
    if sha256(path.read_bytes()).hexdigest() != entry['sha256']:
        raise ValueError('Replay SHA-256 differs from the corpus manifest.')
    words = {
        stage: np.frombuffer(body, dtype='<u2').reshape(-1, 3)[:, 0].copy()
        for stage, body in action_streams(path, entry['title'])
    }
    sizes = {str(stage): len(values) for stage, values in words.items()}
    if sizes != entry['frames_by_stage']:
        raise ValueError('Decoded stage list or frame counts differ from the corpus manifest.')
    return words


def refresh_action_cache(name, words):
    archive = ROOT / f'{Path(name).stem}-actions.npz'
    keys = {f'stage_{stage}_input_words': values for stage, values in words.items()}
    changed = 0
    if archive.exists():
        with np.load(archive) as original:
            changed = sum(int(np.count_nonzero(values != original[key])) for key, values in keys.items())
        if not changed:
            return changed
    temporary = archive.with_suffix(f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **keys)
    temporary.replace(archive)
    return changed


def repair_stage(source, destination, meta, words):
    with np.load(source.with_suffix('.npz')) as stored:
        arrays = dict(stored)
    old_alignment = meta['moving_alignment']
    old_pairs = meta['pairs']
    if source.parent.name == 'rejected':
        paired, summary = align(meta['replay'], arrays, words, meta['stride'], meta['elapsed'])
        meta.pop('failure', None)
        meta.update(summary)
        reviewed_alignment = summary['moving_alignment']
        corrected = None
    else:
        stride = arrays['expert_action_chunk'].shape[1]
        indices = arrays['stage_frame'] + meta['offset']
        legal = (indices >= 0) & (indices + stride < len(words))
        actual = arrays['native'][legal] & CONTROL_BITS
        expected = words[indices[legal]] & CONTROL_BITS
        moving = ((actual | expected) & 0xf0) != 0
        reviewed_alignment = float(np.mean(actual[moving] == expected[moving])) if moving.any() else 0
        if reviewed_alignment < .95:
            raise AlignmentError('Previously paired rows failed revalidation.', {'agreement': reviewed_alignment})
        keep = legal.copy()
        keep[legal] = actual == expected
        paired = {key: values[keep] for key, values in arrays.items()}
        new_chunks = words[indices[keep, None] + np.arange(1, stride + 1)]
        corrected = int(np.count_nonzero(np.any(new_chunks != paired['expert_action_chunk'], axis=1)))
        paired['expert_action_chunk'] = new_chunks
        meta.update(pairs=int(keep.sum()), stride=stride, revalidated_existing_alignment=reviewed_alignment)
    meta['decoder'] = DECODER_VERSION
    save_stage(destination / source.stem, paired, meta)
    return {
        'replay': meta['replay'], 'title': meta['title'], 'split': meta['split'], 'stage': meta['stage'],
        'source': source.parent.name, 'old_alignment': old_alignment,
        'new_alignment': reviewed_alignment, 'old_pairs': old_pairs, 'pairs': meta['pairs'],
        'corrected_existing_chunks': corrected,
    }


def rebuild(item):
    entry, sources, destination, actions = item
    try:
        words = decode(entry)
        changed = refresh_action_cache(entry['name'], words) if actions else 0
    except (OSError, ValueError, KeyError) as error:
        return {'name': entry['name'], 'title': entry['title'], 'decode_error': str(error)}
    stages = []
    for source in sources:
        meta = json.loads(source.read_text(encoding='utf-8'))
        try:
            stages.append(repair_stage(source, destination, meta, words[meta['stage']]))
        except AlignmentError as error:
            stages.append({'replay': entry['name'], 'stage': meta['stage'], 'error': error.summary})
    return {'name': entry['name'], 'title': entry['title'], 'changed_words': changed, 'stages': stages}


def main():
    parser = argparse.ArgumentParser(description='Rebuild replay labels from verified downloads and original game snapshots.')
    parser.add_argument('--paired-out', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--all-actions', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    args.paired_out.mkdir(parents=True, exist_ok=True)
    manifests = [ROOT / f'{name}.json' for name in ('corpus', 'multigame-corpus', 'expanded-corpus')]
    entries = {}
    for manifest in manifests:
        for row in json.loads(manifest.read_text(encoding='utf-8')):
            entries[row['name']] = dict(row, title=row.get('title', 14))

    captures = {}
    for folder in ('paired', 'rejected'):
        for path in sorted((ROOT / folder).glob('th*-stage*.json')):
            row = json.loads(path.read_text(encoding='utf-8'))
            entries.setdefault(row['name'], row)
            captures.setdefault(row['name'], {}).setdefault(row['stage'], path)
    names = sorted(entries if args.all_actions else captures, key=lambda n: (n not in captures, n))
    jobs = ((entries[name], tuple(captures.get(name, {}).values()), args.paired_out, args.all_actions)
            for name in names)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(rebuild, jobs, chunksize=1):
            results.append(result)
            if len(results) % 250 == 0:
                print(f'Checked {len(results)} of {len(names)} replays.', flush=True)
    validated = [row for row in results if 'decode_error' not in row]
    stages = [stage for row in validated for stage in row['stages']]
    accepted = [stage for stage in stages if 'error' not in stage]
    split_counts = Counter()
    for stage in accepted:
        split_counts[stage['split']] += stage['pairs']
    report = {
        'decoder': DECODER_VERSION, 'decodes_checked': len(results),
        'valid_replay_decodes': dict(Counter(row['title'] for row in validated)),
        'decoding_failures': [row for row in results if 'decode_error' in row],
        'action_caches_changed': sum(row['changed_words'] > 0 for row in validated),
        'corrected_action_words': sum(row['changed_words'] for row in validated),
        'paired_stages': len(accepted),
        'split_pairs': dict(split_counts),
        'corrected_existing_chunks': sum(row['corrected_existing_chunks'] or 0 for row in accepted),
        'recovered_stages': [row for row in accepted if row['source'] == 'rejected'],
        'rejected_stages': [row for row in stages if 'error' in row],
        'stages': stages,
    }
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print({key: value for key, value in report.items() if key not in ('stages', 'recovered_stages', 'decoding_failures')},
          flush=True)


if __name__ == '__main__':
    main()
