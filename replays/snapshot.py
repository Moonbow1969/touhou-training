"""Freeze aligned replay shards for repeatable training and held-out evaluation."""

import argparse
from collections import Counter
from hashlib import file_digest
import json
from pathlib import Path
from shutil import copy2

import numpy as np


def sha256(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def collect(sources):
    rows = {}
    players = {}
    for source in sources:
        if not source.is_dir():
            raise ValueError(f'Missing replay directory {source}')
        for path in sorted(source.glob('*.json')):
            row = json.loads(path.read_text(encoding='utf-8'))
            key = row['name'], row['stage']
            split = row['split']
            group = row.get('player_group', row['player'].casefold())
            if key in rows:
                raise ValueError(f'Duplicate replay stage {key}')
            if split not in ('train', 'validation', 'test') or players.setdefault(group, split) != split:
                raise ValueError(f'Inconsistent replay split for {key}')
            array = path.with_suffix('.npz')
            with np.load(array) as data:
                if row['pairs'] < 1 or len(data['expert_action_chunk']) != row['pairs']:
                    raise ValueError(f'Invalid aligned action count in {array}')
                if any(len(data[field]) != row['pairs'] for field in ('stage_frame', 'context', 'bullets')):
                    raise ValueError(f'Inconsistent observation lengths in {array}')
            rows[key] = (path, array, row)
    if not rows:
        raise ValueError('The snapshot needs at least one aligned replay.')
    return rows


def save(sources, destination):
    rows = collect(sources)
    names = [path.name for record in rows.values() for path in record[:2]]
    if len(set(names)) != len(names):
        raise ValueError('Replay filenames collide in the destination.')
    manifest = destination.with_name(destination.name + '.manifest.json')
    if destination.exists() or manifest.exists():
        raise FileExistsError('Choose a fresh snapshot directory and manifest.')
    destination.mkdir(parents=True)
    totals = Counter()
    by_title = Counter()
    entries = []
    for metadata, array, row in rows.values():
        for path in (metadata, array):
            copy2(path, destination / path.name)
        entries.append({'source': metadata.parent.name, 'replay': row['name'], 'stage': row['stage'],
                        'split': row['split'], 'pairs': row['pairs'],
                        'sha256': {path.name: sha256(destination / path.name) for path in (metadata, array)}})
        totals[row['split']] += row['pairs']
        by_title[row['split'], row['title']] += row['pairs']
    counts = {'pairs_by_split': dict(totals), 'pairs_by_split_and_title': {
        f'{split}/TH{title}': count for (split, title), count in sorted(by_title.items())},
        'shards': len(entries)}
    manifest.write_text(json.dumps({**counts, 'entries': entries}, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(counts), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources', nargs='+', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    save(args.sources, args.out)


if __name__ == '__main__':
    main()
