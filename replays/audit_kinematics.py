"""Measure constant-input movement on aligned original game replay frames."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def measurements(path):
    with np.load(path) as source:
        frames = source['stage_frame']
        position = source['context'][:, :2] * (192, 448)
        lives = source['context'][:, 3]
        keys = source['expert_action_chunk'][:-1]
    focused = (keys & 8) != 0
    dx = ((keys & 128) != 0).astype(np.int8) - ((keys & 64) != 0)
    dy = ((keys & 32) != 0).astype(np.int8) - ((keys & 16) != 0)
    directions = np.stack((dx, dy), axis=-1)
    first = directions[:, 0]
    lengths = np.linalg.norm(first, axis=1)
    unit = first / np.maximum(lengths, 1)[:, None]
    unclamped = ((np.abs(position[:, 0]) < 182) & (position[:, 1] > 34) & (position[:, 1] < 430))
    consistent = ((np.diff(frames) == keys.shape[1]) & (np.diff(lives) == 0) & (lengths > 0)
                  & unclamped[:-1] & unclamped[1:] & ((keys & 2) == 0).all(axis=1)
                  & (directions == first[:, None, :]).all(axis=(1, 2))
                  & (focused == focused[:, :1]).all(axis=1))
    velocity = np.diff(position, axis=0) / keys.shape[1]
    speed = (velocity * unit).sum(1)
    lateral = np.linalg.norm(velocity - speed[:, None] * unit, axis=1)
    return [(speed[consistent & (focused[:, 0] == focus)], lateral[consistent & (focused[:, 0] == focus)])
            for focus in (False, True)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'validation', 'test'), default='train')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Use a fresh output path.')
    samples = defaultdict(list)
    for path in sorted(args.data.glob('*.json')):
        row = json.loads(path.read_text(encoding='utf-8'))
        if row['split'] != args.split:
            continue
        for focus, data in zip(('unfocused', 'focused'), measurements(path.with_suffix('.npz'))):
            if len(data[0]):
                samples[f'TH{row["title"]}/{row["shot"]}/{focus}'].append(data)
    profiles = {}
    for profile, values in sorted(samples.items()):
        speed, lateral = (np.concatenate(group) for group in zip(*values))
        profiles[profile] = {'intervals': len(speed), 'speed_p10_p50_p90': np.quantile(speed, [.1, .5, .9]).tolist(),
                             'opposite_or_stalled_fraction': float((speed <= .1).mean()),
                             'lateral_error_p90': float(np.quantile(lateral, .9))}
    if not profiles:
        raise ValueError('No constant-direction intervals in the requested split.')
    args.out.write_text(json.dumps({'split': args.split, 'profiles': profiles}, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(profiles), flush=True)


if __name__ == '__main__':
    main()
