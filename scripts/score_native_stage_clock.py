"""Compare the last observed Stage 1 script tick on matched original-game credits."""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(credits, title, arms):
    groups = {arm: {} for arm in arms}
    for row in credits:
        if row['title'] != title or row['arm'] not in groups:
            continue
        key = row['seed']
        group = groups[row['arm']]
        if key in group or row.get('latest_observed_stage1_native_tick') is None:
            raise ValueError(f'Duplicate seed or missing stage clock for {key}')
        group[key] = row
    left, right = (groups[arm] for arm in arms)
    if not left or left.keys() != right.keys():
        raise ValueError('Both arms require identical nonempty seed sets')
    ticks = {arm: {seed: row['latest_observed_stage1_native_tick'] for seed, row in group.items()}
             for arm, group in groups.items()}
    by_host = {}
    for seed in sorted(left):
        if (left[seed]['host'], left[seed]['game_sha256']) != (right[seed]['host'], right[seed]['game_sha256']):
            raise ValueError(f'Physical host or executable mismatch for {seed}')
        by_host.setdefault(left[seed]['host'], []).append(ticks[arms[1]][seed] - ticks[arms[0]][seed])
    rng = np.random.default_rng(2361126)
    draws = np.concatenate([rng.choice(v, size=(20000, len(v))) for v in by_host.values()], axis=1)
    return dict(measure='Last observed Stage 1 native script tick before credit end or stage transition.',
                title=title, arms=arms, n=len(left), per_seed=ticks,
                mean_ticks_by_arm={arm: np.mean(list(values.values())) for arm, values in ticks.items()},
                mean_paired_change=np.mean([v for group in by_host.values() for v in group]),
                paired_fixed_host_bootstrap95=np.quantile(draws.mean(axis=1), [.025, .975]).tolist())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--credits', type=Path, required=True)
    parser.add_argument('--title', type=int, required=True)
    parser.add_argument('--arms', nargs=2, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads(args.credits.read_bytes()), args.title, args.arms)
    args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
