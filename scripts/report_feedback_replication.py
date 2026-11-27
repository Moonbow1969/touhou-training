"""Combine the two preregistered randomized Hard tests after both physical rosters finish."""
import argparse
import json
import shutil
import time
from pathlib import Path

from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--public', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sources = [root/'native10/blocked-randomization.json', root/'native17/preregistered-primary.json']
    credits = root/'evaluation/credits.json'
    while not all(path.exists() for path in [credits, *sources]):
        time.sleep(30)
    plan = json.loads((root/'camera-protocol.json').read_bytes())
    rows = json.loads(credits.read_bytes())
    expected = {(int(game), seed, sha) for game, seeds in plan['eval_seeds'].items()
                for seed in seeds for sha in (plan['behavior_sha256'], plan['candidate_sha256'])}
    if len(rows) != 128 or {(r['title'], r['seed'], r['policy_sha256']) for r in rows} != expected:
        raise ValueError('Incomplete frozen full-game roster')
    th10 = json.loads(sources[0].read_bytes())['hard']
    th17 = json.loads(sources[1].read_bytes())
    if th10['legal_assignments'] != 46656 or th17['assignments'] != 160000:
        raise ValueError('Unexpected blocked randomization population')
    effects = {'TH10': (th10['mean_difference'], th10['exact_two_sided_randomization_p']),
               'TH17': (th17['candidate_minus_reference'], th17['exact_two_sided_p'])}
    result = {}
    previous = 0.0
    for index, (title, (ticks, raw)) in enumerate(sorted(effects.items(), key=lambda entry: entry[1][1])):
        if not 0 <= raw <= 1:
            raise ValueError('Invalid probability')
        previous = max(previous, min(1.0, (len(effects) - index) * raw))
        result[title] = dict(paired_tick_difference=ticks, exact_two_sided_p=raw, holm_p=previous,
                             reject_at_005=previous <= 0.05)
    output = root/'evaluation/hard-replication.json'
    freeze(output, dict(hard_pairs_per_game=24, results=result, protocol_sha256=digest(root/'camera-protocol.json'),
                        credits_sha256=digest(credits), sources={str(p.relative_to(root).as_posix()):digest(p) for p in sources}))
    public = args.public.resolve()/root.name
    for path in [output, *sources]:
        destination = public/path.relative_to(root)
        destination.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2(path, destination)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
