"""Measure direction changes and reversals for expert and sampled policies on identical TH17 frames."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze
from scripts.train_human_camera_policy import load_replay


def read(path):
    return json.loads(path.read_bytes())


def switches(previous, current):
    same = (previous * current).sum(1)
    opposite = (previous * current[:, ::-1]).sum(1) - previous[:, 4] * current[:, 4]
    return dict(pairs=len(current), changed=float((1 - same).sum()),
                opposite=float(opposite.sum()), stationary=float(current[:, 4].sum()))


def sequences(prob, clock):
    keep = np.diff(clock) == 4
    out = switches(prob[:-1][keep], prob[1:][keep])
    direction = np.array([(x, y) for y in (-1, 0, 1) for x in (-1, 0, 1)], dtype=float)
    direction /= np.maximum(1, np.linalg.norm(direction, axis=1, keepdims=True))
    mean = prob @ direction
    variance = np.maximum(0, 1 - prob[:, 4] - (mean * mean).sum(1))
    length = 8
    starts = np.flatnonzero(np.all(np.lib.stride_tricks.sliding_window_view(keep, length - 1), axis=1))
    indexes = starts[:, None] + np.arange(length)
    # For independent draws E||sum X||² = ||sum E[X]||² + sum trace Var[X].
    resultant = ((mean[indexes].sum(1) ** 2).sum(1) + variance[indexes].sum(1)) / length ** 2
    out.update(eight_command_windows=len(starts), sum_squared_normalized_resultant=float(resultant.sum()))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    d = args.data.resolve()
    root = d / 'camera-direction-sequences'
    root.mkdir(exist_ok=True)
    manifest = d / 'human-retail17-wave2-held/held-original-synchronized-frozen.json'
    models = {'later-sft': d / 'four-title-later-stage-sft/policy', 'later-ppo': d / 'later-human-online/policy'}
    source = read(models['later-sft'] / 'report.json')
    child = read(models['later-ppo'] / 'report.json')
    hashes = {name: digest(path / 'policy.ts') for name, path in models.items()}
    if hashes['later-sft'] != source['torchscript_sha256'] or hashes['later-sft'] != child['native_behavior_sha256'] or hashes['later-ppo'] != child['policy_sha256']:
        raise ValueError('Different deployed parent or child')
    freeze(root / 'protocol.json', dict(manifest=digest(manifest), models=hashes,
        population='All eight independently synchronized TH17 stages from the previously examined four pilots. Compare registered expert, 68-stage SFT and its two-game PPO on identical fixed expert screens. Also describe six-title PPO training actions on their own screens separately.',
        measures='Across adjacent observations exactly four native ticks apart, count any change, direct 180-degree nonstationary reversal, and neutral direction. For camera actors compute exact expectation for independent categorical samples per decision. Across eight contiguous commands report mean squared normalized resultant of direction unit vectors, neutral is zero. Focus speed, boundaries and collisions are outside this command-space diagnostic. Human nominal directions are sampled at the first native tick per four-tick interval. Overlapping windows have no independent-sample interpretation.'))
    torch.set_num_threads(2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    actors = {name: torch.jit.load(str(path / 'policy.ts'), map_location=device).eval() for name, path in models.items()}
    population = read(manifest)['qualified']
    if len(population) != 8 or len({v['player_group'] for v in population}) != 4:
        raise ValueError('Incomplete original human cohort')
    rows = []
    eye = np.eye(9)
    with torch.inference_mode():
        for item in population:
            path = d / item['path']
            if digest(path) != item['archive_sha256'] or item['player_group'].casefold() in {v['group'] for v in source['train']}:
                raise ValueError('Held human source changed or player overlaps')
            row = load_replay(path)
            stats = {'expert': sequences(eye[row['action'][:, 0]], row['native_frame'])}
            for name, actor in actors.items():
                prob = torch.cat([actor(batch.to(device).float() / 255)[0].softmax(1).cpu() for batch in torch.from_numpy(row['raw']).split(64)]).double().numpy()
                prob /= prob.sum(1, keepdims=True)
                stats[name] = sequences(prob, row['native_frame'])
            rows.append(dict(player=row['group'], difficulty=row['difficulty'], stage=item['stage'], sha256=item['archive_sha256'], metrics=stats))
    freeze(root / 'held-human-screens.json', dict(protocol=digest(root / 'protocol.json'), stages=rows))
    native = d / 'six-game-return-horizon/optimizer-input/credits.json'
    ledger = read(native)
    if len(ledger) != 192 or len({(v['title'], v['seed']) for v in ledger}) != 192:
        raise ValueError('Incomplete six-game allocation')
    actual = []
    for record in ledger:
        path = native.parent / record['file']
        if digest(path) != record['sha256']:
            raise ValueError('Changed recorded training actions')
        with np.load(path, allow_pickle=False) as batch:
            move = batch['action'][:, 0]
            keep = np.ones(len(move) - 1, dtype=bool)
            if 'native_action_eligible' in batch:
                eligible = batch['native_action_eligible'].reshape(-1).astype(bool)
                keep = eligible[:-1] & eligible[1:]
        stats = switches(eye[move[:-1][keep]], eye[move[1:][keep]])
        actual.append(dict(title=record['title'], rank=record['difficulty'], seed=record['seed'], **stats))
    freeze(root / 'actual-six-title-commands.json', dict(manifest=digest(native),
        unit='Consecutive sampled direction commands within one real training credit. Native decision intervals vary by title. These are the issued commands recorded at the model interface. Original key register checks are reported separately.', credits=actual))


if __name__ == '__main__':
    main()
