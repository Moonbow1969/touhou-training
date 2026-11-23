import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from scripts.pretrain_policy import label_actions
from scripts.stage_policy import NormalStageOne


def direction(action):
    row, column = divmod(action, 3)
    return np.array([column - 1, row - 1])


def input_word(action):
    move, focus, shoot, bomb = action
    dx, dy = direction(move)
    return (int(dx > 0) * 128 | int(dx < 0) * 64 | int(dy > 0) * 32 | int(dy < 0) * 16
            | focus * 8 | bomb * 2 | shoot)


def calibrate(env):
    observation, _ = env.reset()
    result = []
    scale = np.array([192, 448])
    # Opposite actions keep the player away from playfield edges.
    for focus in (0, 1):
        for repeat in range(3):
            for move in (1, 7, 3, 5, 0, 8, 2, 6, 4):
                action = [move, focus, repeat % 2, 0]
                before = observation['context'][:2] * scale
                start = env.stage_frame()
                observation, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    raise RuntimeError('Action calibration was interrupted before the input sequence finished.')
                delta = observation['context'][:2] * scale - before
                actual_bits = env.reader.read_uint(env.reader.offsets.statics.input, relative=True) & 0xFB
                result.append({'action': action, 'frames': env.stage_frame() - start,
                               'delta': delta.tolist(), 'mask_matches': actual_bits == input_word(action),
                               'direction_matches': np.array_equal(np.where(abs(delta) > 0.3, np.sign(delta), 0),
                                                                   direction(move)),
                               'focus_matches': int(observation['context'][2]) == focus})
    before = env.game.read_game_val('bombs')
    _, _, _, _, _ = env.step([4, 0, 1, 1])
    after = env.game.read_game_val('bombs')
    return {'decisions': len(result), 'mask_matches': sum(row['mask_matches'] for row in result),
            'direction_matches': sum(row['direction_matches'] for row in result),
            'focus_matches': sum(row['focus_matches'] for row in result),
            'frames_per_decision': dict(Counter(row['frames'] for row in result)),
            'bomb_inventory_change': after - before, 'trials': result}


def replay_open_loop(env, words, cutoff, offset):
    env.reset()
    actions = 0
    timing = Counter()
    while env.stage_frame() < cutoff:
        start = env.stage_frame()
        # The saved replay alignment maps the observed stage clock to the input already applied.
        at = start + offset + 1
        if at < 0 or at + 4 > len(words):
            raise ValueError('The requested continuing replay frames are unavailable.')
        action = label_actions(words[None, at:at + 4])[0]
        _, _, terminated, truncated, info = env.step(action)
        timing[env.stage_frame() - start] += 1
        actions += 1
        if terminated or truncated or info['first_miss_frame'] is not None:
            break
    return {'cutoff_frame': cutoff, 'decisions': actions, 'reached_frame': info['stage_frame'],
            'first_miss_frame': info['first_miss_frame'], 'frames_per_decision': dict(timing),
            'replay_rng_restored': False}


def main():
    parser = argparse.ArgumentParser(description='Audit actual native control and replay open-loop execution.')
    parser.add_argument('--actions', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cutoff', type=int, default=1800)
    parser.add_argument('--offset', type=int, default=-1)
    args = parser.parse_args()
    if args.cutoff < 100:
        parser.error('The open-loop cutoff must be at least 100 frames.')
    env = NormalStageOne(Path(__file__).resolve().parents[2] / 'parakit', limit=1000, max_misses=2)
    report = {'replay_source': args.actions.name}
    try:
        report['actuator'] = calibrate(env)
        args.out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print('Calibration', {k: v for k, v in report['actuator'].items() if k != 'trials'}, flush=True)
        with np.load(args.actions) as source:
            words = source['stage_1_input_words']
        report['open_loop'] = replay_open_loop(env, words, args.cutoff, args.offset)
        print('Human action open-loop', report['open_loop'], flush=True)
        args.out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    finally:
        env.close()


if __name__ == '__main__':
    main()
