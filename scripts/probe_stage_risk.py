import argparse
from collections import deque
import json
from pathlib import Path
from statistics import median

import numpy as np
import torch

from scripts.evaluate_stage_policies import read_policy
from scripts.stage_policy import NormalStageOne
from scripts.stage_risk import action_clearances, suggest_dodge
from scripts.train_stage_dodge import DodgeHead, learned_action


def player_radius(env):
    return env.reader.player_hit_radius()


def trial(env, policy, assisted, corrections, head=None):
    observation, _ = env.reset()
    radius = player_radius(env)
    last_frames = deque(maxlen=16)
    opportunities = 0
    executed = 0
    while True:
        action = policy.predict(observation, deterministic=True)[0]
        clearance = action_clearances(observation, radius)
        alternative, original_margin, improved_margin = suggest_dodge(action, clearance)
        changed = not np.array_equal(action[:2], alternative[:2])
        if changed:
            opportunities += 1
            # Keep visited states and the candidate better action for the follow-up learning run.
            corrections.append((observation['bullets'].copy(), observation['context'].copy(),
                                alternative, action, env.last_frame, assisted))
        actual = action
        if assisted:
            actual = learned_action(head, action, clearance) if head is not None else alternative
        executed += int(not np.array_equal(action[:2], actual[:2]))
        last_frames.append({'frame': env.last_frame, 'policy_action': action.tolist(), 'executed': actual.tolist(),
                            'policy_clearance': original_margin, 'alternative_clearance': improved_margin})
        observation, _, done, truncated, info = env.step(actual)
        if done or truncated:
            return dict(info, opportunities=opportunities, interventions=executed,
                        last_decisions=list(last_frames), player_radius=radius)


def main():
    parser = argparse.ArgumentParser(description='Test whether short-horizon dodge guidance fixes closed-loop SFT.')
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--corrections', type=Path, required=True)
    parser.add_argument('--learned-head', type=Path)
    parser.add_argument('--blocks', type=int, default=2)
    parser.add_argument('--limit', type=int, default=750)
    args = parser.parse_args()
    if min(args.blocks, args.limit) <= 0:
        parser.error('Blocks and decision limit must be positive.')
    torch.set_num_threads(1)
    policy = read_policy(args.policy, torch.device('cuda'), motion_history=True)
    head = None
    if args.learned_head is not None:
        head = DodgeHead()
        head.load_state_dict(torch.load(args.learned_head, map_location='cpu', weights_only=True))
        head.eval()
    env = NormalStageOne(Path(__file__).resolve().parents[2] / 'parakit', limit=args.limit,
                         max_misses=1, motion_history=True)
    corrections = []
    variant = 'learned_dodge_sft' if head is not None else 'risk_assisted_sft'
    report = {'training_checkpoint': str(args.policy), 'decision_limit': args.limit,
              'max_misses': 1, 'results': {'sft': [], variant: []}}
    try:
        for block in range(args.blocks):
            for assisted in (False, True) if block % 2 == 0 else (True, False):
                name = variant if assisted else 'sft'
                result = dict(trial(env, policy, assisted, corrections, head), block=block + 1)
                report['results'][name].append(result)
                if corrections:
                    fields = zip(*corrections)
                    names = ('bullets', 'context', 'action', 'original_action', 'stage_frame', 'assisted')
                    np.savez_compressed(args.corrections, **dict(zip(names, (np.stack(field) for field in fields))))
                report['summary'] = {
                    label: {'n': len(rows), 'furthest_frame_median': median(r['stage_frame'] for r in rows),
                            'first_miss_frames': [r['first_miss_frame'] for r in rows],
                            'avoided_miss_to_limit': sum(r['outcome'] == 'limit' for r in rows)}
                    for label, rows in report['results'].items() if rows
                }
                args.out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
                print(name, {key: value for key, value in result.items() if key != 'last_decisions'}, flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
