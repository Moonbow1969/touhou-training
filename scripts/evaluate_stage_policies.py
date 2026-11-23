import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import median

import numpy as np
import torch

from scripts.pretrain_policy import (CONTEXT_SIZE, MovementPolicy, action_space, append_stage_context,
                                    network_kwargs, observation_space)
from scripts.stage_policy import NormalStageOne, run_episode


def summarize(episodes):
    misses = [episode['first_miss_frame'] for episode in episodes if episode['first_miss_frame'] is not None]
    return {
        'trials': len(episodes),
        'clears': sum(episode['outcome'] == 'advanced_stage' for episode in episodes),
        'censored_first_miss': len(episodes) - len(misses),
        'median_first_miss_frame': median(misses) if misses else None,
        'median_furthest_frame': median([episode['stage_frame'] for episode in episodes]) if episodes else None,
    }


def read_policy(path, device, motion_history=False):
    weights = torch.load(path, map_location=device, weights_only=True)
    encoder = 'attention' if 'features_extractor.keys.weight' in weights else 'pool'
    metrics = path.with_name('metrics.json')
    configuration = json.loads(metrics.read_text(encoding='utf-8')) if metrics.is_file() else {}
    retain_recent = configuration.get('retain_recent_motion', False)
    stage_clock = configuration.get('stage_clock', False)
    policy = MovementPolicy(observation_space(motion_history, retain_recent, stage_clock), action_space(),
                            lambda _: 3e-4, **network_kwargs(configuration.get('hidden', 512), encoder)).to(device)
    policy.load_state_dict(weights)
    policy.motion_spans = configuration.get('motion_spans')
    policy.retain_recent_motion = retain_recent
    policy.stage_clock = stage_clock
    policy.factorized_movement = configuration.get('factorized_movement', False)
    policy.eval()
    return policy


def policy_observation(policy, env, observation, uses_history=True):
    if policy.motion_spans:
        observation = env.observation(policy.motion_spans, policy.retain_recent_motion)
    elif not uses_history:
        observation = {'bullets': observation['bullets'], 'context': observation['context'][:CONTEXT_SIZE]}
    if policy.stage_clock:
        context = append_stage_context(observation['context'], env.last_frame, env.snapshot.stage)
        observation = dict(observation, context=context)
    return observation


def stage_action(policy, env, observation, uses_history=True):
    observation = policy_observation(policy, env, observation, uses_history)
    return policy.predict(observation, deterministic=True)[0]


def main():
    parser = argparse.ArgumentParser(description='Compare TH14 policies in rotating blocks under the same native game conditions.')
    parser.add_argument('--uniform', type=Path, required=True)
    parser.add_argument('--title-weighted', type=Path, required=True)
    parser.add_argument('--reference-name', default='sft_uniform')
    parser.add_argument('--candidate-name', default='sft_title_weighted')
    parser.add_argument('--reference-history', action='store_true')
    parser.add_argument('--title-weighted-history', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--blocks', type=int, default=4)
    parser.add_argument('--limit', type=int, default=3200)
    parser.add_argument('--max-misses', type=int, choices=(1, 2), default=2)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.blocks, args.limit) < 1:
        parser.error('Use positive block and decision limits.')
    if len({'stationary', args.reference_name, args.candidate_name}) != 3:
        parser.error('Use distinct policy names.')
    torch.set_num_threads(1)
    device = torch.device(args.device)
    uniform = read_policy(args.uniform, device, args.reference_history)
    weighted = read_policy(args.title_weighted, device, args.title_weighted_history)
    motion_history = args.reference_history or args.title_weighted_history

    strategies = {
        'stationary': lambda _: np.array([4, 0, 1, 0]),
        args.reference_name: lambda obs: stage_action(uniform, env, obs, args.reference_history),
        args.candidate_name: lambda obs: stage_action(weighted, env, obs, args.title_weighted_history),
    }
    report = {
        'task': 'TH14 1.00b, Reimu A, Normal Stage 1',
        'max_misses': args.max_misses, 'decision_limit': args.limit, 'blocks_planned': args.blocks,
        'reference_has_motion_history': args.reference_history,
        'title_weighted_has_motion_history': args.title_weighted_history,
        'checkpoint_sha256': {args.reference_name: sha256(args.uniform.read_bytes()).hexdigest(),
                              args.candidate_name: sha256(args.title_weighted.read_bytes()).hexdigest()},
        'episodes': {name: [] for name in strategies},
    }
    if args.out.exists():
        if not args.resume:
            parser.error('Use a new report or explicitly resume the existing report.')
        saved = json.loads(args.out.read_text(encoding='utf-8'))
        constants = ('task', 'max_misses', 'decision_limit', 'blocks_planned', 'checkpoint_sha256')
        if (any(saved[key] != report[key] for key in constants)
                or saved.get('reference_has_motion_history', False) != args.reference_history
                or saved.get('title_weighted_has_motion_history', False) != args.title_weighted_history):
            parser.error('The saved evaluation uses a different task, schedule or checkpoint.')
        report = saved
    names = list(strategies)
    longest_history = max(*(uniform.motion_spans or [12]), *(weighted.motion_spans or [12]))
    env = NormalStageOne(Path(__file__).resolve().parents[2] / 'parakit', limit=args.limit,
                         max_misses=args.max_misses, motion_history=motion_history,
                         history_frames=longest_history)
    try:
        for block in range(args.blocks):
            pivot = block % len(names)
            for name in names[pivot:] + names[:pivot]:
                if any(episode['block'] == block + 1 for episode in report['episodes'][name]):
                    continue
                episode = dict(run_episode(env, strategies[name]), block=block + 1)
                report['episodes'][name].append(episode)
                report['summary'] = {key: summarize(rows) for key, rows in report['episodes'].items()}
                temporary = args.out.with_suffix('.tmp')
                temporary.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
                temporary.replace(args.out)
                print(name, json.dumps(episode), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
