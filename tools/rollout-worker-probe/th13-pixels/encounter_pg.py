"""Fit one trust-region-bounded REINFORCE step from adjudicated native encounters."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from policy_credit import features


def probabilities(features_, weights):
    logits = features_ @ weights.T
    values = np.exp(logits - logits.max(axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)


def raw_hash(weights):
    return hashlib.sha256(weights.astype('<f8').tobytes()).hexdigest()


def read_credit(folder, weights):
    credit = json.loads((folder / 'credit.json').read_text(encoding='utf8'))
    label = json.loads((folder / 'adjudication.json').read_text(encoding='utf8'))
    if credit['weights_sha256'] != raw_hash(weights):
        raise ValueError(f'Unexpected behavior checkpoint in {folder}')
    if label['endpoint'] not in ('post-countdown', 'ranking'):
        raise ValueError(f'Unqualified encounter endpoint in {folder}')
    with np.load(folder / 'behavior.npz') as data:
        elapsed = (data['action_onset_ns'] - data['monotonic_ns'][0]) * 1e-9
        keep = elapsed <= label['endpoint_wall_s']
        action = data['action'][keep]
        observed = features(data['obs'][:-1])[keep]
        probability = data['prob'][keep]
        elapsed = elapsed[keep]
    if not np.allclose(probability, probabilities(observed, weights), atol=2e-9, rtol=2e-9):
        raise ValueError(f'Logged probabilities differ in {folder}')
    remaining = np.zeros(len(action))
    phase = np.zeros(len(action), dtype=int)
    for event in label['events']:
        before = elapsed <= event['wall_s']
        remaining[before] += event['reward']
        if event['kind'] in ('entered timed encounter', 'half encounter'):
            phase[~before] += 1
    return observed, action, probability, remaining, phase


def fit(weights, credits):
    gradients = []
    for i, (observed, action, probability, remaining, phase) in enumerate(credits):
        phase_baseline = []
        for k in range(3):
            values = [
                other_return[other_phase == k].mean()
                for j, (_, _, _, other_return, other_phase) in enumerate(credits)
                if j != i and (other_phase == k).any()
            ]
            phase_baseline.append(np.mean(values) if values else 0.0)
        advantage = remaining - np.array(phase_baseline)[phase]
        logit_gradient = (np.eye(3)[action] - probability) * advantage[:, None]
        gradients.append(logit_gradient.T @ observed / len(action))
    gradient = np.mean(gradients, axis=0)
    norm = np.linalg.norm(gradient)
    if norm < 1e-8:
        raise ValueError('No measurable gradient')
    direction = gradient / norm
    observed = np.vstack([credit[0] for credit in credits])
    old = probabilities(observed, weights)

    def forward_kl(step):
        current = probabilities(observed, weights + step * direction)
        return (old * np.log(old / current)).sum(axis=1).mean()

    low, high = 0.0, 2.0
    for _ in range(45):
        mid = (low + high) / 2
        if forward_kl(mid) > 0.006:
            high = mid
        else:
            low = mid
    return weights + low * direction, float(forward_kl(low))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial', type=Path, help='Initial actor for a fresh fit')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--train-credit', type=Path, action='append')
    source.add_argument('--reduction', type=Path, help='Rebuild the frozen actor from public per-credit gradients')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.reduction:
        with np.load(args.reduction) as reduced:
            gradient = reduced['per_credit_gradient'].mean(axis=0)
            direction = gradient / np.linalg.norm(gradient)
            new = reduced['parent_weights'] + reduced['accepted_step_frobenius'] * direction
        np.savez(args.output, weights=new)
        print(json.dumps({'candidate_raw_sha256': raw_hash(new)}))
        return
    if args.initial is None:
        parser.error('--initial is required when fitting training credits')
    if len(set(path.resolve() for path in args.train_credit)) != len(args.train_credit) or len(args.train_credit) < 3:
        parser.error('Supply at least three distinct whole training credits')
    weights = np.load(args.initial)['weights']
    new, kl = fit(weights, [read_credit(folder, weights) for folder in args.train_credit])
    np.savez(args.output, weights=new)
    print(json.dumps({'candidate_raw_sha256': raw_hash(new), 'forward_kl': kl}))


if __name__ == '__main__':
    main()
