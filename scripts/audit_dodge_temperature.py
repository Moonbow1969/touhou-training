"""Measure the behavior of the cloned dodge logits before native PPO."""

import argparse
from pathlib import Path

import numpy as np
import torch

from scripts.evaluate_stage_policies import read_policy
from scripts.pretrain_policy import append_motion_history, condition_context
from scripts.record_run import write_json
from scripts.train_stage_dodge import DodgeHead, candidate_features


@torch.inference_mode()
def scores(base, head, paths):
    logits, margins = [], []
    for path in paths:
        with np.load(path) as shard:
            context = append_motion_history(condition_context(shard['context'], 1, 'ReimuA'), shard['stage_frame'])
            proposals = base.predict({'bullets': shard['bullets'], 'context': context}, deterministic=True)[0]
            clearances = shard['action_clearances'].min(axis=1)
        features = np.stack([candidate_features(action, clearance) for action, clearance in zip(proposals, clearances)])
        logits.append(head(torch.from_numpy(features)))
        margins.append(torch.from_numpy(clearances.reshape(-1, 18)))
    return torch.cat(logits), torch.cat(margins)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--head', type=Path, required=True)
    parser.add_argument('--captures', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Use a fresh report path.')
    torch.set_num_threads(1)
    base = read_policy(args.base, 'cpu', motion_history=True)
    if base.stage_clock or base.motion_spans:
        parser.error('This diagnostic uses an immediate-history TH14 Normal Reimu A base without a stage clock.')
    head = DodgeHead().eval()
    head.load_state_dict(torch.load(args.head, map_location='cpu', weights_only=True))
    logits, margins = scores(base, head, args.captures)
    risky = (margins >= 0).any(1) & (margins < 4).any(1)
    if not risky.any():
        raise ValueError('The captures need mixed-clearance candidate sets.')
    report = {'states': len(logits), 'risky_states': int(risky.sum()), 'temperatures': {}}
    for temperature in (1, 2, 3, 4, 6, 8, 12):
        distribution = torch.distributions.Categorical(logits=logits / temperature)
        dangerous_mass = (distribution.probs * (margins < 0)).sum(1)
        report['temperatures'][temperature] = {
            'entropy': float(distribution.entropy().mean()),
            'risky_state_entropy': float(distribution.entropy()[risky].mean()),
            'risky_state_negative_clearance_probability': float(dangerous_mass[risky].mean()),
            'deterministic_directions_unchanged': bool((logits.argmax(1) == (logits / temperature).argmax(1)).all()),
        }
    write_json(args.out, report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
