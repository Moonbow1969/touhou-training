import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.evaluate_replay_policies import prior_direction
from scripts.evaluate_stage_policies import read_policy
from scripts.pretrain_policy import batch_block, index_split, predict_heads, read_shard


@torch.inference_mode()
def evaluate_logits(data, models, split, device):
    features, logits, reference = [], [], []
    for shard in index_split(data, split):
        arrays = read_shard(shard, motion_history=True)
        context, actions = arrays[1:]
        features.append(np.stack((actions[:, 0], prior_direction(context), context[:, -7] > 0), axis=-1))
        for bullets, ctx, _ in batch_block(arrays, 512, device, np.arange(len(actions))):
            logits.append(predict_heads(models[0], bullets, ctx)[0].cpu().numpy())
            reference.append(predict_heads(models[1], bullets, ctx)[0].argmax(-1).cpu().numpy())
    truth, previous, valid = np.concatenate(features).T.astype(np.int64)
    return truth, previous, valid.astype(bool), np.concatenate(logits), np.concatenate(reference)


def apply_bias(logits, previous, valid, idle_bias, persistence_bias):
    adjusted = logits.copy()
    adjusted[:, 4] += idle_bias
    eligible = np.flatnonzero(valid & (previous != 4))
    adjusted[eligible, previous[eligible]] += persistence_bias
    return adjusted.argmax(axis=-1)


def metrics(truth, previous, valid, decisions):
    correct = decisions == truth
    moving = truth != 4
    changing = valid & (truth != previous) & moving
    starting = changing & (previous == 4)
    return {
        'decisions': len(truth), 'movement_accuracy': float(correct.mean()),
        'moving_accuracy': float(correct[moving].mean()),
        'direction_change_accuracy': float(correct[changing].mean()),
        'start_moving_accuracy': float(correct[starting].mean()),
        'direction_change_count': int(changing.sum()),
    }


def choose(validation, maximum_drop):
    truth, previous, valid, logits, reference = validation
    reference_score = metrics(truth, previous, valid, reference)['movement_accuracy']
    threshold = reference_score - maximum_drop
    grid = np.arange(-.25, 2.26, .125)
    available = []
    for idle_bias in grid:
        for persistence_bias in grid:
            decisions = apply_bias(logits, previous, valid, idle_bias, persistence_bias)
            evaluation = metrics(truth, previous, valid, decisions)
            available.append((float(idle_bias), float(persistence_bias), evaluation))
    feasible = [x for x in available if x[2]['movement_accuracy'] >= threshold]
    if feasible:
        chosen = max(feasible, key=lambda x: (x[2]['direction_change_accuracy'], x[2]['movement_accuracy']))
    else:
        chosen = max(available, key=lambda x: x[2]['movement_accuracy'])
    return chosen, threshold, len(feasible)


def main():
    parser = argparse.ArgumentParser(description='Calibrate a direction-weighted classifier using validation replays.')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-validation-drop', type=float, default=.01)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device('cuda')
    models = [read_policy(p, device, motion_history=True) for p in (args.policy, args.reference)]
    if any(model.motion_spans or model.stage_clock for model in models):
        parser.error('Use models with the same standard recent motion input.')
    validation = evaluate_logits(args.data, models, 'validation', device)
    (idle, persistence, _), threshold, feasible = choose(validation, args.max_validation_drop)
    report = {
        'policy': str(args.policy), 'reference': str(args.reference), 'idle_logit_bias': idle,
        'continue_moving_logit_bias': persistence, 'validation_movement_floor': threshold,
        'feasible_validation_settings': feasible,
    }
    test = evaluate_logits(args.data, models, 'test', device)
    for split, (truth, previous, valid, logits, reference) in (('validation', validation), ('test', test)):
        report[split] = {
            'reference': metrics(truth, previous, valid, reference),
            'weighted': metrics(truth, previous, valid, logits.argmax(-1)),
            'calibrated': metrics(truth, previous, valid, apply_bias(logits, previous, valid, idle, persistence)),
        }
    args.out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
