import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.evaluate_stage_policies import read_policy
from scripts.pretrain_policy import batch_block, decode_movement, index_split, predict_heads, read_shard
from scripts.replay_risk import collision_mask


def prior_direction(context):
    velocity = context[:, -9:-7]
    dx, dy = np.sign(np.where(abs(velocity) > 0.01, velocity, 0)).T.astype(np.int64)
    return (dy + 1) * 3 + dx + 1


@torch.inference_mode()
def evaluate(shards, policies, device, geometry_eval=False):
    counts = {}
    risk_counts = {}
    stages = []
    for shard in shards:
        path, meta = shard
        data = read_shard(shard, motion_history=True)
        context, actions = data[1:]
        truth = actions[:, 0]
        previous = prior_direction(context)
        samples = {
            'movement': np.ones(len(truth), dtype=bool),
            'while_moving': truth != 4,
            'direction_change_to_motion': (context[:, -7] > 0) & (truth != 4) & (truth != previous),
            'start_moving': (context[:, -7] > 0) & (truth != 4) & (previous == 4),
        }
        predictions = {'keep_last_direction': previous}
        focus_predictions = {'keep_last_direction': (context[:, 2] > .5).astype(int)}
        if geometry_eval:
            masks = np.concatenate([collision_mask(bullets, ctx).cpu().numpy()
                                    for bullets, ctx, _ in batch_block(data, 256, device, np.arange(len(truth)))])
            indices = np.arange(len(truth))
            eligible = masks.any(axis=(1, 2)) & ~masks[indices, actions[:, 1], truth]
        by_motion_spans = {(None, False, False): data}
        for name, (policy, spans, retained) in policies.items():
            key = spans, retained, policy.stage_clock
            if key not in by_motion_spans:
                by_motion_spans[key] = read_shard(shard, motion_history=True, motion_spans=spans,
                                                  retain_recent_motion=retained, stage_clock=policy.stage_clock)
            direction_batches, focus_batches = [], []
            for bullets, ctx, _ in batch_block(by_motion_spans[key], 512, device, np.arange(len(truth))):
                movement, focus, _, _ = predict_heads(policy, bullets, ctx)
                direction_batches.append(decode_movement(movement, policy.factorized_movement).cpu().numpy())
                if geometry_eval:
                    focus_batches.append(focus.argmax(1).cpu().numpy())
            predictions[name] = np.concatenate(direction_batches)
            if geometry_eval:
                focus_predictions[name] = np.concatenate(focus_batches)
        stage_counts = {}
        for label, prediction in predictions.items():
            correct = prediction == truth
            stage_counts[label] = {
                key: (int(correct[mask].sum()), int(mask.sum())) for key, mask in samples.items()
            }
            title = f'TH{meta["title"]}'
            rank = f'{title}/{meta["difficulty"]}'
            for group in ('overall', title, rank, f'{rank}/{meta["shot"]}'):
                group_counts = counts.setdefault(group, {}).setdefault(label, {})
                for key, (wins, total) in stage_counts[label].items():
                    summed = group_counts.setdefault(key, [0, 0])
                    summed[0] += wins
                    summed[1] += total
            if geometry_eval:
                total = risk_counts.setdefault(label, [0, 0])
                total[0] += int((masks[indices, focus_predictions[label], prediction] & eligible).sum())
                total[1] += int(eligible.sum())
        stages.append({'shard': path.stem, **meta, 'counts': stage_counts})
    summary = {
        group: {
            policy: {key: {'accuracy': wins / total if total else None, 'n': total}
                     for key, (wins, total) in metric.items()}
            for policy, metric in group_counts.items()
        }
        for group, group_counts in counts.items()
    }
    result = {'summary': summary, 'stages': stages}
    if geometry_eval:
        result['geometry'] = {label: {'predicted_collisions': collisions, 'eligible_decisions': count,
                                      'predicted_collision_fraction': collisions / count if count else None}
                              for label, (collisions, count) in risk_counts.items()}
    return result


def main():
    parser = argparse.ArgumentParser(description='Compare motion policies on identical held-out replay labels.')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--additional', nargs=2, action='append', metavar=('NAME', 'CHECKPOINT'), default=[])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--holdout-title', type=int,
                        help='Evaluate this title and verify the candidate did not use it for training or selection.')
    parser.add_argument('--geometry-eval', action='store_true', help='Also compare predicted four-tick collisions.')
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    choices = [('original', args.reference), ('corrected_data', args.candidate)]
    choices.extend((label, Path(path)) for label, path in args.additional)
    if len({label for label, _ in choices}) != len(choices) or any(label == 'keep_last_direction' for label, _ in choices):
        parser.error('Use distinct policy names that do not conflict with the baseline.')
    if args.holdout_title is not None:
        manifest = args.candidate.with_name('metrics.json')
        if not manifest.is_file():
            parser.error('The holdout candidate needs its metrics sidecar.')
        metrics = json.loads(manifest.read_text(encoding='utf-8'))
        for key in ('training_replays', 'validation_replays'):
            rows = metrics.get(key)
            if not rows or any(row['title'] == args.holdout_title for row in rows):
                parser.error(f'{key} must document a split that excludes the held-out title.')
    policies = {}
    for label, path in choices:
        policy = read_policy(path, device, motion_history=True)
        policies[label] = (policy, tuple(policy.motion_spans) if policy.motion_spans else None,
                           policy.retain_recent_motion)
    report = {
        'reference': str(args.reference), 'candidate': str(args.candidate),
        'models': {label: {'checkpoint': str(path), 'motion_spans': policies[label][1],
                           'retain_recent_motion': policies[label][2],
                           'stage_clock': policies[label][0].stage_clock} for label, path in choices},
        'holdout_title': args.holdout_title,
        **{split: evaluate([shard for shard in index_split(args.data, split)
                            if args.holdout_title is None or shard[1]['title'] == args.holdout_title], policies, device,
                           args.geometry_eval)
           for split in ('validation', 'test')},
    }
    args.out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({split: report[split]['summary']['overall'] for split in ('validation', 'test')}),
          flush=True)


if __name__ == '__main__':
    main()
