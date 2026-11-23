import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from scripts.evaluate_boss_aim import AimHead, mask_aim_logits


def read_examples(paths):
    fields = []
    for path in paths:
        with np.load(path) as shard:
            fields.append([shard[key] for key in ('features', 'target', 'prior')])
    features, target, prior = (np.concatenate(items) for items in zip(*fields))
    return torch.from_numpy(features).float(), torch.from_numpy(target).long(), torch.from_numpy(prior).long()


@torch.inference_mode()
def metrics(model, examples, mask_training):
    features, targets, prior = examples
    correction = targets != prior
    logits = model(features)
    masked = mask_aim_logits(logits, features)
    loss_logits = masked if mask_training else logits
    weights = 1 + 2 * correction
    loss = float((F.cross_entropy(loss_logits, targets, reduction='none') * weights).mean())
    direction = masked.argmax(-1)
    matched = direction == targets
    actual_change = direction != prior
    usable = features[torch.arange(len(features)), direction, 1] >= 0
    return {'rows': len(features), 'corrections': int(correction.sum()), 'loss': loss,
            'teacher_agreement': float(matched.float().mean()),
            'correction_agreement': float(matched[correction].float().mean()) if correction.any() else None,
            'changes_within_clearance_budget': float(usable[actual_change].float().mean()) if actual_change.any() else None}


def main():
    parser = argparse.ArgumentParser(description='Distill the boss alignment teacher on native controller trajectories.')
    parser.add_argument('--train', type=Path, nargs='+', required=True)
    parser.add_argument('--validation', type=Path, nargs='+', required=True)
    parser.add_argument('--test', type=Path, nargs='+', help='Evaluate this split after selecting the best checkpoint.')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=45)
    parser.add_argument('--mask-training', action='store_true', help='Use the inference clearance constraint in the loss.')
    args = parser.parse_args()
    paths = [p.resolve() for split in (args.train, args.validation, args.test or []) for p in split]
    if args.epochs < 1 or len(set(paths)) != len(paths):
        parser.error('Use positive epochs and separate native episodes for training and validation.')
    torch.set_num_threads(1)
    torch.manual_seed(972)
    train = read_examples(args.train)
    validation = read_examples(args.validation)
    model = AimHead()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    report = {'training_rows': len(train[0]), 'validation_rows': len(validation[0]),
              'training_corrections': int((train[1] != train[2]).sum()),
              'validation_corrections': int((validation[1] != validation[2]).sum()),
              'mask_training': args.mask_training, 'epochs': []}
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'aim.pt').exists() or (args.out / 'metrics.json').exists():
        parser.error('Choose a fresh output directory for training.')
    best = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y, previous in DataLoader(TensorDataset(*train), batch_size=96, shuffle=True):
            weights = 1 + 2 * (y != previous)
            logits = model(x)
            if args.mask_training:
                logits = mask_aim_logits(logits, x)
            loss = (F.cross_entropy(logits, y, reduction='none') * weights).mean()
            if not torch.isfinite(loss):
                raise ValueError('Non-finite aim loss; check the teacher targets against the clearance mask.')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        row = {'epoch': epoch, **metrics(model, validation, args.mask_training)}
        report['epochs'].append(row)
        if row['loss'] < best:
            best = row['loss']
            report['best'] = row
            torch.save(model.state_dict(), args.out / 'aim.pt')
    if args.test:
        model.load_state_dict(torch.load(args.out / 'aim.pt', map_location='cpu', weights_only=True))
        report['test'] = metrics(model, read_examples(args.test), args.mask_training)
    (args.out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print({key: value for key, value in report.items() if key != 'epochs'}, flush=True)


if __name__ == '__main__':
    main()
