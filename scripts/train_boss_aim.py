import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from scripts.evaluate_boss_aim import AimHead


def read_examples(paths):
    fields = []
    for path in paths:
        with np.load(path) as shard:
            fields.append([shard[key] for key in ('features', 'target', 'prior')])
    features, target, prior = (np.concatenate(items) for items in zip(*fields))
    return torch.from_numpy(features).float(), torch.from_numpy(target).long(), torch.from_numpy(prior).long()


def main():
    parser = argparse.ArgumentParser(description='Distill the boss alignment teacher on native controller trajectories.')
    parser.add_argument('--train', type=Path, nargs='+', required=True)
    parser.add_argument('--validation', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=45)
    args = parser.parse_args()
    if args.epochs < 1 or {p.resolve() for p in args.train} & {p.resolve() for p in args.validation}:
        parser.error('Use positive epochs and separate native episodes for training and validation.')
    torch.set_num_threads(1)
    torch.manual_seed(972)
    train = read_examples(args.train)
    validation = read_examples(args.validation)
    features, targets, prior = validation
    correction = targets != prior
    model = AimHead()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    report = {'training_rows': len(train[0]), 'validation_rows': len(features),
              'training_corrections': int((train[1] != train[2]).sum()),
              'validation_corrections': int(correction.sum()), 'epochs': []}
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'aim.pt').exists() or (args.out / 'metrics.json').exists():
        parser.error('Choose a fresh output directory for training.')
    best = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y, previous in DataLoader(TensorDataset(*train), batch_size=96, shuffle=True):
            weights = 1 + 2 * (y != previous)
            loss = (F.cross_entropy(model(x), y, reduction='none') * weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            logits = model(features)
            weights = 1 + 2 * correction
            loss = float((F.cross_entropy(logits, targets, reduction='none') * weights).mean())
            direction = logits.argmax(-1)
            matched = direction == targets
            actual_change = direction != prior
            usable = features[torch.arange(len(features)), direction, 1] >= 0
        row = {'epoch': epoch, 'loss': loss, 'teacher_agreement': float(matched.float().mean()),
               'correction_agreement': float(matched[correction].float().mean()) if correction.any() else None,
               'changes_within_clearance_budget': float(usable[actual_change].float().mean()) if actual_change.any() else None}
        report['epochs'].append(row)
        if loss < best:
            best = loss
            report['best'] = row
            torch.save(model.state_dict(), args.out / 'aim.pt')
    (args.out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print({key: value for key, value in report.items() if key != 'epochs'}, flush=True)


if __name__ == '__main__':
    main()
