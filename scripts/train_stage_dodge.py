import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from scripts.evaluate_stage_policies import read_policy
from scripts.pretrain_policy import batch_block, index_split, predict_heads, read_shard
from scripts.stage_risk import action_clearances, suggest_dodge


def candidate_features(action, clearances):
    move, focus = map(int, action[:2])
    margin = clearances.ravel()
    selected = 9 * focus + move
    ids = np.arange(18)
    current = margin[selected]
    full = np.full_like(margin, np.clip(current, -16, 32) / 16)
    return np.stack((np.clip(margin, -16, 32) / 16, full,
                     np.clip(margin - current, -32, 32) / 16,
                     ids == selected, ids // 9 == focus), axis=-1).astype(np.float32)


class DodgeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(5, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, features):
        return self.score(features).squeeze(-1)


def learned_action(model, action, clearances):
    with torch.inference_mode():
        scores = model(torch.from_numpy(candidate_features(action, clearances)))
    focus, move = divmod(int(scores.argmax()), 9)
    proposal = action.copy()
    proposal[:2] = (move, focus)
    return proposal


def sample_states(directory, split, maximum, rng, policy, device):
    shards = [read_shard((path, meta), True) for path, meta in index_split(directory, split) if meta['title'] == 14]
    if not shards:
        raise ValueError(f'No TH14 replay data in {split}.')
    combined = tuple(np.concatenate(fields) for fields in zip(*shards))
    selection = rng.choice(len(combined[2]), min(maximum, len(combined[2])), replace=False)
    for bullets, contexts, _ in batch_block(combined, 256, device, selection):
        with torch.inference_mode():
            logits = predict_heads(policy, bullets, contexts)
            actions = torch.stack([head.argmax(dim=1) for head in logits], dim=1).cpu().numpy()
        yield from zip(bullets.cpu().numpy(), contexts.cpu().numpy(), actions)


def examples(states, rng):
    result = []
    for bullets, context, action in states:
        clearance = action_clearances({'bullets': bullets, 'context': context}, 1.0)
        randomized = action.copy()
        randomized[:2] = (rng.integers(9), rng.integers(2))
        for prior in (action, randomized):
            target, _, _ = suggest_dodge(prior, clearance)
            result.append((candidate_features(prior, clearance), 9 * target[1] + target[0],
                           bool((target[:2] != prior[:2]).any())))
    return result


def demonstrated_corrections(path):
    result = []
    with np.load(path) as data:
        for bullets, context, prior, target in zip(data['bullets'], data['context'],
                                                  data['original_action'], data['action']):
            clearance = action_clearances({'bullets': bullets, 'context': context}, 1.0)
            result.append((candidate_features(prior, clearance), 9 * target[1] + target[0], True))
    return result


def dataset(rows):
    return TensorDataset(*(torch.from_numpy(np.stack(field)) for field in zip(*rows)))


def main():
    parser = argparse.ArgumentParser(description='Train a neural action selector from replay and closed-loop dodge corrections.')
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--corrections', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('Epochs must be positive.')
    torch.manual_seed(203)
    torch.set_num_threads(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(203)
    base = read_policy(args.base, device, motion_history=True)
    replay_train = examples(sample_states(args.data, 'train', 4500, rng, base, device), rng)
    corrections = demonstrated_corrections(args.corrections)
    replay_validation = examples(sample_states(args.data, 'validation', 700, rng, base, device), rng)
    train = dataset(replay_train + corrections * 12)
    validation = dataset(replay_validation)
    inputs, labels, changed = (t.to(device) for t in validation.tensors)
    model = DodgeHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    args.out.mkdir(parents=True, exist_ok=False)
    best = float('inf')
    epochs = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y, changed_action in DataLoader(train, batch_size=192, shuffle=True):
            x, y = x.to(device), y.to(device)
            weights = 1 + changed_action.to(device).float()
            loss = (F.cross_entropy(model(x), y, reduction='none') * weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            predictions = model(inputs)
            correct = predictions.argmax(dim=1) == labels
            val_loss = float(F.cross_entropy(predictions, labels))
        row = {'epoch': epoch, 'validation_loss': val_loss, 'all_accuracy': float(correct.float().mean()),
               'correction_accuracy': float(correct[changed].float().mean()) if changed.any() else None}
        epochs.append(row)
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), args.out / 'head.pt')
    report = {'replay_training_decisions': len(replay_train), 'on_policy_corrections': len(corrections),
              'validation_decisions': len(validation), 'validation_corrections': int(changed.sum()),
              'best': min(epochs, key=lambda row: row['validation_loss']), 'epochs': epochs}
    (args.out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print({key: value for key, value in report.items() if key != 'epochs'}, flush=True)


if __name__ == '__main__':
    main()
